"""
Pitcher strikeout prop model.

Target: will pitcher throw OVER or UNDER the book's K line for this start?

This is a binary classification problem at its core (over/under), but we
first build a regression model to estimate the expected K total, then
compare to the line to get a probability.

Features per start:
  Pitcher rolling stats (last 5 starts, no look-ahead):
    - k_per_9_l5      : K/9 in last 5 starts
    - k_per_9_season  : season K/9 to date
    - swstr_pct       : swinging strike rate (season average)
    - csw_pct         : called strike + whiff rate
    - ip_per_start    : avg IP in last 5 starts (innings-pitched ceiling)
    - pitches_per_start : avg pitch count (how deep does he go?)

  Opponent:
    - opp_k_pct       : opposing lineup K% (season avg)

  Context:
    - is_home         : home/away (home pitchers get more run support, fewer IP)
    - park_k_factor   : park factor for strikeouts (some parks boost Ks)
    - line            : the book's over/under line (normalised signal)

Walk-forward validation: train on seasons S1..SN-1, predict SN.
"""

from __future__ import annotations

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sqlalchemy import text
from scipy.stats import poisson
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.schema import get_engine, get_session, PitcherGameLog

MIN_START_IP = 3.0

MODEL_DIR = Path(__file__).parent.parent / "model"

# NOTE (2026-06-10): the previous hand-typed PARK_K_FACTOR table was removed.
# It was applied as a multiplier at predict time only — the Ridge model was
# never trained with it, so it double-counted anything park-correlated in the
# features, covered only 15 of 30 parks, and the values were guesses. If park
# K effects are worth modelling, add a park feature to FEATURE_COLS_K and
# retrain + walk-forward validate.


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_starter_features(
    session,
    mlbam_id: int,
    game_date: str,
    n_lookback: int = 5,
) -> dict | None:
    """
    Build features for a single pitcher start using only data
    available *before* that game date.
    """
    prior_starts = (
        session.query(PitcherGameLog)
        .filter(
            PitcherGameLog.mlbam_id == mlbam_id,
            PitcherGameLog.game_date < game_date,
        )
        .order_by(PitcherGameLog.game_date.desc())
        .all()
    )

    if len(prior_starts) < 3:
        return None   # not enough history

    recent = prior_starts[:n_lookback]
    season_starts = prior_starts  # all prior starts this season + prior

    def safe_mean(lst):
        vals = [x for x in lst if x is not None]
        return float(np.mean(vals)) if vals else None

    # Rolling last-N stats
    recent_ks   = [s.strikeouts or 0 for s in recent]
    recent_ip   = [s.ip or 0 for s in recent]
    recent_pit  = [s.pitches or 0 for s in recent]

    k_per_9_l5 = (
        sum(recent_ks) / sum(recent_ip) * 9
        if sum(recent_ip) > 0 else None
    )
    k_per_pitch_l5 = (
        sum(recent_ks) / sum(recent_pit)
        if sum(recent_pit) > 0 else None
    )
    ip_per_start_l5  = safe_mean(recent_ip)
    pit_per_start_l5 = safe_mean(recent_pit)

    # Season-to-date K/9
    all_ks = [s.strikeouts or 0 for s in season_starts]
    all_ip = [s.ip or 0 for s in season_starts]
    k_per_9_season = sum(all_ks) / sum(all_ip) * 9 if sum(all_ip) > 0 else None

    return {
        "k_per_9_l5":       k_per_9_l5,
        "k_per_9_season":   k_per_9_season,
        "k_per_pitch_l5":   k_per_pitch_l5,
        "ip_per_start_l5":  ip_per_start_l5,
        "pit_per_start_l5": pit_per_start_l5,
    }


def build_opp_k9_cache(engine) -> dict[tuple[str, int, str], float]:
    """
    Build a rolling season-to-date opponent K9 cache using game_pk to
    identify the actual opponent team (the pitcher_game_logs.opponent field
    is unreliable; joining games is authoritative).

    Returns {(opponent_abbr, season, game_date): opp_k9_before_that_date}
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT "
                "  CASE WHEN p.home_away='home' THEN g.away_team ELSE g.home_team END AS opp, "
                "  p.season, p.game_date, p.strikeouts, p.ip "
                "FROM pitcher_game_logs p "
                "JOIN games g ON p.game_pk = g.game_pk "
                "WHERE p.strikeouts IS NOT NULL AND p.ip >= :min_ip "
                "ORDER BY p.season, p.game_date"
            ),
            {"min_ip": MIN_START_IP},
        ).fetchall()

    by_opp_season: dict[tuple[str, int], list[tuple[str, float, float]]] = defaultdict(list)
    for r in rows:
        by_opp_season[(r.opp, r.season)].append((r.game_date, r.strikeouts, r.ip))

    cache: dict[tuple[str, int, str], float] = {}
    for (opp, season), starts in by_opp_season.items():
        starts_sorted = sorted(starts, key=lambda x: x[0])
        cum_k = cum_ip = 0.0
        for date, ks, ip in starts_sorted:
            if cum_ip >= 9.0:
                cache[(opp, season, date)] = cum_k / cum_ip * 9.0
            cum_k += ks
            cum_ip += ip

    return cache


def build_prior_swstr_cache(engine) -> dict[tuple[int, int], tuple[float, float]]:
    """
    {(mlbam_id, season): (swstr_pct, csw_pct)} — season-level Statcast
    aggregates stored in pitcher_game_logs (constant across a season's rows).
    Look up (pid, season - 1) to get the leakage-safe prior-season value.
    """
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT mlbam_id, season, MAX(swstr_pct) AS sw, MAX(csw_pct) AS csw "
            "FROM pitcher_game_logs WHERE swstr_pct IS NOT NULL "
            "GROUP BY mlbam_id, season"
        )).fetchall()
    return {
        (int(r.mlbam_id), int(r.season)): (float(r.sw), float(r.csw) if r.csw is not None else None)
        for r in rows
    }


def build_opp_lineup_kpct_cache(
    engine,
    min_lineup_pa: int = 500,
) -> dict[tuple[int, str], float]:
    """
    {(game_pk, pitcher_side): opposing lineup K%}.

    For each game, the opposing lineup is the set of batters who appeared
    for the other team; their K% is total prior-season-to-date strikeouts /
    PA, strictly before the game date (no look-ahead on the rates).

    Caveat: using the batters who *actually* appeared is a mild proxy for
    the confirmed lineup known at bet time — at live pick time the confirmed
    lineup is exact, so the live feature is cleaner than the training one.

    Only populated where batter_game_logs has data (2023+); earlier starts
    fall back to median fill with opp_lineup_data = 0.
    """
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT season, game_date, game_pk, home_away, mlbam_id, pa, strikeouts "
            "FROM batter_game_logs WHERE pa > 0 "
            "ORDER BY season, game_date"
        )).fetchall()

    cache: dict[tuple[int, str], float] = {}
    cum: dict[tuple[int, int], list] = {}     # (season, mlbam_id) -> [pa, k]
    prev_season: int | None = None

    i, n = 0, len(rows)
    while i < n:
        season, date = int(rows[i].season), rows[i].game_date
        if season != prev_season:
            cum = {}
            prev_season = season
        j = i
        while j < n and rows[j].game_date == date and int(rows[j].season) == season:
            j += 1
        day = rows[i:j]

        # Lineups per (game, side) from today's appearances
        by_game: dict[tuple[int, str], list[int]] = {}
        for r in day:
            if r.game_pk and r.home_away in ("home", "away"):
                by_game.setdefault((int(r.game_pk), r.home_away), []).append(int(r.mlbam_id))

        # Compute lineup K% from stats accumulated BEFORE today
        for (pk, batter_side), pids in by_game.items():
            tot_pa = tot_k = 0
            for pid in pids:
                st = cum.get((season, pid))
                if st:
                    tot_pa += st[0]
                    tot_k  += st[1]
            if tot_pa >= min_lineup_pa:
                # home batters face the AWAY pitcher and vice versa
                pitcher_side = "away" if batter_side == "home" else "home"
                cache[(pk, pitcher_side)] = tot_k / tot_pa

        # Update cumulative totals with today's games
        for r in day:
            st = cum.setdefault((season, int(r.mlbam_id)), [0, 0])
            st[0] += r.pa or 0
            st[1] += r.strikeouts or 0
        i = j

    return cache


def build_training_dataset(sessions_list: list[int] | None = None) -> pd.DataFrame:
    """
    Build the full training dataset from pitcher game logs.
    Each row is one start with rolling features computed from prior starts.
    Target: strikeouts in this start.
    """
    engine  = get_engine()
    session = get_session(engine)

    # Pre-build caches (avoids N+1 queries)
    opp_k9_cache     = build_opp_k9_cache(engine)
    swstr_cache      = build_prior_swstr_cache(engine)
    opp_lineup_cache = build_opp_lineup_kpct_cache(engine)

    # League-average K9 fallback
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT SUM(strikeouts)*9.0/SUM(ip) FROM pitcher_game_logs WHERE ip >= :m"),
            {"m": MIN_START_IP},
        ).scalar()
    league_k9 = float(row) if row else 9.0

    # Build game_pk → opponent map (start.opponent field is unreliable)
    with engine.connect() as conn:
        pk_rows = conn.execute(text(
            "SELECT p.game_pk, p.home_away, g.home_team, g.away_team "
            "FROM pitcher_game_logs p "
            "JOIN games g ON p.game_pk = g.game_pk"
        )).fetchall()
    pk_to_opp = {
        r.game_pk: (r.away_team if r.home_away == "home" else r.home_team)
        for r in pk_rows
    }

    # Build game_pk → ump_k9_vs_avg lookup from umpire assignments
    with engine.connect() as conn:
        ump_k_rows = conn.execute(text("""
            SELECT u.game_pk, s.k9_vs_avg
            FROM   game_umpires u
            JOIN   umpire_stats s ON s.ump_name = u.ump_name
            WHERE  s.k9_vs_avg IS NOT NULL
        """)).fetchall()
    ump_k9_cache: dict[int, float] = {int(r.game_pk): float(r.k9_vs_avg) for r in ump_k_rows}

    query = session.query(PitcherGameLog).filter(
        PitcherGameLog.strikeouts.isnot(None),
        PitcherGameLog.ip >= MIN_START_IP,
    )
    if sessions_list:
        query = query.filter(PitcherGameLog.season.in_(sessions_list))

    all_starts = query.order_by(
        PitcherGameLog.mlbam_id, PitcherGameLog.game_date
    ).all()

    rows = []
    for start in all_starts:
        feats = build_starter_features(session, start.mlbam_id, start.game_date)
        if feats is None:
            continue

        actual_opp = pk_to_opp.get(start.game_pk, "")
        opp_k9 = opp_k9_cache.get(
            (actual_opp, start.season, start.game_date), league_k9
        )

        feats["opp_k9"]            = opp_k9
        feats["ump_k9_vs_avg"]     = ump_k9_cache.get(start.game_pk or -1, 0.0)

        # Prior-season whiff quality
        sw = swstr_cache.get((start.mlbam_id, start.season - 1))
        feats["swstr_prior"] = sw[0] if sw else None
        feats["csw_prior"]   = sw[1] if sw else None

        # Opposing lineup K%
        lk = opp_lineup_cache.get((start.game_pk or -1, start.home_away))
        feats["opp_lineup_k_pct"] = lk
        feats["opp_lineup_data"]  = 1.0 if lk is not None else 0.0

        feats["strikeouts_actual"] = start.strikeouts
        feats["mlbam_id"]          = start.mlbam_id
        feats["game_date"]         = start.game_date
        feats["game_pk"]           = start.game_pk
        feats["season"]            = start.season
        feats["is_home"]           = 1 if start.home_away == "home" else 0
        feats["opponent"]          = start.opponent
        rows.append(feats)

    session.close()
    return pd.DataFrame(rows)


FEATURE_COLS_K = [
    "k_per_9_l5",        # rolling K/9, last 5 starts
    "k_per_9_season",    # season-to-date K/9
    "k_per_pitch_l5",    # Ks per pitch last 5 starts (proxy for stuff/efficiency)
    "ip_per_start_l5",   # avg IP last 5 starts (K ceiling)
    "pit_per_start_l5",  # avg pitches last 5 starts
    "opp_k9",            # opponent team K/9 allowed, rolling season-to-date
    "is_home",
    "ump_k9_vs_avg",     # HP umpire career K9 vs league avg (0 when unavailable)

    # Whiff quality, PRIOR season (full-season Statcast aggregates — current-
    # season values in pitcher_game_logs are full-season numbers and would
    # leak the future, so only the prior season is safe to use).
    "swstr_prior",       # swinging-strike rate, prior season
    "csw_prior",         # called-strike-plus-whiff rate, prior season

    # Opposing lineup strikeout-proneness (PA-weighted season-to-date K% of
    # the batters who actually appeared; live picks use the confirmed lineup).
    "opp_lineup_k_pct",
    "opp_lineup_data",   # 1.0 when lineup K% was computable
]


# ---------------------------------------------------------------------------
# Over/under probability conversion
# ---------------------------------------------------------------------------

def over_under_probs(expected: float, line: float) -> tuple[float, float]:
    """
    Convert an expected count into (prob_over, prob_under) for a given line
    via a Poisson tail, handling integer lines correctly.

    For an integer line (e.g. 6.0), landing exactly on the line is a PUSH —
    stake returned, neither side wins. The bettable probabilities are
    therefore conditional on the push not happening:
        P(over)  = P(X > line) / (P(X > line) + P(X < line))
    For half lines (e.g. 5.5) the push probability is zero and this reduces
    to the plain tail probability.

    The old code computed prob_under = 1 - prob_over, silently folding the
    push probability into the under side and overstating its edge.

    Caveat: real K/hit distributions are bounded (pitch counts, ABs), so the
    Poisson tail is an approximation — validate calibration walk-forward
    before trusting prop edges.
    """
    import math
    mu = max(expected, 0.1)
    p_over  = 1.0 - poisson.cdf(math.floor(line), mu)            # P(X > line)
    p_under = float(poisson.cdf(math.ceil(line) - 1, mu))        # P(X < line)
    denom = p_over + p_under
    if denom <= 0:
        return 0.5, 0.5
    return p_over / denom, p_under / denom


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_k_regression() -> Pipeline:
    """Ridge regression to estimate expected K total."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=1.0)),
    ])


def train_k_model(verbose: bool = True) -> Pipeline:
    """Train the K regression model on all available data."""
    print("Building strikeout training dataset...")
    df = build_training_dataset()

    if df.empty:
        raise RuntimeError(
            "No pitcher game log data in DB. "
            "Run: python props/strikeout_data.py first."
        )

    # Fill missing features with column medians, and remember the medians so
    # live prediction fills missing values the same way (NOT with 0.0, which
    # for a rate feature is an extreme value, not a neutral one).
    medians: dict[str, float] = {}
    for col in FEATURE_COLS_K:
        if col in df.columns:
            med = df[col].median()
            medians[col] = float(med) if pd.notna(med) else 0.0
            df[col] = df[col].fillna(medians[col])

    X = df[FEATURE_COLS_K]
    y = df["strikeouts_actual"]

    model = make_k_regression()
    model.fit(X, y)

    preds = model.predict(X)
    mae   = mean_absolute_error(y, preds)
    print(f"  Training MAE: {mae:.2f} Ks  (in-sample, not walk-forward)")
    print(f"  Mean Ks/start: {y.mean():.2f}  Std: {y.std():.2f}")

    path = MODEL_DIR / "model_strikeouts.pkl"
    with open(path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": FEATURE_COLS_K,
            "medians": medians,
            "league_k9": float(df["k_per_9_season"].mean()) if "k_per_9_season" in df else 9.0,
        }, f)
    print(f"  Saved to {path}")

    return model


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validate() -> None:
    """
    Walk-forward validation of the K model: train on seasons <= N-1, test on N.

    Reports MAE versus a naive baseline AND binned calibration of the
    over-probability at common lines — for betting it is the probability
    calibration that matters, not the point estimate.
    """
    df = build_training_dataset()
    if df.empty:
        print("No data.")
        return

    for col in FEATURE_COLS_K:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)

    seasons = sorted(df["season"].unique())
    if len(seasons) < 3:
        print(f"Need >= 3 seasons for walk-forward; have {seasons}. "
              "MAE below is the last train/test split only.")

    print("\nWalk-forward validation (strikeout model):")
    naive = mean_absolute_error(
        df["strikeouts_actual"], [df["strikeouts_actual"].mean()] * len(df)
    )

    all_probs: list[float] = []
    all_overs: list[int] = []

    for i in range(2, len(seasons)):
        train_s = seasons[:i]
        test_s  = int(seasons[i])
        tr = df[df["season"].isin(train_s)]
        te = df[df["season"] == test_s]
        if tr.empty or te.empty:
            continue

        model = make_k_regression()
        model.fit(tr[FEATURE_COLS_K], tr["strikeouts_actual"])
        preds = model.predict(te[FEATURE_COLS_K])
        mae   = mean_absolute_error(te["strikeouts_actual"], preds)

        # Probability calibration at typical book lines
        line_accs = {}
        for line in [4.5, 5.5, 6.5]:
            probs  = [over_under_probs(p, line)[0] for p in preds]
            actual = (te["strikeouts_actual"] > line).astype(int).values
            all_probs.extend(probs)
            all_overs.extend(actual.tolist())
            pred_rate   = float(np.mean(probs))
            actual_rate = float(np.mean(actual))
            line_accs[line] = (pred_rate, actual_rate)

        acc_str = "  ".join(
            f"@{l}: pred {p:.3f} vs act {a:.3f}" for l, (p, a) in line_accs.items()
        )
        print(f"  {max(train_s)}→{test_s}: n={len(te):,}  MAE={mae:.3f}  {acc_str}")

    print(f"  Naive baseline MAE: {naive:.3f}")

    # Pooled calibration: do 60% over-probs go over ~60% of the time?
    if all_probs:
        bins = pd.cut(pd.Series(all_probs), bins=[0, .3, .4, .5, .6, .7, 1.0])
        cal = (
            pd.DataFrame({"prob": all_probs, "over": all_overs, "bin": bins})
            .groupby("bin", observed=True)
            .agg(n=("over", "count"), predicted=("prob", "mean"), actual=("over", "mean"))
        )
        print("\n  Over-probability calibration (pooled, all lines):")
        print(cal.to_string(float_format="{:.3f}".format))
        print("\n  If 'actual' diverges from 'predicted' in the tails, do not bet")
        print("  K props at those probabilities — the Poisson tail is mispricing.")


# ---------------------------------------------------------------------------
# Live prediction
# ---------------------------------------------------------------------------

def predict_strikeouts(
    pitcher_id: int,
    game_date: str,
    line: float,
    home_team: str,
    opponent_team: str,
    pitcher_is_home: bool = False,
    opp_k9_cache: dict | None = None,
    ump_k9_vs_avg: float = 0.0,
    opp_lineup_k_pct: float | None = None,
    swstr_cache: dict | None = None,
) -> dict | None:
    """
    Predict expected Ks and probability of going over/under the line.

    opp_lineup_k_pct : PA-weighted K% of the confirmed opposing lineup, if
                       posted (caller computes it from batter_game_logs).
                       None → filled with the training median.
    swstr_cache      : output of build_prior_swstr_cache(); rebuilt per call
                       when not supplied.

    Returns dict with expected_k, prob_over, prob_under or None if
    insufficient pitcher history.
    """
    model_path = MODEL_DIR / "model_strikeouts.pkl"
    if not model_path.exists():
        print("  Strikeout model not trained yet. Run: python props/strikeout_model.py")
        return None

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    medians      = bundle.get("medians", {})
    league_k9    = bundle.get("league_k9", 9.0)

    engine  = get_engine()
    session = get_session(engine)
    feats   = build_starter_features(session, pitcher_id, game_date)
    session.close()

    cache  = opp_k9_cache if opp_k9_cache is not None else build_opp_k9_cache(engine)
    season = int(game_date[:4])
    opp_k9 = cache.get((opponent_team, season, game_date), league_k9)

    if feats is None:
        return None

    feats["is_home"]       = 1 if pitcher_is_home else 0
    feats["opp_k9"]        = opp_k9
    feats["ump_k9_vs_avg"] = ump_k9_vs_avg

    # Prior-season whiff quality
    sw_cache = swstr_cache if swstr_cache is not None else build_prior_swstr_cache(engine)
    sw = sw_cache.get((int(pitcher_id), season - 1))
    feats["swstr_prior"] = sw[0] if sw else None
    feats["csw_prior"]   = sw[1] if sw else None

    # Confirmed opposing lineup K% (None → training median, flag 0)
    feats["opp_lineup_k_pct"] = opp_lineup_k_pct
    feats["opp_lineup_data"]  = 1.0 if opp_lineup_k_pct is not None else 0.0

    X = pd.DataFrame([{
        col: feats[col] if feats.get(col) is not None else medians.get(col, 0.0)
        for col in feature_cols
    }])
    expected_k = float(model.predict(X)[0])

    prob_over, prob_under = over_under_probs(expected_k, line)

    return {
        "pitcher_id":  pitcher_id,
        "game_date":   game_date,
        "line":        line,
        "expected_k":  round(expected_k, 2),
        "prob_over":   round(prob_over, 4),
        "prob_under":  round(prob_under, 4),
    }


if __name__ == "__main__":
    walk_forward_validate()
    train_k_model()

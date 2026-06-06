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

# Park strikeout factors — relative to league average (1.0 = neutral)
# Higher = more Ks in that park (bigger, pitcher-friendly, etc.)
PARK_K_FACTOR: dict[str, float] = {
    "COL": 0.94,  # thin air, more contact
    "NYY": 1.03,
    "BOS": 1.01,
    "CHC": 1.02,
    "SF":  1.01,
    "LAD": 1.02,
    "HOU": 1.01,
    "TB":  1.01,
    "NYM": 1.00,
    "ATL": 1.01,
    "MIL": 0.99,
    "CIN": 0.99,
    "SD":  1.00,
    "TEX": 1.00,
    "PHI": 1.00,
}


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


def build_training_dataset(sessions_list: list[int] | None = None) -> pd.DataFrame:
    """
    Build the full training dataset from pitcher game logs.
    Each row is one start with rolling features computed from prior starts.
    Target: strikeouts in this start.
    """
    engine  = get_engine()
    session = get_session(engine)

    # Pre-build opponent K9 cache (avoids N+1 queries)
    opp_k9_cache = build_opp_k9_cache(engine)

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
]


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

    # Fill missing features with column medians
    for col in FEATURE_COLS_K:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

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
            "league_k9": float(df["k_per_9_season"].mean()) if "k_per_9_season" in df else 9.0,
        }, f)
    print(f"  Saved to {path}")

    return model


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
) -> dict | None:
    """
    Predict expected Ks and probability of going over/under the line.

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

    feats["is_home"] = 1 if pitcher_is_home else 0
    feats["opp_k9"]  = opp_k9
    pk_factor        = PARK_K_FACTOR.get(home_team, 1.0)

    X = pd.DataFrame([{col: feats.get(col) or 0.0 for col in feature_cols}])
    expected_k = float(model.predict(X)[0]) * pk_factor

    k_threshold = int(line) + 1
    prob_over   = 1 - poisson.cdf(k_threshold - 1, max(expected_k, 0.1))
    prob_under  = 1 - prob_over

    return {
        "pitcher_id":  pitcher_id,
        "game_date":   game_date,
        "line":        line,
        "expected_k":  round(expected_k, 2),
        "prob_over":   round(prob_over, 4),
        "prob_under":  round(prob_under, 4),
    }


if __name__ == "__main__":
    train_k_model()

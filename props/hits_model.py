"""
Batter hits prop model.

Predicts expected hits per game using Ridge regression, then converts to
P(hits >= line) via a Poisson approximation — same pattern as the K model.

Features (all look-ahead-bias-safe: strictly prior to game_date):
  ba_l10         : batting average last 10 games (rolling hot/cold)
  ba_season      : season-to-date batting average
  contact_pct_l10: (1 - K/PA) last 10 games — strikeout tendency
  pa_per_game_l10: avg PA last 10 games (lineup position proxy)
  opp_sp_era     : opposing starter's prior-season ERA (from games table)
  is_home        : home batters see slightly more PA

Walk-forward: train on seasons ≤ N-1, test on N.
"""

from __future__ import annotations

import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from sqlalchemy import text
from scipy.stats import poisson
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.schema import get_engine, get_session, BatterGameLog

MODEL_DIR = Path(__file__).parent.parent / "model"

FEATURE_COLS_HITS = [
    "ba_l10",
    "ba_season",
    "contact_pct_l10",
    "pa_per_game_l10",
    "opp_sp_era",
    "is_home",
]


# ---------------------------------------------------------------------------
# Opponent SP ERA cache (from games table via game_pk join)
# ---------------------------------------------------------------------------

def build_opp_sp_era_cache(engine) -> dict[int, float]:
    """
    Returns {game_pk: opp_sp_era} where opp_sp_era is the opposing
    starter's prior-season ERA. Uses the games table home/away SP ERA.
    We don't know the batter's side yet, so we store both:
    {game_pk: {"home_opp_era": ..., "away_opp_era": ...}}
    """
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT game_pk, home_sp_era, away_sp_era FROM games "
            "WHERE game_pk IS NOT NULL"
        )).fetchall()

    league_era = 4.20
    cache: dict[int, dict] = {}
    for r in rows:
        cache[r.game_pk] = {
            # Home batter faces the away SP
            "home": float(r.away_sp_era) if r.away_sp_era else league_era,
            # Away batter faces the home SP
            "away": float(r.home_sp_era) if r.home_sp_era else league_era,
        }
    return cache


# ---------------------------------------------------------------------------
# Per-batter rolling features
# ---------------------------------------------------------------------------

def build_batter_features(
    session,
    mlbam_id: int,
    game_date: str,
    n_lookback: int = 10,
    opp_era_cache: dict | None = None,
    game_pk: int | None = None,
    home_away: str = "away",
) -> dict | None:
    """
    Build features for a single batter appearance using only data
    strictly before game_date.
    """
    prior = (
        session.query(BatterGameLog)
        .filter(
            BatterGameLog.mlbam_id  == mlbam_id,
            BatterGameLog.game_date <  game_date,
            BatterGameLog.pa        >= 1,
        )
        .order_by(BatterGameLog.game_date.desc())
        .all()
    )

    if len(prior) < 5:
        return None

    recent   = prior[:n_lookback]
    all_prev = prior

    # Rolling BA last N games
    r_h  = sum(g.hits or 0 for g in recent)
    r_ab = sum(g.ab   or 0 for g in recent)
    ba_l10 = r_h / r_ab if r_ab > 0 else None

    # Season-to-date BA
    s_h  = sum(g.hits or 0 for g in all_prev)
    s_ab = sum(g.ab   or 0 for g in all_prev)
    ba_season = s_h / s_ab if s_ab > 0 else None

    # Contact rate (1 - K/PA) last N games
    r_k  = sum(g.strikeouts or 0 for g in recent)
    r_pa = sum(g.pa         or 0 for g in recent)
    contact_pct = 1 - (r_k / r_pa) if r_pa > 0 else None

    # Avg PA per game last N (lineup position proxy)
    pa_pg = sum(g.pa or 0 for g in recent) / len(recent)

    # Opposing SP ERA
    league_era = 4.20
    opp_era = league_era
    if opp_era_cache and game_pk and game_pk in opp_era_cache:
        opp_era = opp_era_cache[game_pk].get(home_away, league_era)

    return {
        "ba_l10":          ba_l10,
        "ba_season":       ba_season,
        "contact_pct_l10": contact_pct,
        "pa_per_game_l10": pa_pg,
        "opp_sp_era":      opp_era,
        "is_home":         1 if home_away == "home" else 0,
    }


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

def build_hits_training_dataset() -> pd.DataFrame:
    engine  = get_engine()
    session = get_session(engine)
    opp_era_cache = build_opp_sp_era_cache(engine)

    all_games = (
        session.query(BatterGameLog)
        .filter(BatterGameLog.hits.isnot(None), BatterGameLog.pa >= 1)
        .order_by(BatterGameLog.mlbam_id, BatterGameLog.game_date)
        .all()
    )

    rows = []
    for g in all_games:
        feats = build_batter_features(
            session, g.mlbam_id, g.game_date,
            opp_era_cache=opp_era_cache,
            game_pk=g.game_pk,
            home_away=g.home_away or "away",
        )
        if feats is None:
            continue
        feats["hits_actual"] = g.hits
        feats["mlbam_id"]    = g.mlbam_id
        feats["game_date"]   = g.game_date
        feats["season"]      = g.season
        rows.append(feats)

    session.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_hits_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=1.0)),
    ])


def train_hits_model(verbose: bool = True) -> Pipeline:
    print("Building hits training dataset...")
    df = build_hits_training_dataset()

    if df.empty:
        raise RuntimeError("No batter game log data — run ingestion/batter_data.py first.")

    print(f"  Rows: {len(df):,}  Seasons: {sorted(df['season'].unique())}")
    print(f"  Avg hits/game: {df['hits_actual'].mean():.3f}  "
          f"H≥1 rate: {(df['hits_actual']>=1).mean():.3f}  "
          f"H≥2 rate: {(df['hits_actual']>=2).mean():.3f}")

    for col in FEATURE_COLS_HITS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)

    X = df[FEATURE_COLS_HITS]
    y = df["hits_actual"]

    model = make_hits_model()
    model.fit(X, y)

    preds = model.predict(X)
    mae   = mean_absolute_error(y, preds)
    print(f"  Training MAE: {mae:.3f} hits  (in-sample)")

    path = MODEL_DIR / "model_hits.pkl"
    with open(path, "wb") as f:
        pickle.dump({
            "model":        model,
            "feature_cols": FEATURE_COLS_HITS,
            "league_era":   4.20,
        }, f)
    print(f"  Saved to {path}")
    return model


def walk_forward_validate() -> None:
    df = build_hits_training_dataset()
    if df.empty:
        print("No data.")
        return

    for col in FEATURE_COLS_HITS:
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)

    seasons = sorted(df["season"].unique())
    print("\nWalk-forward validation (hits model):")
    results = []
    for i in range(2, len(seasons)):
        train_s = seasons[:i]
        test_s  = int(seasons[i])
        tr = df[df["season"].isin(train_s)]
        te = df[df["season"] == test_s]
        model = make_hits_model()
        model.fit(tr[FEATURE_COLS_HITS], tr["hits_actual"])
        preds = model.predict(te[FEATURE_COLS_HITS])
        mae = mean_absolute_error(te["hits_actual"], preds)

        accs = {}
        for line in [0.5, 1.5, 2.5]:
            thresh = int(line) + 1
            accs[line] = sum(
                ((1 - poisson.cdf(thresh - 1, max(p, 0.01))) > 0.5) == (a >= thresh)
                for a, p in zip(te["hits_actual"], preds)
            ) / len(te)

        results.append((test_s, mae, accs))
        print(f"  {max(train_s)}→{test_s}: n={len(te):,}  MAE={mae:.3f}  "
              f"@0.5={accs[0.5]:.3f}  @1.5={accs[1.5]:.3f}  @2.5={accs[2.5]:.3f}")

    maes = [r[1] for r in results]
    naive = mean_absolute_error(df["hits_actual"],
                                [df["hits_actual"].mean()] * len(df))
    print(f"\nMean MAE: {np.mean(maes):.3f}  naive baseline: {naive:.3f}")

    rg = model.named_steps["reg"]
    print("\nFeature weights (last fold):")
    for f, c in sorted(zip(FEATURE_COLS_HITS, rg.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {f:<22}: {c:+.4f}")


# ---------------------------------------------------------------------------
# Live prediction
# ---------------------------------------------------------------------------

def predict_hits(
    batter_id: int,
    game_date: str,
    line: float,
    game_pk: int | None = None,
    home_away: str = "away",
    opp_era_cache: dict | None = None,
) -> dict | None:
    """
    Predict expected hits and P(over/under line) for a batter.
    Returns dict with expected_hits, prob_over, prob_under or None if
    insufficient history.
    """
    model_path = MODEL_DIR / "model_hits.pkl"
    if not model_path.exists():
        return None

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]

    engine  = get_engine()
    session = get_session(engine)

    cache = opp_era_cache
    if cache is None:
        cache = build_opp_sp_era_cache(engine)

    feats = build_batter_features(
        session, batter_id, game_date,
        opp_era_cache=cache,
        game_pk=game_pk,
        home_away=home_away,
    )
    session.close()

    if feats is None:
        return None

    X = pd.DataFrame([{col: feats.get(col) or 0.0 for col in feature_cols}])
    expected = float(model.predict(X)[0])
    expected = max(expected, 0.01)

    # Push-aware over/under conversion (integer lines void on exact hit)
    from props.strikeout_model import over_under_probs
    prob_over, prob_under = over_under_probs(expected, line)

    return {
        "batter_id":    batter_id,
        "game_date":    game_date,
        "line":         line,
        "expected_hits": round(expected, 3),
        "prob_over":    round(prob_over, 4),
        "prob_under":   round(prob_under, 4),
    }


if __name__ == "__main__":
    walk_forward_validate()
    train_hits_model()

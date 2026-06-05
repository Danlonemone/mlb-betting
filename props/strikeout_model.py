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
from sqlalchemy import text
from scipy.stats import poisson
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.schema import get_engine, get_session, PitcherGameLog

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
    ip_per_start_l5  = safe_mean(recent_ip)
    pit_per_start_l5 = safe_mean(recent_pit)

    # Season-to-date K/9
    all_ks = [s.strikeouts or 0 for s in season_starts]
    all_ip = [s.ip or 0 for s in season_starts]
    k_per_9_season = sum(all_ks) / sum(all_ip) * 9 if sum(all_ip) > 0 else None

    # Statcast quality (use most recent non-null value)
    swstr = next((s.swstr_pct for s in prior_starts if s.swstr_pct is not None), None)
    csw   = next((s.csw_pct   for s in prior_starts if s.csw_pct   is not None), None)

    return {
        "k_per_9_l5":       k_per_9_l5,
        "k_per_9_season":   k_per_9_season,
        "swstr_pct":        swstr,
        "csw_pct":          csw,
        "ip_per_start_l5":  ip_per_start_l5,
        "pit_per_start_l5": pit_per_start_l5,
    }


def build_training_dataset(sessions_list: list[int] | None = None) -> pd.DataFrame:
    """
    Build the full training dataset from pitcher game logs.
    Each row is one start with rolling features computed from prior starts.
    Target: strikeouts in this start.
    """
    engine  = get_engine()
    session = get_session(engine)

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

        feats["strikeouts_actual"] = start.strikeouts
        feats["mlbam_id"]   = start.mlbam_id
        feats["game_date"]  = start.game_date
        feats["game_pk"]    = start.game_pk
        feats["season"]     = start.season
        feats["is_home"]    = 1 if start.home_away == "home" else 0
        feats["opponent"]   = start.opponent
        rows.append(feats)

    session.close()
    return pd.DataFrame(rows)


MIN_START_IP = 3.0

FEATURE_COLS_K = [
    "k_per_9_l5",
    "k_per_9_season",
    "swstr_pct",
    "csw_pct",
    "ip_per_start_l5",
    "pit_per_start_l5",
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

    # Save model
    path = MODEL_DIR / "model_strikeouts.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS_K}, f)
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
    pitcher_is_home: bool = False,
) -> dict | None:
    """
    Predict expected Ks and probability of going over/under the line.

    Returns dict with:
      expected_k, prob_over, prob_under, edge_over, edge_under
    or None if insufficient data.
    """
    model_path = MODEL_DIR / "model_strikeouts.pkl"
    if not model_path.exists():
        print("  Strikeout model not trained yet. Run: python props/strikeout_model.py")
        return None

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model       = bundle["model"]
    feature_cols = bundle["feature_cols"]

    engine  = get_engine()
    session = get_session(engine)
    feats   = build_starter_features(session, pitcher_id, game_date)
    session.close()

    if feats is None:
        return None

    # Add context
    feats["is_home"] = 1 if pitcher_is_home else 0
    pk_factor = PARK_K_FACTOR.get(home_team, 1.0)

    X = pd.DataFrame([{col: feats.get(col, 0.0) or 0.0 for col in feature_cols}])
    expected_k = float(model.predict(X)[0]) * pk_factor

    # Convert to over/under probability using a Poisson approximation
    # P(X >= line + 0.5) ≈ P(Poisson(lambda=expected_k) >= ceil(line))
    k_threshold = int(line) + 1    # over K.5 means K+1 or more whole Ks
    prob_over   = 1 - poisson.cdf(k_threshold - 1, expected_k)
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

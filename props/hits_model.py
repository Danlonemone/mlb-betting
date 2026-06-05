"""
Batter hits prop model.

Target: will this batter get OVER or UNDER the book's hits line (usually 0.5 or 1.5)?

The 0.5 hits market is essentially "does this player get a hit today?" which
is a very liquid prop. The 1.5 line is harder to beat but has more signal.

Features per batter start:
  Batter rolling:
    - ba_l10          : batting average over last 10 games (rolling)
    - babip_season    : BABIP season-to-date (luck normalisaiton)
    - contact_pct_l10 : contact rate (1 - K%) over last 10 games
    - hard_hit_pct    : hard hit % from Statcast (season avg)

  Platoon matchup:
    - same_hand       : 1 if batter and pitcher same handedness (platoon disadvantage)
    - opp_babip_allowed : opposing pitcher's BABIP allowed (hard to sustain)

  Context:
    - batting_order   : lineup spot (1-9; leadoff gets more PA)
    - park_hits_factor: park factor for hits
    - line            : the book's line (0.5 or 1.5)
"""

from __future__ import annotations

import sys
import pickle
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.schema import get_engine, get_session, BatterGameLog

MODEL_DIR = Path(__file__).parent.parent / "model"

FEATURE_COLS_HITS = [
    "ba_l10",
    "babip_season",
    "contact_pct_l10",
    "hard_hit_pct",
    "same_hand",
    "batting_order",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_batter_features(
    session,
    mlbam_id: int,
    game_date: str,
    n_lookback: int = 10,
) -> dict | None:
    prior_games = (
        session.query(BatterGameLog)
        .filter(
            BatterGameLog.mlbam_id == mlbam_id,
            BatterGameLog.game_date < game_date,
            BatterGameLog.ab >= 1,
        )
        .order_by(BatterGameLog.game_date.desc())
        .all()
    )

    if len(prior_games) < 5:
        return None

    recent      = prior_games[:n_lookback]
    all_games   = prior_games

    # Rolling batting average (last N)
    r_h  = sum(g.hits or 0 for g in recent)
    r_ab = sum(g.ab  or 0 for g in recent)
    ba_l10 = r_h / r_ab if r_ab > 0 else None

    # Season BABIP
    s_h  = sum(g.hits        or 0 for g in all_games)
    s_hr = sum(g.home_runs   or 0 for g in all_games)
    s_k  = sum(g.strikeouts  or 0 for g in all_games)
    s_ab = sum(g.ab          or 0 for g in all_games)
    s_sf = 0  # not tracked in game log, approximate as 0
    babip_denom = s_ab - s_k - s_hr + s_sf
    babip_season = (s_h - s_hr) / babip_denom if babip_denom > 0 else None

    # Contact rate (1 - K/PA) rolling
    r_k  = sum(g.strikeouts or 0 for g in recent)
    r_pa = sum(g.pa         or 0 for g in recent)
    contact_pct = 1 - (r_k / r_pa) if r_pa > 0 else None

    # Hard hit %
    hard_hit = next(
        (g.hard_hit_pct for g in all_games if g.hard_hit_pct is not None), None
    )

    return {
        "ba_l10":          ba_l10,
        "babip_season":    babip_season,
        "contact_pct_l10": contact_pct,
        "hard_hit_pct":    hard_hit,
    }


def build_hits_training_dataset() -> pd.DataFrame:
    engine  = get_engine()
    session = get_session(engine)

    all_games = (
        session.query(BatterGameLog)
        .filter(
            BatterGameLog.hits.isnot(None),
            BatterGameLog.ab >= 1,
        )
        .order_by(BatterGameLog.mlbam_id, BatterGameLog.game_date)
        .all()
    )

    rows = []
    for g in all_games:
        feats = build_batter_features(session, g.mlbam_id, g.game_date)
        if feats is None:
            continue

        feats["got_hit"]    = 1 if (g.hits or 0) >= 1 else 0
        feats["got_1p5"]    = 1 if (g.hits or 0) >= 2 else 0
        feats["hits_actual"] = g.hits
        feats["mlbam_id"]   = g.mlbam_id
        feats["game_date"]  = g.game_date
        feats["season"]     = g.season

        # Placeholder context features — filled from game schedule at predict time
        feats["same_hand"]     = 0
        feats["batting_order"] = 5

        rows.append(feats)

    session.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_hits_classifier() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=1000)),
    ])


def train_hits_model(line: float = 0.5) -> Pipeline:
    """
    Train a logistic model for P(batter gets ≥ line hits).
    line=0.5 → P(≥1 hit), line=1.5 → P(≥2 hits).
    """
    print(f"Building hits training dataset (line={line})...")
    df = build_hits_training_dataset()

    if df.empty:
        raise RuntimeError(
            "No batter game log data in DB. "
            "Batter game log ingestion not yet implemented (Phase 4 extension)."
        )

    target = "got_hit" if line == 0.5 else "got_1p5"

    for col in FEATURE_COLS_HITS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    X = df[FEATURE_COLS_HITS]
    y = df[target]

    model = make_hits_classifier()
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    ll    = log_loss(y, probs)
    print(f"  Training log loss: {ll:.4f}  Base rate: {y.mean():.3f}")

    path = MODEL_DIR / f"model_hits_{str(line).replace('.','p')}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS_HITS, "line": line}, f)
    print(f"  Saved to {path}")

    return model


if __name__ == "__main__":
    train_hits_model(line=0.5)

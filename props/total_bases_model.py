"""
Batter total bases (TB) prop model.

Same architecture as the hits model: Ridge regression for expected TB,
push-aware Poisson tail for P(over/under). TB = 1B + 2*2B + 3*3B + 4*HR.

Features (all strictly prior to game_date — no look-ahead):
  tb_per_game_l10 : avg total bases last 10 games (rolling power form)
  slg_l10         : TB/AB last 10 games
  slg_season      : season-to-date TB/AB
  xbh_rate_l10    : extra-base hits per AB last 10 (power frequency)
  contact_pct_l10 : 1 - K/PA last 10 (can't slug what you don't touch)
  pa_per_game_l10 : avg PA last 10 (lineup-position proxy)
  opp_sp_era      : opposing starter's prior-season ERA
  is_home

NOTE: BatterGameLog has avg_exit_velo / barrel_pct / hard_hit_pct columns
but the ingestion never populates them. If per-game Statcast quality is
ever ingested, those are the first features to add — they predict TB
better than results do.

Walk-forward: train on seasons <= N-1, test on N. Run this file directly
to validate BEFORE letting the daily loop bet this market.
"""

from __future__ import annotations

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from db.schema import get_engine, get_session, BatterGameLog
from props.hits_model import build_opp_sp_era_cache
from props.strikeout_model import over_under_probs

MODEL_DIR = Path(__file__).parent.parent / "model"

FEATURE_COLS_TB = [
    "tb_per_game_l10",
    "slg_l10",
    "slg_season",
    "xbh_rate_l10",
    "contact_pct_l10",
    "pa_per_game_l10",
    "opp_sp_era",
    "is_home",

    # Per-game Statcast power quality, rolling last 10 (populated by
    # ingestion/statcast_batter_data.py; median-filled + flagged when absent)
    "ev_l10",            # avg exit velocity on batted balls
    "barrel_pct_l10",    # barrel rate
    "hard_hit_l10",      # hard-hit rate (>= 95 mph)
    "statcast_data",     # 1.0 when the rolling window had Statcast values
]


# ---------------------------------------------------------------------------
# Per-batter rolling features
# ---------------------------------------------------------------------------

def build_tb_features(
    session,
    mlbam_id: int,
    game_date: str,
    n_lookback: int = 10,
    opp_era_cache: dict | None = None,
    game_pk: int | None = None,
    home_away: str = "away",
) -> dict | None:
    """Features for one batter appearance using only data before game_date."""
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

    recent = prior[:n_lookback]

    r_tb  = sum(g.total_bases or 0 for g in recent)
    r_ab  = sum(g.ab or 0 for g in recent)
    r_pa  = sum(g.pa or 0 for g in recent)
    r_k   = sum(g.strikeouts or 0 for g in recent)
    r_xbh = sum((g.doubles or 0) + (g.triples or 0) + (g.home_runs or 0) for g in recent)

    s_tb = sum(g.total_bases or 0 for g in prior)
    s_ab = sum(g.ab or 0 for g in prior)

    league_era = 4.20
    opp_era = league_era
    if opp_era_cache and game_pk and game_pk in opp_era_cache:
        opp_era = opp_era_cache[game_pk].get(home_away, league_era)

    # Rolling Statcast quality (only over games where it was ingested)
    evs     = [g.avg_exit_velo for g in recent if g.avg_exit_velo is not None]
    barrels = [g.barrel_pct    for g in recent if g.barrel_pct    is not None]
    hards   = [g.hard_hit_pct  for g in recent if g.hard_hit_pct  is not None]
    has_sc  = len(evs) >= 3   # need a few games for a meaningful average

    return {
        "tb_per_game_l10": r_tb / len(recent),
        "slg_l10":         r_tb / r_ab if r_ab > 0 else None,
        "slg_season":      s_tb / s_ab if s_ab > 0 else None,
        "xbh_rate_l10":    r_xbh / r_ab if r_ab > 0 else None,
        "contact_pct_l10": 1 - (r_k / r_pa) if r_pa > 0 else None,
        "pa_per_game_l10": r_pa / len(recent),
        "opp_sp_era":      opp_era,
        "is_home":         1 if home_away == "home" else 0,
        "ev_l10":          sum(evs) / len(evs)         if has_sc else None,
        "barrel_pct_l10":  sum(barrels) / len(barrels) if barrels and has_sc else None,
        "hard_hit_l10":    sum(hards) / len(hards)     if hards and has_sc else None,
        "statcast_data":   1.0 if has_sc else 0.0,
    }


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

def build_tb_training_dataset() -> pd.DataFrame:
    engine  = get_engine()
    session = get_session(engine)
    opp_era_cache = build_opp_sp_era_cache(engine)

    all_games = (
        session.query(BatterGameLog)
        .filter(BatterGameLog.total_bases.isnot(None), BatterGameLog.pa >= 1)
        .order_by(BatterGameLog.mlbam_id, BatterGameLog.game_date)
        .all()
    )

    rows = []
    for g in all_games:
        feats = build_tb_features(
            session, g.mlbam_id, g.game_date,
            opp_era_cache=opp_era_cache,
            game_pk=g.game_pk,
            home_away=g.home_away or "away",
        )
        if feats is None:
            continue
        feats["tb_actual"] = g.total_bases
        feats["mlbam_id"]  = g.mlbam_id
        feats["game_date"] = g.game_date
        feats["season"]    = g.season
        rows.append(feats)

    session.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_tb_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=1.0)),
    ])


def train_tb_model(verbose: bool = True) -> Pipeline:
    print("Building total-bases training dataset...")
    df = build_tb_training_dataset()
    if df.empty:
        raise RuntimeError("No batter game log data — run ingestion/batter_data.py first.")

    print(f"  Rows: {len(df):,}  Seasons: {sorted(df['season'].unique())}")
    print(f"  Avg TB/game: {df['tb_actual'].mean():.3f}  "
          f"TB≥2 rate: {(df['tb_actual']>=2).mean():.3f}")

    medians: dict[str, float] = {}
    for col in FEATURE_COLS_TB:
        if col in df.columns:
            med = df[col].median()
            medians[col] = float(med) if pd.notna(med) else 0.0
            df[col] = df[col].fillna(medians[col])

    X = df[FEATURE_COLS_TB]
    y = df["tb_actual"]

    model = make_tb_model()
    model.fit(X, y)

    mae = mean_absolute_error(y, model.predict(X))
    print(f"  Training MAE: {mae:.3f} TB  (in-sample)")

    path = MODEL_DIR / "model_total_bases.pkl"
    with open(path, "wb") as f:
        pickle.dump({
            "model":        model,
            "feature_cols": FEATURE_COLS_TB,
            "medians":      medians,
        }, f)
    print(f"  Saved to {path}")
    return model


def walk_forward_validate() -> None:
    """Walk-forward MAE + over-probability calibration at common TB lines."""
    df = build_tb_training_dataset()
    if df.empty:
        print("No data.")
        return

    for col in FEATURE_COLS_TB:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)

    seasons = sorted(df["season"].unique())
    print("\nWalk-forward validation (total bases model):")

    all_probs: list[float] = []
    all_overs: list[int] = []

    for i in range(2, len(seasons)):
        train_s = seasons[:i]
        test_s  = int(seasons[i])
        tr = df[df["season"].isin(train_s)]
        te = df[df["season"] == test_s]
        if tr.empty or te.empty:
            continue

        model = make_tb_model()
        model.fit(tr[FEATURE_COLS_TB], tr["tb_actual"])
        preds = model.predict(te[FEATURE_COLS_TB])
        mae   = mean_absolute_error(te["tb_actual"], preds)

        accs = {}
        for line in [0.5, 1.5, 2.5]:
            probs  = [over_under_probs(max(p, 0.01), line)[0] for p in preds]
            actual = (te["tb_actual"] > line).astype(int).values
            all_probs.extend(probs)
            all_overs.extend(actual.tolist())
            accs[line] = (float(np.mean(probs)), float(np.mean(actual)))

        acc_str = "  ".join(f"@{l}: pred {p:.3f} vs act {a:.3f}" for l, (p, a) in accs.items())
        print(f"  {max(train_s)}→{test_s}: n={len(te):,}  MAE={mae:.3f}  {acc_str}")

    naive = mean_absolute_error(df["tb_actual"], [df["tb_actual"].mean()] * len(df))
    print(f"  Naive baseline MAE: {naive:.3f}")

    if all_probs:
        bins = pd.cut(pd.Series(all_probs), bins=[0, .3, .4, .5, .6, .7, 1.0])
        cal = (
            pd.DataFrame({"prob": all_probs, "over": all_overs, "bin": bins})
            .groupby("bin", observed=True)
            .agg(n=("over", "count"), predicted=("prob", "mean"), actual=("over", "mean"))
        )
        print("\n  Over-probability calibration (pooled, all lines):")
        print(cal.to_string(float_format="{:.3f}".format))


# ---------------------------------------------------------------------------
# Live prediction
# ---------------------------------------------------------------------------

def predict_total_bases(
    batter_id: int,
    game_date: str,
    line: float,
    game_pk: int | None = None,
    home_away: str = "away",
    opp_era_cache: dict | None = None,
) -> dict | None:
    """Expected TB and P(over/under). None if insufficient history."""
    model_path = MODEL_DIR / "model_total_bases.pkl"
    if not model_path.exists():
        return None

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    medians      = bundle.get("medians", {})

    engine  = get_engine()
    session = get_session(engine)
    cache   = opp_era_cache if opp_era_cache is not None else build_opp_sp_era_cache(engine)
    feats   = build_tb_features(
        session, batter_id, game_date,
        opp_era_cache=cache, game_pk=game_pk, home_away=home_away,
    )
    session.close()

    if feats is None:
        return None

    X = pd.DataFrame([{
        col: feats[col] if feats.get(col) is not None else medians.get(col, 0.0)
        for col in feature_cols
    }])
    expected = max(float(model.predict(X)[0]), 0.01)
    prob_over, prob_under = over_under_probs(expected, line)

    return {
        "batter_id":   batter_id,
        "game_date":   game_date,
        "line":        line,
        "expected_tb": round(expected, 3),
        "prob_over":   round(prob_over, 4),
        "prob_under":  round(prob_under, 4),
    }


if __name__ == "__main__":
    walk_forward_validate()
    train_tb_model()

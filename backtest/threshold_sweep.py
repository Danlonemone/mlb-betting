"""
Sweep moneyline edge thresholds against the walk-forward backtest.

This uses the same recommender guardrails as live betting:
  - minimum vig-free market probability
  - maximum American odds
  - current Kelly fraction

Output is saved to data/threshold_sweep.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.harness import run_backtest

DATA_DIR = Path(__file__).parent.parent / "data"


def summarize(df: pd.DataFrame, threshold: float) -> dict:
    if df.empty:
        return {
            "min_edge": threshold,
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "staked": 0.0,
            "profit": 0.0,
            "roi": 0.0,
            "avg_edge": 0.0,
            "real_odds_bets": 0,
            "synthetic_odds_bets": 0,
        }

    staked = float(df["stake"].sum())
    profit = float(df["profit"].sum())
    wins = int(df["outcome"].sum())
    bets = len(df)
    return {
        "min_edge": threshold,
        "bets": bets,
        "wins": wins,
        "losses": bets - wins,
        "win_rate": wins / bets if bets else 0.0,
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked else 0.0,
        "avg_edge": float(df["edge"].mean()),
        "real_odds_bets": int((df["odds_source"] == "real").sum()),
        "synthetic_odds_bets": int((df["odds_source"] == "synthetic").sum()),
    }


def run_sweep(
    thresholds: list[float] | None = None,
    model_type: str = "logistic",
    real_odds_only: bool = True,
) -> pd.DataFrame:
    """
    Sweep edge thresholds.

    real_odds_only (default True): tune thresholds using ONLY bets priced
    against real bookmaker closing odds. The synthetic log5 market is a
    much weaker opponent than a real book — including synthetic-priced bets
    inflates ROI and selects a threshold that won't survive contact with
    real prices. The full (real + synthetic) summary is still saved
    alongside for reference.

    Note: ROI from a sweep is in-sample for whichever threshold you pick.
    Treat paper-trading CLV, not this table, as the deciding metric.
    """
    thresholds = thresholds or [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    rows_all, rows_real = [], []
    for threshold in thresholds:
        print(f"\n=== Threshold {threshold:.0%} ===")
        bets = run_backtest(min_edge=threshold, model_type=model_type, verbose=False)
        row_all  = summarize(bets, threshold)
        row_real = summarize(
            bets[bets["odds_source"] == "real"] if not bets.empty else bets, threshold
        )
        rows_all.append(row_all)
        rows_real.append(row_real)
        print(
            f"ALL : bets={row_all['bets']:>4}  win={row_all['win_rate']:.1%}  "
            f"roi={row_all['roi']:+.1%}\n"
            f"REAL: bets={row_real['bets']:>4}  win={row_real['win_rate']:.1%}  "
            f"roi={row_real['roi']:+.1%}"
        )

    df_all  = pd.DataFrame(rows_all)
    df_real = pd.DataFrame(rows_real)
    DATA_DIR.mkdir(exist_ok=True)
    df_all.to_csv(DATA_DIR / "threshold_sweep_all.csv", index=False)
    df_real.to_csv(DATA_DIR / "threshold_sweep.csv", index=False)
    print(f"\nSaved sweeps to {DATA_DIR / 'threshold_sweep.csv'} (real odds only)")
    print(f"           and {DATA_DIR / 'threshold_sweep_all.csv'} (incl. synthetic)")
    return df_real if real_odds_only else df_all


if __name__ == "__main__":
    df = run_sweep()
    print("\nSummary (real closing odds only):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

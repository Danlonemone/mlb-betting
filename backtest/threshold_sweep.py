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
) -> pd.DataFrame:
    thresholds = thresholds or [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    rows = []
    for threshold in thresholds:
        print(f"\n=== Threshold {threshold:.0%} ===")
        bets = run_backtest(min_edge=threshold, model_type=model_type, verbose=False)
        row = summarize(bets, threshold)
        rows.append(row)
        print(
            f"bets={row['bets']:>4}  "
            f"win={row['win_rate']:.1%}  "
            f"roi={row['roi']:+.1%}  "
            f"profit={row['profit']:+.3f}"
        )

    result = pd.DataFrame(rows)
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "threshold_sweep.csv"
    result.to_csv(out, index=False)
    print(f"\nSaved sweep to {out}")
    return result


if __name__ == "__main__":
    df = run_sweep()
    print("\nSummary:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

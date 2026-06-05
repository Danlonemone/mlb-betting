"""
Running performance summary for paper trading.

Prints the key metrics we care about as bets accumulate:
  - ROI and P&L
  - Win rate vs implied win rate (calibration)
  - CLV (against real closing lines, when recorded)
  - Rolling 50-bet ROI to spot trend changes
  - Bankroll progression

Usage:
    python paper_trade/performance.py
    python paper_trade/performance.py --min-bets 20
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.schema import get_engine
from betting.odds import format_american

DATA_DIR = Path(__file__).parent.parent / "data"


def load_paper_bets(settled_only: bool = False) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM paper_bets ORDER BY game_date, game_pk"),
            conn,
        )
    if settled_only:
        df = df[df["outcome"].notna()].copy()
    return df


def print_performance(min_bets: int = 10):
    df_all = load_paper_bets(settled_only=False)
    df = load_paper_bets(settled_only=True)

    print(f"\n{'='*60}")
    print("PAPER TRADING PERFORMANCE SUMMARY")
    print(f"{'='*60}")

    total_logged   = len(df_all)
    total_settled  = len(df)
    total_pending  = total_logged - total_settled

    print(f"Bets logged:   {total_logged}")
    print(f"Settled:       {total_settled}")
    print(f"Pending:       {total_pending}")

    if total_settled < min_bets:
        print(f"\n  ⏳ Not enough settled bets yet ({total_settled}/{min_bets} minimum).")
        print(f"  Keep running daily_picks.py and settle.py each day.")
        _print_pending(df_all[df_all["outcome"].isna()])
        return

    # --- ROI ---
    total_staked  = df["stake_dollars"].sum()
    total_profit  = df["profit_dollars"].sum()
    roi           = total_profit / total_staked if total_staked else 0
    win_rate      = df["outcome"].mean()
    wins          = int(df["outcome"].sum())
    losses        = total_settled - wins

    print(f"\n--- ROI ---")
    print(f"  Win/Loss:    {wins}W / {losses}L  ({win_rate:.1%} win rate)")
    print(f"  Total staked:${total_staked:.2f}")
    print(f"  Total profit:${total_profit:+.2f}")
    print(f"  ROI:         {roi:+.2%}")

    # --- Mean edge vs actual ROI ---
    mean_edge = df["edge"].mean()
    print(f"\n--- Edge vs Outcome ---")
    print(f"  Mean edge at bet time: {mean_edge:+.2%}")
    print(f"  Actual ROI:            {roi:+.2%}")
    gap = roi - mean_edge
    if abs(gap) < 0.05:
        verdict = "Edge and ROI are in the same ballpark — consistent with model accuracy."
    elif gap < -0.05:
        verdict = "ROI below mean edge — model may be overestimating its advantage."
    else:
        verdict = "ROI above mean edge — positive variance or model is conservative."
    print(f"  Gap:                   {gap:+.2%}  ({verdict})")

    # --- CLV (only for bets with closing odds recorded) ---
    clv_df = df[df["clv"].notna()]
    if len(clv_df) >= 5:
        mean_clv  = clv_df["clv"].mean()
        pct_pos   = (clv_df["clv"] > 0).mean()
        print(f"\n--- Closing Line Value (real bookmaker closes) ---")
        print(f"  Bets with CLV data: {len(clv_df)}")
        print(f"  Mean CLV:           {mean_clv:+.2%}")
        print(f"  % positive CLV:     {pct_pos:.0%}")
        if mean_clv > 0.01:
            print(f"  ✓ Positive CLV — strongest signal that the edge is real.")
        elif mean_clv > 0:
            print(f"  ~ Marginally positive CLV — accumulate more data.")
        else:
            print(f"  ✗ Negative CLV — we're buying bad prices, not finding value.")
    else:
        print(f"\n--- CLV ---")
        print(f"  Not enough closing odds recorded yet.")
        print(f"  Use daily_picks.record_closing_odds() before each game starts.")

    # --- Calibration ---
    if total_settled >= 30:
        df["prob_bin"] = pd.cut(df["model_prob"], bins=5)
        cal = (
            df.groupby("prob_bin", observed=True)
            .agg(n=("outcome", "count"),
                 predicted=("model_prob", "mean"),
                 actual=("outcome", "mean"))
            .reset_index()
        )
        print(f"\n--- Calibration (binned) ---")
        print(f"  {'Predicted':>11} {'Actual':>8} {'Gap':>7} {'N':>5}")
        for _, row in cal.iterrows():
            gap = row["actual"] - row["predicted"]
            print(f"  {row['predicted']:>10.1%} {row['actual']:>7.1%} "
                  f"{gap:>+6.1%} {int(row['n']):>5}")

    # --- By date ---
    print(f"\n--- By Date ---")
    by_date = (
        df.groupby("game_date")
        .agg(bets=("outcome","count"),
             wins=("outcome","sum"),
             staked=("stake_dollars","sum"),
             profit=("profit_dollars","sum"))
        .reset_index()
    )
    by_date["roi"] = by_date["profit"] / by_date["staked"]
    print(f"  {'Date':<12} {'Bets':<6} {'W':<4} {'L':<4} {'Staked':>8} {'P&L':>8} {'ROI':>7}")
    for _, row in by_date.iterrows():
        l = int(row["bets"] - row["wins"])
        print(f"  {row['game_date']:<12} {int(row['bets']):<6} "
              f"{int(row['wins']):<4} {l:<4} "
              f"${row['staked']:>7.2f} ${row['profit']:>+7.2f} "
              f"{row['roi']:>+6.1%}")

    # --- Bankroll progression ---
    df_sorted = df.sort_values("game_date").copy()
    initial_bankroll = df_sorted["bankroll_at_bet"].iloc[0] if not df_sorted.empty else 1000
    df_sorted["cum_profit"] = df_sorted["profit_dollars"].cumsum()
    df_sorted["bankroll"]   = initial_bankroll + df_sorted["cum_profit"]
    current_bankroll = df_sorted["bankroll"].iloc[-1]

    print(f"\n--- Bankroll ---")
    print(f"  Starting:  ${initial_bankroll:,.2f}")
    print(f"  Current:   ${current_bankroll:,.2f}  ({(current_bankroll/initial_bankroll - 1):+.1%})")

    # Rolling 50-bet ROI
    if total_settled >= 50:
        df_sorted["rolling_roi"] = (
            df_sorted["profit_dollars"].rolling(50).sum() /
            df_sorted["stake_dollars"].rolling(50).sum()
        )
        print(f"\n--- Rolling 50-bet ROI ---")
        rolling_latest = df_sorted["rolling_roi"].iloc[-1]
        print(f"  Latest:  {rolling_latest:+.2%}")

    # Plot
    _plot(df_sorted, DATA_DIR / "paper_trade_performance.png")
    print(f"\n{'='*60}")


def _print_pending(pending: pd.DataFrame):
    if pending.empty:
        return
    print(f"\n--- Pending Bets ---")
    for _, row in pending.iterrows():
        side = row["home_team"] if row["bet_side"] == "home" else row["away_team"]
        print(f"  {row['game_date']}  {row['away_team']}@{row['home_team']}  "
              f"Bet {side}  {format_american(row['bet_american_odds'])}  "
              f"${row['stake_dollars']:.2f}")


def _plot(df: pd.DataFrame, path: Path):
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Paper Trading Performance", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(range(len(df)), df["bankroll"].values, color="#2196F3", lw=1.5)
    ax.axhline(df["bankroll_at_bet"].iloc[0], color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Bet number")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Bankroll over time")

    ax = axes[1]
    cum_roi = df["cum_profit"] / df["stake_dollars"].cumsum()
    ax.plot(range(len(df)), cum_roi * 100, color="#9C27B0", lw=1.5)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Bet number")
    ax.set_ylabel("Cumulative ROI (%)")
    ax.set_title("Cumulative ROI")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Plot saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-bets", type=int, default=10,
                        help="Minimum settled bets before printing full report")
    args = parser.parse_args()
    print_performance(min_bets=args.min_bets)

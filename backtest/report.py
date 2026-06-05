"""
Backtest analysis and reporting.

Produces three evaluations required by Phase 3:
  1. Calibration  — predicted probabilities vs actual win rates
  2. ROI          — P&L per unit staked, cumulative over the backtest
  3. CLV proxy    — does our model consistently beat the synthetic market price?
                    (Real CLV against actual closing lines begins in Phase 4.)

All plots are saved to data/backtest_*.png.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.harness import run_backtest

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_analysis(df: pd.DataFrame) -> dict:
    probs  = df["model_prob"].values
    actual = df["outcome"].values

    ll    = log_loss(actual, probs)
    brier = brier_score_loss(actual, probs)

    frac_pos, mean_pred = calibration_curve(actual, probs, n_bins=10, strategy="quantile")
    cal_df = pd.DataFrame({
        "predicted": mean_pred,
        "actual":    frac_pos,
        "gap":       frac_pos - mean_pred,
        "n":         [len(df) // 10] * len(mean_pred),   # approx
    })

    return {"log_loss": ll, "brier": brier, "table": cal_df,
            "probs": probs, "actual": actual}


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------

def roi_analysis(df: pd.DataFrame) -> dict:
    total_staked  = df["stake"].sum()
    total_profit  = df["profit"].sum()
    roi           = total_profit / total_staked if total_staked else 0
    win_rate      = df["outcome"].mean()
    n_bets        = len(df)

    # Yield per bet (avg profit / avg stake)
    yield_per_bet = df["profit"].mean() / df["stake"].mean() if not df.empty else 0

    # By season
    by_season = (
        df.groupby("season")
        .apply(lambda g: pd.Series({
            "bets":   len(g),
            "wins":   g["outcome"].sum(),
            "staked": g["stake"].sum(),
            "profit": g["profit"].sum(),
            "roi":    g["profit"].sum() / g["stake"].sum() if g["stake"].sum() > 0 else 0,
        }))
        .reset_index()
    )

    # Cumulative P&L (sorted by date)
    cum_df = df.sort_values("game_date").copy()
    cum_df["cum_profit"] = cum_df["profit"].cumsum()
    cum_df["cum_staked"] = cum_df["stake"].cumsum()
    cum_df["cum_roi"]    = cum_df["cum_profit"] / cum_df["cum_staked"]

    return {
        "total_staked":  total_staked,
        "total_profit":  total_profit,
        "roi":           roi,
        "win_rate":      win_rate,
        "n_bets":        n_bets,
        "yield_per_bet": yield_per_bet,
        "by_season":     by_season,
        "cumulative":    cum_df,
    }


# ---------------------------------------------------------------------------
# CLV proxy
# ---------------------------------------------------------------------------

def clv_analysis(df: pd.DataFrame) -> dict:
    """
    Closing line value proxy: compare our model's probability against
    the synthetic market's fair probability for the same side.

    CLV = model_prob - market_fair_prob  (same as 'edge' column)

    A positive mean CLV means we consistently see the game differently
    from the log5 market — the right direction for a profitable model.

    Note: This is NOT real CLV because we're comparing to a synthetic
    market, not actual bookmaker closing lines. Real CLV measurement
    starts in Phase 4 when we record live odds.
    """
    df = df.copy()
    df["clv"] = df["model_prob"] - df["fair_prob"]   # same as edge column

    mean_clv   = df["clv"].mean()
    median_clv = df["clv"].median()
    pct_pos    = (df["clv"] > 0).mean()

    # CLV by edge bin
    df["edge_bin"] = pd.cut(df["edge"], bins=[0, 0.03, 0.05, 0.08, 0.15, 1.0],
                            labels=["<3%", "3-5%", "5-8%", "8-15%", ">15%"])
    by_edge = (
        df.groupby("edge_bin", observed=True)
        .apply(lambda g: pd.Series({
            "bets":     len(g),
            "mean_clv": g["clv"].mean(),
            "win_rate": g["outcome"].mean(),
            "roi":      g["profit"].sum() / g["stake"].sum() if g["stake"].sum() > 0 else 0,
        }))
        .reset_index()
    )

    return {
        "mean_clv":   mean_clv,
        "median_clv": median_clv,
        "pct_pos":    pct_pos,
        "by_edge":    by_edge,
        "df":         df,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_all(cal: dict, roi: dict, clv: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 3 Backtest Report (Synthetic Market)", fontsize=13, fontweight="bold")

    # 1. Calibration curve
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    frac, pred = calibration_curve(cal["actual"], cal["probs"], n_bins=10, strategy="quantile")
    ax.plot(pred, frac, "o-", color="#2196F3", label="Model")
    ax.fill_between(pred, pred, frac, alpha=0.15, color="#2196F3")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual win rate")
    ax.set_title(f"Calibration  (Brier={cal['brier']:.4f}  LogLoss={cal['log_loss']:.4f})")
    ax.legend()
    ax.set_xlim(0.3, 0.75)
    ax.set_ylim(0.3, 0.75)

    # 2. Predicted probability distribution
    ax = axes[0, 1]
    wins  = cal["probs"][cal["actual"] == 1]
    losses = cal["probs"][cal["actual"] == 0]
    ax.hist(wins,   bins=20, alpha=0.6, color="#4CAF50", label="Won",  density=True)
    ax.hist(losses, bins=20, alpha=0.6, color="#F44336", label="Lost", density=True)
    ax.set_xlabel("Model probability (bet side)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted probability: wins vs losses")
    ax.legend()

    # 3. Cumulative ROI over time
    ax = axes[1, 0]
    cum = roi["cumulative"]
    ax.plot(range(len(cum)), cum["cum_roi"] * 100, color="#9C27B0", lw=1.5)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Bet number (chronological)")
    ax.set_ylabel("Cumulative ROI (%)")
    ax.set_title(f"Cumulative ROI  (final: {roi['roi']:+.1%}  n={roi['n_bets']:,})")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # 4. ROI by edge bucket
    ax = axes[1, 1]
    by_edge = clv["by_edge"]
    colors = ["#F44336" if r < 0 else "#4CAF50" for r in by_edge["roi"]]
    bars = ax.bar(range(len(by_edge)), by_edge["roi"] * 100, color=colors, alpha=0.8)
    ax.set_xticks(range(len(by_edge)))
    ax.set_xticklabels(by_edge["edge_bin"].astype(str), fontsize=9)
    ax.set_xlabel("Edge bucket")
    ax.set_ylabel("ROI (%)")
    ax.set_title("ROI by edge size")
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, n in zip(bars, by_edge["bets"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"n={n}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = DATA_DIR / "backtest_report.png"
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  Plot saved to {path}")


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame):
    n_real = (df["odds_source"] == "real").sum() if "odds_source" in df.columns else 0
    n_synth = len(df) - n_real
    market_label = (
        "real closing lines" if n_real == len(df)
        else f"mixed ({n_real:,} real / {n_synth:,} synthetic)"
        if n_real > 0
        else "synthetic log5 (prior-season team W%)"
    )

    print(f"\n{'='*65}")
    print("PHASE 3 BACKTEST REPORT")
    print(f"Market: {market_label}")
    if n_synth > 0:
        print(f"  ⚠ {n_synth:,} bets priced vs synthetic market — ROI is inflated.")
        print(f"  Run: python ingestion/historical_odds.py  to fix this.")
    print(f"{'='*65}")
    print(f"Total bets: {len(df):,}  |  Seasons: {sorted(df['season'].unique())}")

    # --- ROI ---
    roi = roi_analysis(df)
    print(f"\n--- ROI ---")
    print(f"  Bets:        {roi['n_bets']:,}")
    print(f"  Win rate:    {roi['win_rate']:.1%}")
    print(f"  Total staked:{roi['total_staked']:.4f} units")
    print(f"  Total profit:{roi['total_profit']:+.4f} units")
    print(f"  ROI:         {roi['roi']:+.2%}")

    print(f"\n  By season:")
    print(f"  {'Season':<8} {'Bets':<6} {'W':<5} {'L':<5} {'Win%':<7} {'ROI'}")
    for _, row in roi["by_season"].iterrows():
        wins = int(row["wins"])
        losses = int(row["bets"] - row["wins"])
        print(f"  {int(row['season']):<8} {int(row['bets']):<6} "
              f"{wins:<5} {losses:<5} {row['wins']/row['bets']:.1%}   "
              f"{row['roi']:+.2%}")

    # --- Calibration ---
    cal = calibration_analysis(df)
    print(f"\n--- Calibration ---")
    print(f"  Brier score: {cal['brier']:.4f}  (0 = perfect, 0.25 = random)")
    print(f"  Log loss:    {cal['log_loss']:.4f}  (lower = better)")
    print(f"\n  Binned calibration (predicted vs actual win rate):")
    print(f"  {'Predicted':>11} {'Actual':>8} {'Gap':>7}")
    for _, row in cal["table"].iterrows():
        flag = "⚠" if abs(row["gap"]) > 0.04 else " "
        print(f"  {row['predicted']:>10.1%} {row['actual']:>7.1%} "
              f"{row['gap']:>+6.1%}  {flag}")

    # --- CLV proxy ---
    clv = clv_analysis(df)
    print(f"\n--- CLV Proxy (vs synthetic log5 market) ---")
    print(f"  Mean CLV:    {clv['mean_clv']:+.3%}")
    print(f"  Median CLV:  {clv['median_clv']:+.3%}")
    print(f"  % bets positive CLV: {clv['pct_pos']:.1%}")
    print(f"\n  ⚠ These are NOT real closing line values.")
    print(f"  Real CLV measurement starts in Phase 4 (live odds recording).")
    print(f"\n  ROI by edge bucket:")
    print(f"  {'Edge':>8} {'Bets':>6} {'Win%':>7} {'ROI':>8}")
    for _, row in clv["by_edge"].iterrows():
        print(f"  {str(row['edge_bin']):>8} {int(row['bets']):>6} "
              f"{row['win_rate']:>6.1%} {row['roi']:>+7.2%}")

    # Verdict
    print(f"\n--- Verdict ---")
    if roi["roi"] > 0.02:
        verdict = "POSITIVE edge vs synthetic market. Promising — proceed to Phase 4."
    elif roi["roi"] > -0.02:
        verdict = "NEAR-ZERO edge. Inconclusive — feature improvements needed OR wait for real odds in Phase 4."
    else:
        verdict = "NEGATIVE edge vs synthetic market. Model needs work before going live."
    print(f"  {verdict}")
    print(f"\n  Remember: beating a synthetic market is necessary but NOT sufficient.")
    print(f"  Beating real closing lines (Phase 4) is the only honest test.")
    print(f"{'='*65}")

    plot_all(cal, roi, clv)
    return roi, cal, clv


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running walk-forward backtest...")
    df = run_backtest(verbose=True)

    if df.empty:
        print("\nNo bets were placed. Try lowering MIN_EDGE in config.py.")
    else:
        print_report(df)

        # Save bet log
        path = DATA_DIR / "backtest_bets.csv"
        df.to_csv(path, index=False)
        print(f"\nBet log saved to {path}")

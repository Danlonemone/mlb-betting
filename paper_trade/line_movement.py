"""
Line movement analysis — diagnose why CLV is negative.

For each bet placed, compares:
  open  (7am)  → morning (10am) : how much had the line already moved before we bet?
  morning (10am) → close (4pm)  : did the line move with or against us after we bet?

A "timing problem" looks like: lines already 2-3pp moved by morning, close ≈ morning.
A "model problem" looks like: lines flat from open to morning, then move against us open→close.

Usage:
    python paper_trade/line_movement.py
    python paper_trade/line_movement.py --days 14
"""

from __future__ import annotations

import sys
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.schema import get_engine
from betting.odds import american_to_implied_prob, remove_vig


def _implied(american: float | None) -> float | None:
    if american is None:
        return None
    try:
        return american_to_implied_prob(float(american))
    except Exception:
        return None


def _fair(home_am: float | None, away_am: float | None, side: str) -> float | None:
    if home_am is None or away_am is None:
        return None
    try:
        home_fair, away_fair, _ = remove_vig(float(home_am), float(away_am))
        return home_fair if side == "home" else away_fair
    except Exception:
        return None


def load_movement_table(days: int = 30) -> pd.DataFrame:
    """
    Join paper_bets with line_snapshots to produce a per-bet movement table.

    Columns returned:
      game_date, matchup, bet_side, bet_american
      open_home, open_away, open_fair          — 7am snapshot
      morning_home, morning_away, morning_fair  — 10am snapshot (bet time)
      close_home, close_away, close_fair        — 4pm snapshot
      pre_move   : fair prob change open→morning (positive = moved toward our side)
      post_move  : fair prob change morning→close (positive = moved toward our side)
      clv        : recorded CLV from paper_bets
      outcome    : 1/0/None
    """
    engine = get_engine()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    bets = pd.read_sql(text("""
        SELECT  b.id, b.game_date, b.home_team, b.away_team, b.bet_side,
                b.bet_american_odds, b.clv, b.outcome
        FROM    paper_bets b
        WHERE   b.game_date >= :cutoff
        ORDER   BY b.game_date, b.id
    """), engine, params={"cutoff": cutoff})

    snaps = pd.read_sql(text("""
        SELECT  game_date, home_team, away_team, snapshot_label,
                home_american, away_american
        FROM    line_snapshots
        WHERE   game_date >= :cutoff
    """), engine, params={"cutoff": cutoff})

    if bets.empty:
        return pd.DataFrame()

    # Pivot snapshots wide: one row per (game_date, home_team, away_team)
    snap_wide = snaps.pivot_table(
        index=["game_date", "home_team", "away_team"],
        columns="snapshot_label",
        values=["home_american", "away_american"],
        aggfunc="first",
    )
    snap_wide.columns = [f"{v}_{l}" for v, l in snap_wide.columns]
    snap_wide = snap_wide.reset_index()

    df = bets.merge(snap_wide, on=["game_date", "home_team", "away_team"], how="left")

    # Compute fair probs at each label for the bet side
    for label in ("open", "morning", "close"):
        h_col = f"home_american_{label}"
        a_col = f"away_american_{label}"
        if h_col in df.columns and a_col in df.columns:
            df[f"fair_{label}"] = df.apply(
                lambda r: _fair(r.get(h_col), r.get(a_col), r["bet_side"]), axis=1
            )
        else:
            df[f"fair_{label}"] = None

    # Movement: positive = line moved toward our side (good)
    df["pre_move"] = df.apply(
        lambda r: (r["fair_morning"] - r["fair_open"])
        if pd.notna(r.get("fair_morning")) and pd.notna(r.get("fair_open")) else None,
        axis=1,
    )
    df["post_move"] = df.apply(
        lambda r: (r["fair_close"] - r["fair_morning"])
        if pd.notna(r.get("fair_close")) and pd.notna(r.get("fair_morning")) else None,
        axis=1,
    )

    df["matchup"] = df["away_team"] + "@" + df["home_team"]
    return df


def print_movement_report(days: int = 30):
    df = load_movement_table(days=days)

    print(f"\n{'='*70}")
    print("LINE MOVEMENT ANALYSIS")
    print(f"{'='*70}")

    if df.empty:
        print("  No bets found in this window.")
        return

    n_bets     = len(df)
    n_open     = df["fair_open"].notna().sum()
    n_morning  = df["fair_morning"].notna().sum()
    n_close    = df["fair_close"].notna().sum()
    n_pre      = df["pre_move"].notna().sum()
    n_post     = df["post_move"].notna().sum()

    print(f"\nBets in window:          {n_bets}")
    print(f"With open snapshot:      {n_open}  (7am line captured)")
    print(f"With morning snapshot:   {n_morning}  (bet-time line captured)")
    print(f"With close snapshot:     {n_close}  (4pm line captured)")

    if n_pre < 3 and n_post < 3:
        print("\n  Not enough snapshot data yet for movement analysis.")
        print("  The logger started today — data accumulates over time.")
        print("  Come back after 3-5 days of snapshots.")
        _print_raw(df)
        return

    # --- Pre-bet movement (open → morning) ---
    if n_pre >= 3:
        pre = df["pre_move"].dropna()
        mean_pre = pre.mean()
        pct_pos  = (pre > 0).mean()
        print(f"\n--- Pre-bet movement (open → morning, n={n_pre}) ---")
        print(f"  Mean shift toward our side:  {mean_pre:+.3f} prob pts")
        print(f"  % bets where line moved FOR us before we bet:  {pct_pos:.0%}")
        if mean_pre < -0.01:
            print(f"  ✗ Lines already moving AGAINST us by bet time — timing problem")
            print(f"    → Sharps are betting the other side overnight; we're getting on late")
        elif mean_pre > 0.01:
            print(f"  ✓ Lines moving in our direction before we bet — we're on the right side")
        else:
            print(f"  ~ Minimal pre-bet movement — timing is not the issue")

    # --- Post-bet movement (morning → close) ---
    if n_post >= 3:
        post = df["post_move"].dropna()
        mean_post = post.mean()
        pct_pos   = (post > 0).mean()
        print(f"\n--- Post-bet movement (morning → close, n={n_post}) ---")
        print(f"  Mean shift toward our side:  {mean_post:+.3f} prob pts")
        print(f"  % bets where line moved FOR us after we bet:  {pct_pos:.0%}")
        if mean_post < -0.01:
            print(f"  ✗ Lines moving AGAINST us after we bet — model quality problem")
            print(f"    → Market disagrees with our picks; sharps are on the other side")
        elif mean_post > 0.01:
            print(f"  ✓ Lines moving in our direction after we bet — model has real signal")
        else:
            print(f"  ~ Minimal post-bet movement — market is ambivalent about our picks")

    # --- Diagnosis ---
    if n_pre >= 3 and n_post >= 3:
        mean_pre  = df["pre_move"].dropna().mean()
        mean_post = df["post_move"].dropna().mean()
        print(f"\n--- Diagnosis ---")
        if mean_pre < -0.01 and abs(mean_post) < 0.01:
            print("  TIMING PROBLEM: lines already moved before we bet, stable after.")
            print("  Fix: shift the morning automation to 7-7:30am.")
        elif mean_post < -0.01:
            print("  MODEL PROBLEM: lines move against us after we bet.")
            print("  Fix: improve model features (lineup-weighted wOBA, weather, etc.)")
        elif mean_pre < -0.01 and mean_post < -0.01:
            print("  BOTH: lines already moved AND continue moving against us.")
            print("  Fix: earlier automation + model quality improvements.")
        else:
            print("  No clear problem detected — continue monitoring.")

    # --- CLV correlation ---
    clv_df = df[df["clv"].notna() & df["post_move"].notna()]
    if len(clv_df) >= 5:
        corr = clv_df["clv"].corr(clv_df["post_move"])
        print(f"\n--- CLV vs post-bet movement correlation (n={len(clv_df)}) ---")
        print(f"  Pearson r = {corr:.3f}")
        if corr > 0.4:
            print("  ✓ Strong positive — post-move and CLV agree; model has consistent signal")
        elif corr > 0.1:
            print("  ~ Mild positive correlation — improving")
        else:
            print("  ✗ Weak/negative — movement and CLV diverge; check data quality")

    # --- Per-bet table ---
    _print_raw(df)


def _print_raw(df: pd.DataFrame):
    print(f"\n--- Per-bet detail ---")
    print(f"  {'Date':<12} {'Matchup':<14} {'Side':<5} {'Odds':>6}  "
          f"{'PreMove':>8} {'PostMove':>9} {'CLV':>7} {'Out':<5}")
    print(f"  {'─'*70}")
    for _, r in df.iterrows():
        pre  = f"{r['pre_move']:+.3f}"  if pd.notna(r.get("pre_move"))  else "  -  "
        post = f"{r['post_move']:+.3f}" if pd.notna(r.get("post_move")) else "  -  "
        clv  = f"{r['clv']:+.3f}"       if pd.notna(r.get("clv"))       else "  -  "
        out  = "W" if r["outcome"] == 1 else ("L" if r["outcome"] == 0 else "-")
        side = r["home_team"] if r["bet_side"] == "home" else r["away_team"]
        odds = int(r["bet_american_odds"]) if pd.notna(r["bet_american_odds"]) else 0
        sign = "+" if odds >= 0 else ""
        print(f"  {r['game_date']:<12} {r['matchup']:<14} {side:<5} "
              f"{sign}{odds:>5}  {pre:>8} {post:>9} {clv:>7} {out:<5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Line movement analysis")
    parser.add_argument("--days", type=int, default=30,
                        help="Look back this many days (default: 30)")
    args = parser.parse_args()
    print_movement_report(days=args.days)

"""
Fill the per-game Statcast quality columns on batter_game_logs:
  avg_exit_velo  — mean launch speed on batted balls
  barrel_pct     — barrels / batted balls (launch_speed_angle == 6)
  hard_hit_pct   — batted balls >= 95 mph / batted balls

These are the stickiest power signals available and predict total bases
better than recent results do. Pulled from Baseball Savant via pybaseball.

Usage:
    python ingestion/statcast_batter_data.py                  # last 3 days
    python ingestion/statcast_batter_data.py --date 2026-06-09
    python ingestion/statcast_batter_data.py --backfill 2024 2025 2026

Backfill note: one Savant request per game date (~185/season) with a
politeness delay — a full season takes ~15-30 minutes. Run once, then the
daily loop keeps it current. Re-runs only fetch dates that still have
rows missing exit-velo data.
"""

from __future__ import annotations

import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.schema import init_db, get_engine

BARREL_CODE = 6        # Statcast launch_speed_angle: 6 = barrel
HARD_HIT_MPH = 95.0
SLEEP_BETWEEN_DAYS = 2.0


def _aggregate_day(df: pd.DataFrame) -> pd.DataFrame:
    """Pitch-level Statcast frame → one row per (batter, game_pk)."""
    bb = df[(df["type"] == "X") & df["launch_speed"].notna()].copy()
    if bb.empty:
        return pd.DataFrame()
    # launch_speed_angle can be NA even when launch_speed is present
    # (weak contact the classifier skips) — treat NA as not-a-barrel.
    bb["is_barrel"] = (bb["launch_speed_angle"].fillna(0) == BARREL_CODE).astype(int)
    bb["is_hard"]   = (bb["launch_speed"] >= HARD_HIT_MPH).fillna(False).astype(int)
    out = (
        bb.groupby(["batter", "game_pk"])
        .agg(
            n=("launch_speed", "size"),
            avg_exit_velo=("launch_speed", "mean"),
            barrels=("is_barrel", "sum"),
            hard=("is_hard", "sum"),
        )
        .reset_index()
    )
    out["barrel_pct"]   = out["barrels"] / out["n"]
    out["hard_hit_pct"] = out["hard"] / out["n"]
    return out


def ingest_statcast_day(date_str: str, engine=None, verbose: bool = True) -> int:
    """Pull one date from Savant and update matching batter_game_logs rows."""
    from pybaseball import statcast

    if engine is None:
        engine = init_db()

    try:
        df = statcast(start_dt=date_str, end_dt=date_str, verbose=False)
    except Exception as e:
        print(f"  ⚠ Statcast fetch failed for {date_str}: {e}")
        return 0

    if df is None or df.empty:
        return 0

    agg = _aggregate_day(df)
    if agg.empty:
        return 0

    updated = 0
    with engine.begin() as conn:
        for _, r in agg.iterrows():
            res = conn.execute(text(
                "UPDATE batter_game_logs "
                "SET avg_exit_velo = :ev, barrel_pct = :bp, hard_hit_pct = :hh "
                "WHERE mlbam_id = :pid AND game_pk = :pk"
            ), {
                "ev": round(float(r["avg_exit_velo"]), 2),
                "bp": round(float(r["barrel_pct"]), 4),
                "hh": round(float(r["hard_hit_pct"]), 4),
                "pid": int(r["batter"]),
                "pk": int(r["game_pk"]),
            })
            updated += res.rowcount or 0

    if verbose and updated:
        print(f"  {date_str}: Statcast EV/barrel data on {updated} batter-game rows")
    return updated


def _dates_missing_statcast(engine, season: int) -> list[str]:
    """Game dates in a season that still have rows without exit-velo data."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT game_date FROM batter_game_logs "
            "WHERE season = :s AND avg_exit_velo IS NULL "
            "ORDER BY game_date"
        ), {"s": season}).fetchall()
    return [r.game_date for r in rows]


def backfill_seasons(seasons: list[int]) -> None:
    engine = init_db()
    for season in seasons:
        dates = _dates_missing_statcast(engine, season)
        print(f"\nSeason {season}: {len(dates)} dates need Statcast data")
        for i, d in enumerate(dates, 1):
            ingest_statcast_day(d, engine=engine)
            if i % 20 == 0:
                print(f"  ... {i}/{len(dates)} dates done")
            time.sleep(SLEEP_BETWEEN_DAYS)


def ingest_recent(days: int = 3) -> int:
    """Update the last N days — called from the daily loop."""
    engine = init_db()
    total = 0
    for d in range(1, days + 1):
        date_str = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")
        total += ingest_statcast_day(date_str, engine=engine)
        time.sleep(SLEEP_BETWEEN_DAYS)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest per-game Statcast batter quality data")
    parser.add_argument("--date", type=str, help="Single date YYYY-MM-DD")
    parser.add_argument("--backfill", nargs="+", type=int, metavar="SEASON",
                        help="Backfill whole seasons, e.g. --backfill 2024 2025 2026")
    parser.add_argument("--days", type=int, default=3,
                        help="Default mode: update the last N days (default 3)")
    args = parser.parse_args()

    if args.backfill:
        backfill_seasons(args.backfill)
    elif args.date:
        ingest_statcast_day(args.date)
    else:
        ingest_recent(days=args.days)

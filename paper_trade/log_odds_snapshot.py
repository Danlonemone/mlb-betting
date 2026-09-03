"""
Capture a timestamped snapshot of today's MLB moneyline odds.

Run at three points each day to track how lines move:
  7:00am  → --label open     (first look; launchd job)
  ~10:00am → --label morning (called by update_and_pick.py automatically)
  ~4:00pm  → --label close   (called by capture_clv.py automatically)

Comparing open→morning shows whether lines already moved before we bet.
Comparing morning→close shows whether lines moved with or against us after.

Usage:
    python paper_trade/log_odds_snapshot.py --label open
    python paper_trade/log_odds_snapshot.py --label morning
    python paper_trade/log_odds_snapshot.py --label close
"""

from __future__ import annotations

import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.schema import init_db, get_session, LineSnapshot
from paper_trade.odds_api import fetch_mlb_odds, parse_game_odds, OddsAPIError

VALID_LABELS = {"open", "morning", "close"}


def log_snapshot(label: str, game_date: str | None = None) -> int:
    """
    Fetch current MLB odds and write one LineSnapshot row per game.

    Skips games already snapshotted with this label today (idempotent).
    Returns the number of rows written.
    """
    if label not in VALID_LABELS:
        raise ValueError(f"snapshot_label must be one of {VALID_LABELS}, got '{label}'")

    if game_date is None:
        game_date = datetime.now().strftime("%Y-%m-%d")

    now_iso = datetime.now(timezone.utc).isoformat()

    print(f"\n[Snapshot] label={label}  date={game_date}  time={now_iso[:19]}Z")

    try:
        raw   = fetch_mlb_odds(game_date)
        games = parse_game_odds(raw)
    except OddsAPIError as exc:
        print(f"  ✗ Odds API error: {exc}")
        return 0

    today_games = [g for g in games if g.get("game_date") == game_date]
    if not today_games:
        print(f"  No games with odds found for {game_date}.")
        return 0

    # Resolve game_pk from the DB for today's games
    engine  = init_db()
    session = get_session(engine)

    from sqlalchemy import text
    pk_rows = engine.connect().execute(
        text("SELECT game_pk, home_team, away_team FROM games WHERE game_date = :d"),
        {"d": game_date},
    ).fetchall()
    pk_map: dict[tuple[str, str], int] = {
        (r.home_team, r.away_team): int(r.game_pk) for r in pk_rows
    }

    # Fall back to live MLB schedule for games not yet in DB
    if len(pk_map) < len(today_games):
        try:
            from ingestion.mlb_api import fetch_date_game_map
            live_pks = fetch_date_game_map(game_date)
            for k, v in live_pks.items():
                if k not in pk_map:
                    pk_map[k] = v
        except Exception as exc:
            print(f"  ⚠ Could not fetch live game_pks: {exc}")

    written = skipped = 0
    for g in today_games:
        home = g["home_team"]
        away = g["away_team"]

        existing = (
            session.query(LineSnapshot)
            .filter(
                LineSnapshot.game_date      == game_date,
                LineSnapshot.home_team      == home,
                LineSnapshot.away_team      == away,
                LineSnapshot.snapshot_label == label,
            )
            .first()
        )
        if existing:
            skipped += 1
            continue

        session.add(LineSnapshot(
            game_pk        = pk_map.get((home, away), 0),
            game_date      = game_date,
            home_team      = home,
            away_team      = away,
            snapshot_label = label,
            snapshot_time  = now_iso,
            home_american  = g.get("home_american"),
            away_american  = g.get("away_american"),
            bookmaker      = g.get("bookmaker", ""),
            best_home_american = g.get("best_home_american", g.get("home_american")),
            best_away_american = g.get("best_away_american", g.get("away_american")),
            best_home_book     = g.get("best_home_book", ""),
            best_away_book     = g.get("best_away_book", ""),
        ))
        written += 1

    session.commit()
    session.close()

    print(f"  Wrote {written} snapshot(s)  (skipped {skipped} already captured)")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture MLB odds snapshot")
    parser.add_argument(
        "--label",
        required=True,
        choices=list(VALID_LABELS),
        help="Snapshot label: open | morning | close",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Game date YYYY-MM-DD (defaults to today)",
    )
    args = parser.parse_args()
    log_snapshot(label=args.label, game_date=args.date)

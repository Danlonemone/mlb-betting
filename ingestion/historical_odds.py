"""
Pull historical closing odds from The Odds API and store them in the games table.

The Odds API historical endpoint returns a snapshot of all active odds at a
given ISO timestamp. To get closing lines we query ~15 minutes before each
game's scheduled start time — after that point the book has seen the most
sharp action but before the game begins.

Request cost: 1 request per unique (date, market) query.
For ~800 game-dates across 2019-2024 that's ~800 requests — well within
any paid plan's monthly quota if spread across a few days.

Strategy to minimise requests:
  - Group games by date → one API call per date fetches ALL that day's games
  - Store results immediately so a re-run skips already-filled rows
  - Rate-limit to ~2 req/sec to stay within API guidelines
"""

from __future__ import annotations

import sys
import time
import requests
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ODDS_API_KEY
from db.schema import init_db, get_engine, get_session, Game
from betting.odds import remove_vig, american_to_decimal
from paper_trade.odds_api import ODDS_API_TEAM_MAP, _extract_h2h, PREFERRED_BOOKS

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT    = "baseball_mlb"

# How far before game start to query (minutes). 15 min = near-closing line.
CLOSE_OFFSET_MIN = 15


def _check_key():
    if not ODDS_API_KEY or ODDS_API_KEY == "your_key_here":
        raise RuntimeError(
            "ODDS_API_KEY not set. Add it to mlb_betting/.env before running."
        )


def fetch_historical_snapshot(iso_timestamp: str) -> list[dict] | None:
    """
    Fetch all MLB h2h odds at a specific point in time.
    Returns the raw API response list, or None on failure.
    """
    url = f"{BASE_URL}/historical/sports/{SPORT}/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "h2h",
        "oddsFormat": "american",
        "date":       iso_timestamp,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        remaining = r.headers.get("x-requests-remaining", "?")
        if r.status_code == 422:
            # No data available for this timestamp (before API coverage started)
            return None
        r.raise_for_status()
        data = r.json()
        return data.get("data", data) if isinstance(data, dict) else data
    except requests.RequestException as e:
        print(f"    API error for {iso_timestamp}: {e}")
        return None


def parse_snapshot_for_game(
    events: list[dict],
    home_team: str,
    away_team: str,
) -> tuple[float, float, str] | None:
    """
    Find the closing odds for a specific game in a snapshot response.
    Returns (home_american, away_american, bookmaker) or None.
    """
    # Build reverse map: our abbreviation → full name
    abbr_to_name = {v: k for k, v in ODDS_API_TEAM_MAP.items()}
    home_name = abbr_to_name.get(home_team, home_team)
    away_name = abbr_to_name.get(away_team, away_team)

    for event in events:
        e_home = event.get("home_team", "")
        e_away = event.get("away_team", "")
        if e_home != home_name or e_away != away_name:
            continue

        bookmakers = event.get("bookmakers", [])
        book_map   = {b["key"]: b for b in bookmakers}

        for pref in PREFERRED_BOOKS:
            if pref in book_map:
                h, a, bk = _extract_h2h(book_map[pref], home_name, away_name, pref)
                if h and a:
                    return h, a, bk

        # Fallback to first available
        for book in bookmakers:
            h, a, bk = _extract_h2h(book, home_name, away_name, book["key"])
            if h and a:
                return h, a, bk

    return None


def get_game_start_time(game_pk: int) -> str | None:
    """
    Fetch the scheduled start time of a game from the MLB Stats API.
    Returns ISO 8601 UTC string or None.
    """
    try:
        r = requests.get(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
            headers={"User-Agent": "mlb-betting-model/0.1"},
            timeout=20,
        )
        r.raise_for_status()
        dt_str = r.json().get("gameData", {}).get("datetime", {}).get("dateTime")
        return dt_str  # e.g. "2023-04-01T18:05:00Z"
    except Exception:
        return None


def closing_timestamp(start_iso: str) -> str:
    """
    Return an ISO timestamp CLOSE_OFFSET_MIN before the game starts.
    The Odds API expects UTC in ISO 8601 format.
    """
    dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    close_dt = dt - timedelta(minutes=CLOSE_OFFSET_MIN)
    return close_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_historical_odds(
    seasons: list[int] | None = None,
    max_requests: int = 400,
    delay_sec: float = 0.6,
) -> dict:
    """
    For every game in the games table that is missing closing odds,
    fetch the historical snapshot and store it.

    Parameters
    ----------
    seasons      : restrict to specific seasons (None = all)
    max_requests : safety cap on API calls per run (free tier: stay under quota)
    delay_sec    : pause between API calls (be polite)

    Returns summary dict.
    """
    _check_key()

    engine  = init_db()
    session = get_session(engine)

    # Find games missing closing odds
    query = session.query(Game).filter(Game.home_win.isnot(None),
                                       Game.home_close_american.is_(None))
    if seasons:
        query = query.filter(Game.season.in_(seasons))

    games_to_fill = query.order_by(Game.game_date).all()
    print(f"  Games missing closing odds: {len(games_to_fill)}")
    print(f"  Request cap this run: {max_requests}")

    # Group by date to minimise API calls (one call per date fetches all games)
    by_date: dict[str, list[Game]] = defaultdict(list)
    for g in games_to_fill:
        by_date[g.game_date].append(g)

    filled = failed = skipped = requests_used = 0

    for date_str, day_games in sorted(by_date.items()):
        if requests_used >= max_requests:
            print(f"  ⚠ Reached request cap ({max_requests}). Re-run to continue.")
            break

        # Use start time of the first game on this date for the snapshot
        # (all games on a date are in the same snapshot call anyway)
        sample_game = day_games[0]
        start_iso = get_game_start_time(sample_game.game_pk)

        if not start_iso:
            # Fall back to noon ET on game day
            start_iso = f"{date_str}T17:00:00Z"

        close_ts = closing_timestamp(start_iso)

        snapshot = fetch_historical_snapshot(close_ts)
        requests_used += 1
        time.sleep(delay_sec)

        if snapshot is None:
            print(f"  {date_str}: no snapshot data (API coverage may not reach this date)")
            skipped += len(day_games)
            continue

        # Match each game in the day to the snapshot
        day_filled = 0
        for game in day_games:
            result = parse_snapshot_for_game(snapshot, game.home_team, game.away_team)
            if result is None:
                failed += 1
                continue

            home_am, away_am, book = result
            home_fair, away_fair, overround = remove_vig(home_am, away_am)

            game.home_close_american = home_am
            game.away_close_american = away_am
            game.close_overround     = overround
            game.close_home_fair     = home_fair
            game.close_away_fair     = away_fair
            game.closing_bookmaker   = book
            filled += 1
            day_filled += 1

        session.commit()

        pct = requests_used / max_requests * 100
        print(f"  {date_str}: {day_filled}/{len(day_games)} games filled  "
              f"[{requests_used} req used, {pct:.0f}% of cap]")

    session.close()

    summary = {
        "filled":          filled,
        "failed":          failed,
        "skipped":         skipped,
        "requests_used":   requests_used,
    }
    print(f"\n  Done. Filled {filled} games with real closing odds "
          f"({requests_used} API requests used).")
    return summary


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def odds_coverage_report():
    """Print how many games have real closing odds vs total."""
    engine = get_engine()
    with engine.connect() as conn:
        total   = conn.execute(text("SELECT COUNT(*) FROM games WHERE home_win IS NOT NULL")).scalar()
        covered = conn.execute(text("SELECT COUNT(*) FROM games WHERE home_close_american IS NOT NULL")).scalar()
        by_season = conn.execute(text(
            "SELECT season, "
            "COUNT(*) as total, "
            "SUM(CASE WHEN home_close_american IS NOT NULL THEN 1 ELSE 0 END) as covered "
            "FROM games WHERE home_win IS NOT NULL GROUP BY season ORDER BY season"
        )).fetchall()

    print(f"\nClosing odds coverage: {covered:,} / {total:,} games ({covered/total:.0%})")
    print(f"{'Season':<8} {'Total':<8} {'Covered':<10} {'Pct'}")
    for row in by_season:
        pct = row[2] / row[1] if row[1] else 0
        print(f"  {row[0]:<6} {row[1]:<8} {row[2]:<10} {pct:.0%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest historical closing odds")
    parser.add_argument("--seasons", nargs="+", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=400)
    parser.add_argument("--report", action="store_true",
                        help="Just print coverage report, don't fetch")
    args = parser.parse_args()

    if args.report:
        from db.schema import get_engine
        odds_coverage_report()
    else:
        print("Ingesting historical closing odds from The Odds API...")
        ingest_historical_odds(seasons=args.seasons, max_requests=args.max_requests)
        from db.schema import get_engine
        odds_coverage_report()

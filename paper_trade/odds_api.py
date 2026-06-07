"""
The Odds API client for live MLB moneyline prices.

Free tier: ~500 requests/month. We make at most 2 requests per day
(morning open line + pre-game close line), so the free tier lasts
~8 months of daily use before upgrading.

API docs: https://the-odds-api.com/liveapi/guides/v4/

Team names from The Odds API use full names ("Houston Astros").
We map them to our internal abbreviations via ODDS_API_TEAM_MAP.
"""

from __future__ import annotations

import requests
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ODDS_API_KEY

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT    = "baseball_mlb"

# Full team name → our internal abbreviation
ODDS_API_TEAM_MAP: dict[str, str] = {
    "Los Angeles Angels":        "LAA",
    "Arizona Diamondbacks":      "AZ",
    "Baltimore Orioles":         "BAL",
    "Boston Red Sox":            "BOS",
    "Chicago Cubs":              "CHC",
    "Cincinnati Reds":           "CIN",
    "Cleveland Guardians":       "CLE",
    "Colorado Rockies":          "COL",
    "Detroit Tigers":            "DET",
    "Houston Astros":            "HOU",
    "Kansas City Royals":        "KC",
    "Los Angeles Dodgers":       "LAD",
    "Washington Nationals":      "WSH",
    "New York Mets":             "NYM",
    "Oakland Athletics":         "OAK",
    "Athletics":                 "OAK",   # relocated name variants
    "Las Vegas Athletics":       "OAK",
    "Pittsburgh Pirates":        "PIT",
    "San Diego Padres":          "SD",
    "Seattle Mariners":          "SEA",
    "San Francisco Giants":      "SF",
    "St. Louis Cardinals":       "STL",
    "Tampa Bay Rays":            "TB",
    "Texas Rangers":             "TEX",
    "Toronto Blue Jays":         "TOR",
    "Minnesota Twins":           "MIN",
    "Philadelphia Phillies":     "PHI",
    "Atlanta Braves":            "ATL",
    "Chicago White Sox":         "CWS",
    "Miami Marlins":             "MIA",
    "New York Yankees":          "NYY",
    "Milwaukee Brewers":         "MIL",
}

# Preferred books for consensus line (vig removal / edge calculation)
PREFERRED_BOOKS = [
    "draftkings", "fanduel", "betmgm", "caesars", "pointsbet",
    "williamhill_us", "bovada",
]


def _decimal(american: float) -> float:
    """American odds → decimal (bettor's perspective: higher is better)."""
    if american >= 0:
        return 1.0 + american / 100.0
    return 1.0 - 100.0 / american


def _best_side_odds(
    bookmakers: list[dict],
    team_name: str,
    market_key: str = "h2h",
) -> tuple[float | None, str]:
    """
    Find the highest (best-for-bettor) American odds for one team across all books.
    Returns (american_odds, book_key) or (None, "").
    """
    best: float | None = None
    best_book = ""
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market["key"] != market_key:
                continue
            for outcome in market.get("outcomes", []):
                if outcome["name"] == team_name:
                    price = float(outcome["price"])
                    if best is None or _decimal(price) > _decimal(best):
                        best = price
                        best_book = bm["key"]
    return best, best_book


class OddsAPIError(Exception):
    pass


def _check_key():
    if not ODDS_API_KEY or ODDS_API_KEY == "your_key_here":
        raise OddsAPIError(
            "ODDS_API_KEY not set. Copy .env.example to .env and add your key.\n"
            "Get a free key at https://the-odds-api.com"
        )


def fetch_mlb_odds(
    regions: str = "us",
    markets: str = "h2h",
    odds_format: str = "american",
) -> list[dict]:
    """
    Fetch current MLB moneyline odds for all upcoming games.

    Returns a list of raw event dicts from the API, one per game.
    Each event has: id, sport_key, commence_time, home_team, away_team, bookmakers.
    """
    _check_key()
    url = f"{BASE_URL}/sports/{SPORT}/odds/"
    params = {
        "apiKey":      ODDS_API_KEY,
        "regions":     regions,
        "markets":     markets,
        "oddsFormat":  odds_format,
        "dateFormat":  "iso",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 401:
        raise OddsAPIError("Invalid API key. Check your .env file.")
    if r.status_code == 429:
        raise OddsAPIError("Rate limit hit. Check your request quota at the-odds-api.com.")
    r.raise_for_status()

    remaining = r.headers.get("x-requests-remaining", "?")
    used = r.headers.get("x-requests-used", "?")
    print(f"  Odds API: {used} requests used, {remaining} remaining this month")

    return r.json()


def parse_game_odds(events: list[dict]) -> list[dict]:
    """
    Parse raw API events into clean dicts with our team abbreviations.

    Returns list of dicts:
      game_id, game_date, home_team, away_team,
      home_american, away_american, bookmaker, commence_time
    """
    games = []
    for event in events:
        home_name = event.get("home_team", "")
        away_name = event.get("away_team", "")

        home_abbr = ODDS_API_TEAM_MAP.get(home_name)
        away_abbr = ODDS_API_TEAM_MAP.get(away_name)

        if not home_abbr or not away_abbr:
            print(f"  ⚠ Unknown team name: '{home_name}' or '{away_name}' — skipping")
            continue

        commence_iso = event.get("commence_time", "")
        try:
            commence_dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
            game_date   = commence_dt.astimezone().strftime("%Y-%m-%d")
        except Exception:
            game_date = commence_iso[:10]

        # Consensus line: first preferred book with both sides (used for vig removal)
        bookmakers = event.get("bookmakers", [])
        home_odds, away_odds, book_name = _best_odds(bookmakers, home_name, away_name)

        if home_odds is None or away_odds is None:
            continue

        # Best per-side odds across all books (used for actual bet placement)
        best_home_am, best_home_book = _best_side_odds(bookmakers, home_name, "h2h")
        best_away_am, best_away_book = _best_side_odds(bookmakers, away_name, "h2h")

        games.append({
            "game_id":           event.get("id"),
            "game_date":         game_date,
            "commence_time":     commence_iso,
            "home_team":         home_abbr,
            "away_team":         away_abbr,
            "home_american":     home_odds,
            "away_american":     away_odds,
            "bookmaker":         book_name,
            "best_home_american": best_home_am if best_home_am is not None else home_odds,
            "best_home_book":     best_home_book or book_name,
            "best_away_american": best_away_am if best_away_am is not None else away_odds,
            "best_away_book":     best_away_book or book_name,
        })

    return games


def _best_odds(
    bookmakers: list[dict],
    home_name: str,
    away_name: str,
) -> tuple[float | None, float | None, str]:
    """
    Pick the best-line bookmaker. Falls back to first available if none
    in the preferred list. Returns (home_american, away_american, book_name).
    """
    book_map = {b["key"]: b for b in bookmakers}

    for pref in PREFERRED_BOOKS:
        if pref in book_map:
            return _extract_h2h(book_map[pref], home_name, away_name, pref)

    # Fallback: first available book
    for book in bookmakers:
        result = _extract_h2h(book, home_name, away_name, book["key"])
        if result[0] is not None:
            return result

    return None, None, ""


def _extract_h2h(
    bookmaker: dict,
    home_name: str,
    away_name: str,
    book_key: str,
) -> tuple[float | None, float | None, str]:
    for market in bookmaker.get("markets", []):
        if market["key"] != "h2h":
            continue
        odds_map = {o["name"]: o["price"] for o in market.get("outcomes", [])}
        home = odds_map.get(home_name)
        away = odds_map.get(away_name)
        if home is not None and away is not None:
            return float(home), float(away), book_key
    return None, None, book_key


def fetch_today_odds() -> list[dict]:
    """Convenience wrapper: fetch + parse today's full-game odds."""
    raw = fetch_mlb_odds()
    games = parse_game_odds(raw)
    today = datetime.now().strftime("%Y-%m-%d")
    today_games = [g for g in games if g["game_date"] == today]
    print(f"  Found {len(today_games)} games with odds for {today}")
    return today_games


def _extract_h2h_h1(
    bookmaker: dict,
    home_name: str,
    away_name: str,
    book_key: str,
) -> tuple[float | None, float | None, str]:
    """Extract F5 (h2h_h1 = first half) odds from a bookmaker entry."""
    for market in bookmaker.get("markets", []):
        if market["key"] != "h2h_h1":
            continue
        odds_map = {o["name"]: o["price"] for o in market.get("outcomes", [])}
        home = odds_map.get(home_name)
        away = odds_map.get(away_name)
        if home is not None and away is not None:
            return float(home), float(away), book_key
    return None, None, book_key


def parse_f5_odds(events: list[dict]) -> list[dict]:
    """Parse raw API events for the h2h_h1 (first 5 innings) market."""
    games = []
    for event in events:
        home_name = event.get("home_team", "")
        away_name = event.get("away_team", "")
        home_abbr = ODDS_API_TEAM_MAP.get(home_name)
        away_abbr = ODDS_API_TEAM_MAP.get(away_name)
        if not home_abbr or not away_abbr:
            continue

        commence_iso = event.get("commence_time", "")
        try:
            commence_dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
            game_date   = commence_dt.astimezone().strftime("%Y-%m-%d")
        except Exception:
            game_date = commence_iso[:10]

        bookmakers = event.get("bookmakers", [])
        book_map   = {b["key"]: b for b in bookmakers}

        home_odds = away_odds = book_name = None
        for pref in PREFERRED_BOOKS:
            if pref in book_map:
                h, a, bn = _extract_h2h_h1(book_map[pref], home_name, away_name, pref)
                if h is not None:
                    home_odds, away_odds, book_name = h, a, bn
                    break
        if home_odds is None:
            for book in bookmakers:
                h, a, bn = _extract_h2h_h1(book, home_name, away_name, book["key"])
                if h is not None:
                    home_odds, away_odds, book_name = h, a, bn
                    break

        if home_odds is None:
            continue

        # Best per-side odds across all books
        best_home_am, best_home_book = _best_side_odds(bookmakers, home_name, "h2h_h1")
        best_away_am, best_away_book = _best_side_odds(bookmakers, away_name, "h2h_h1")

        games.append({
            "game_id":           event.get("id"),
            "game_date":         game_date,
            "commence_time":     commence_iso,
            "home_team":         home_abbr,
            "away_team":         away_abbr,
            "home_american":     home_odds,
            "away_american":     away_odds,
            "bookmaker":         book_name,
            "best_home_american": best_home_am if best_home_am is not None else home_odds,
            "best_home_book":     best_home_book or book_name,
            "best_away_american": best_away_am if best_away_am is not None else away_odds,
            "best_away_book":     best_away_book or book_name,
        })

    return games


def fetch_today_f5_odds() -> list[dict]:
    """
    Fetch today's first-5-innings (h2h_h1) odds via the per-event endpoint.
    The main /odds/ endpoint does not support h2h_h1; this fetches one call
    per today's game. Books only post F5 lines for select games and pull them
    closer to first pitch, so empty results on a given day are normal.
    """
    _check_key()
    today = datetime.now().strftime("%Y-%m-%d")

    # Get today's event IDs (one lightweight call)
    r = requests.get(
        f"{BASE_URL}/sports/{SPORT}/events",
        params={"apiKey": ODDS_API_KEY, "dateFormat": "iso"},
        timeout=30,
    )
    r.raise_for_status()
    remaining = r.headers.get("x-requests-remaining", "?")
    used      = r.headers.get("x-requests-used", "?")
    print(f"  Odds API (F5 events): {used} used, {remaining} remaining")

    events = r.json()
    today_events = []
    for e in events:
        try:
            dt = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
            if dt.astimezone().strftime("%Y-%m-%d") == today:
                today_events.append(e)
        except Exception:
            continue

    if not today_events:
        print(f"  No events found for {today}.")
        return []

    games = []
    for event in today_events:
        r2 = requests.get(
            f"{BASE_URL}/sports/{SPORT}/events/{event['id']}/odds",
            params={
                "apiKey":     ODDS_API_KEY,
                "regions":    "us",
                "markets":    "h2h_h1",
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
            timeout=30,
        )
        if not r2.ok:
            continue
        parsed = parse_f5_odds([r2.json()])
        games.extend(parsed)

    today_games = [g for g in games if g["game_date"] == today]
    print(f"  Found {len(today_games)} games with F5 odds for {today}")
    return today_games

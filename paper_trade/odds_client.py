"""
Self-hosted MLB odds client — replaces The Odds API.

Data source: Action Network scoreboard API (free, no key required).
Returns the same schema as the old odds_api.py so all callers work unchanged.

Book IDs used: 15=DraftKings, 30=FanDuel, 76=BetMGM, 75=Caesars,
               123=PointsBet, 69=Bovada, 68=BetOnline, 71=Bet365
"""

from __future__ import annotations

import requests
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Action Network API config
# ---------------------------------------------------------------------------

_AN_URL = "https://api.actionnetwork.com/web/v1/scoreboard/mlb"

_AN_BOOK_IDS: list[int] = [15, 30, 76, 75, 123, 69, 68, 71]

_AN_BOOK_NAMES: dict[int, str] = {
    15:  "draftkings",
    30:  "fanduel",
    76:  "betmgm",
    75:  "caesars",
    123: "pointsbet",
    69:  "bovada",
    68:  "draftkings_nj",    # DraftKings NJ state variant
    71:  "betrivers_nj",     # BetRivers NJ state variant
    59:  "carbon",           # offshore exchange (is_legal=False in AN)
}

# Preferred order for consensus line selection (highest liquidity / sharpest first)
_AN_PREFERRED: list[int] = [15, 30, 76, 75, 123, 69]

# Action Network uses "ATH" for the Athletics; all other abbrevs match our internal format
_AN_ABBR_FIX: dict[str, str] = {"ATH": "OAK"}

# ---------------------------------------------------------------------------
# Backwards-compat exports — used by historical_odds.py and props_picks.py
# ---------------------------------------------------------------------------

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
    "Athletics":                 "OAK",
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

PREFERRED_BOOKS = [
    "draftkings", "fanduel", "betmgm", "caesars", "pointsbet",
    "williamhill_us", "bovada",
]


class OddsAPIError(Exception):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decimal(american: float) -> float:
    if american >= 0:
        return 1.0 + american / 100.0
    return 1.0 - 100.0 / american


def _extract_h2h(
    bookmaker: dict,
    home_name: str,
    away_name: str,
    book_key: str,
) -> tuple[float | None, float | None, str]:
    """Parse a bookmaker dict in The Odds API format. Used by historical_odds.py."""
    for market in bookmaker.get("markets", []):
        if market["key"] != "h2h":
            continue
        odds_map = {o["name"]: o["price"] for o in market.get("outcomes", [])}
        home = odds_map.get(home_name)
        away = odds_map.get(away_name)
        if home is not None and away is not None:
            return float(home), float(away), book_key
    return None, None, book_key


# ---------------------------------------------------------------------------
# Action Network fetcher
# ---------------------------------------------------------------------------

def _fetch_an_games(date_str: str) -> list[dict]:
    """Fetch Action Network scoreboard for a date (YYYY-MM-DD). Raises OddsAPIError on failure."""
    date_compact = date_str.replace("-", "")
    try:
        r = requests.get(
            _AN_URL,
            params={
                "periods": "event",
                "bookIds": ",".join(str(b) for b in _AN_BOOK_IDS),
                "date":    date_compact,
            },
            headers={"User-Agent": "Mozilla/5.0 (compatible; mlb-betting-model/1.0)"},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("games", [])
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise OddsAPIError(f"Action Network request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API — same interface as the old odds_api.py
# ---------------------------------------------------------------------------

def fetch_mlb_odds(date_str: str | None = None) -> list[dict]:
    """
    Fetch MLB game data for the given date (YYYY-MM-DD, defaults to today).
    Returns raw Action Network game dicts; pass to parse_game_odds() to normalize.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    games = _fetch_an_games(date_str)
    print(f"  Action Network: {len(games)} game(s) for {date_str}")
    return games


def parse_game_odds(
    games: list[dict],
    market: str = "game",
) -> list[dict]:
    """
    Parse raw Action Network game dicts into clean game odds records.

    market: "game" for full-game moneyline, "firstfiveinnings" for F5

    Returns list of dicts:
      game_id, game_date, commence_time, home_team, away_team,
      home_american, away_american, bookmaker,
      best_home_american, best_home_book, best_away_american, best_away_book
    """
    result = []
    for g in games:
        teams = {t["id"]: t for t in g.get("teams", [])}
        home_info = teams.get(g.get("home_team_id"), {})
        away_info = teams.get(g.get("away_team_id"), {})

        home_abbr_raw = home_info.get("abbr", "")
        away_abbr_raw = away_info.get("abbr", "")
        home_abbr = _AN_ABBR_FIX.get(home_abbr_raw, home_abbr_raw)
        away_abbr = _AN_ABBR_FIX.get(away_abbr_raw, away_abbr_raw)

        if not home_abbr or not away_abbr:
            print(f"  ⚠ Unknown team in game {g.get('id')} — skipping")
            continue

        start_iso = g.get("start_time", "")
        try:
            commence_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            game_date = commence_dt.astimezone().strftime("%Y-%m-%d")
        except Exception:
            game_date = start_iso[:10]

        # Collect per-book odds for this market type
        book_odds: dict[int, tuple[float, float]] = {}
        for o in g.get("odds", []):
            if o.get("type") != market:
                continue
            ml_home = o.get("ml_home")
            ml_away = o.get("ml_away")
            if ml_home is None or ml_away is None:
                continue
            book_odds[o["book_id"]] = (float(ml_home), float(ml_away))

        if not book_odds:
            continue

        # Consensus line: first preferred book with prices
        home_am = away_am = None
        book_name = ""
        for bid in _AN_PREFERRED:
            if bid in book_odds:
                home_am, away_am = book_odds[bid]
                book_name = _AN_BOOK_NAMES.get(bid, str(bid))
                break
        if home_am is None:
            first_id = next(iter(book_odds))
            home_am, away_am = book_odds[first_id]
            book_name = _AN_BOOK_NAMES.get(first_id, str(first_id))

        # Best per-side odds across all books
        best_home_id = max(book_odds, key=lambda k: _decimal(book_odds[k][0]))
        best_away_id = max(book_odds, key=lambda k: _decimal(book_odds[k][1]))

        result.append({
            "game_id":            str(g["id"]),
            "game_date":          game_date,
            "commence_time":      start_iso,
            "home_team":          home_abbr,
            "away_team":          away_abbr,
            "home_american":      home_am,
            "away_american":      away_am,
            "bookmaker":          book_name,
            "best_home_american": book_odds[best_home_id][0],
            "best_home_book":     _AN_BOOK_NAMES.get(best_home_id, str(best_home_id)),
            "best_away_american": book_odds[best_away_id][1],
            "best_away_book":     _AN_BOOK_NAMES.get(best_away_id, str(best_away_id)),
        })

    return result


def fetch_today_odds() -> list[dict]:
    """Convenience: fetch + parse today's full-game moneyline odds."""
    today = datetime.now().strftime("%Y-%m-%d")
    raw = fetch_mlb_odds(today)
    games = parse_game_odds(raw, market="game")
    today_games = [g for g in games if g["game_date"] == today]
    print(f"  Found {len(today_games)} games with odds for {today}")
    return today_games


def fetch_today_f5_odds() -> list[dict]:
    """Fetch + parse today's first-five-innings moneyline odds."""
    today = datetime.now().strftime("%Y-%m-%d")
    raw = fetch_mlb_odds(today)
    games = parse_game_odds(raw, market="firstfiveinnings")
    today_games = [g for g in games if g["game_date"] == today]
    print(f"  Found {len(today_games)} games with F5 odds for {today}")
    return today_games

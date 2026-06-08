"""
Pull game schedules, results, and probable/actual starting pitchers
from the free MLB Stats API (no key required).

Key endpoints used:
  /api/v1/schedule  - season schedule with probable pitchers and final scores
  /api/v1.1/game/{gamePk}/feed/live - confirmed starters (boxscore)
"""

import requests
import time
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SKIP_SEASONS

BASE_URL = "https://statsapi.mlb.com"
HEADERS = {"User-Agent": "mlb-betting-model/0.1"}

# FanGraphs uses different abbreviations for some teams than the MLB API does
MLB_API_TO_FG: dict[str, str] = {
    "AZ": "ARI", "TB": "TBR", "SD": "SDP", "SF": "SFG",
    "KC": "KCR", "CWS": "CHW", "WSH": "WSN",
}

# Static map of MLB team IDs to abbreviations (stable across seasons)
TEAM_ID_TO_ABBR: dict[int, str] = {
    108: "LAA", 109: "AZ",  110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


def _get(path: str, params: dict = None, retries: int = 3) -> dict:
    url = BASE_URL + path
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def fetch_season_schedule(season: int) -> list[dict]:
    """
    Return a list of dicts, one per regular-season game, with:
      game_pk, game_date, home_team, away_team,
      home_sp_name, home_sp_id, away_sp_name, away_sp_id,
      home_score, away_score, status
    """
    if season in SKIP_SEASONS:
        return []

    data = _get("/api/v1/schedule", params={
        "sportId": 1,
        "season": season,
        "gameType": "R",
        "hydrate": "probablePitcher,linescore",
    })

    games = []
    for date_block in data.get("dates", []):
        game_date = date_block["date"]
        for g in date_block.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            home = g["teams"]["home"]
            away = g["teams"]["away"]

            home_id = home["team"]["id"]
            away_id = away["team"]["id"]

            home_pp = home.get("probablePitcher", {})
            away_pp = away.get("probablePitcher", {})

            # Scores are directly on the team object when the game is final
            home_runs = home.get("score")
            away_runs = away.get("score")

            games.append({
                "game_pk": g["gamePk"],
                "game_date": game_date,
                "season": season,
                "status": status,
                "home_team": TEAM_ID_TO_ABBR.get(home_id, str(home_id)),
                "away_team": TEAM_ID_TO_ABBR.get(away_id, str(away_id)),
                "home_sp_name": home_pp.get("fullName"),
                "home_sp_id": home_pp.get("id"),
                "away_sp_name": away_pp.get("fullName"),
                "away_sp_id": away_pp.get("id"),
                "home_score": home_runs,
                "away_score": away_runs,
            })

    return games


def fetch_actual_starters(game_pk: int) -> dict:
    """
    Return the confirmed starting pitcher IDs from the boxscore
    (only meaningful for completed games).
    """
    data = _get(f"/api/v1.1/game/{game_pk}/feed/live")
    boxscore = data.get("liveData", {}).get("boxscore", {})
    teams = boxscore.get("teams", {})

    result = {}
    for side in ("home", "away"):
        pitchers = teams.get(side, {}).get("pitchers", [])
        pitcher_info = teams.get(side, {}).get("players", {})
        if pitchers:
            starter_id = pitchers[0]
            key = f"ID{starter_id}"
            info = pitcher_info.get(key, {}).get("person", {})
            result[f"{side}_sp_name"] = info.get("fullName")
            result[f"{side}_sp_id"] = starter_id
        else:
            result[f"{side}_sp_name"] = None
            result[f"{side}_sp_id"] = None

    return result


def fetch_season_f5_outcomes(season: int) -> dict[int, dict]:
    """
    Fetch per-inning linescore for all completed games in a season and return
    {game_pk: {"home_score_f5": int, "away_score_f5": int, "home_win_f5": int|None}}.

    home_win_f5 is None when the score is tied after 5 innings (push in the F5 market).
    Games with fewer than 5 completed innings (rain, etc.) are excluded.
    """
    if season in SKIP_SEASONS:
        return {}

    data = _get("/api/v1/schedule", params={
        "sportId":  1,
        "season":   season,
        "gameType": "R",
        "hydrate":  "linescore",
    })

    results: dict[int, dict] = {}
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            if status not in ("Final", "Game Over", "Completed Early"):
                continue
            innings = g.get("linescore", {}).get("innings", [])
            if len(innings) < 5:
                continue
            home_f5 = sum(inn.get("home", {}).get("runs", 0) or 0 for inn in innings[:5])
            away_f5 = sum(inn.get("away", {}).get("runs", 0) or 0 for inn in innings[:5])
            if home_f5 > away_f5:
                win_f5 = 1
            elif away_f5 > home_f5:
                win_f5 = 0
            else:
                win_f5 = None  # tie = push in F5 market
            results[int(g["gamePk"])] = {
                "home_score_f5": home_f5,
                "away_score_f5": away_f5,
                "home_win_f5":   win_f5,
            }
    return results


def fetch_today_lineups(game_date: str) -> dict[int, dict]:
    """
    Fetch confirmed lineup info for all games on game_date.

    When the batting-order lineup is posted (typically 1-2 hours before first
    pitch), probablePitcher is also confirmed. Returns:

        {game_pk: {
            "lineups_posted":  bool,   # True if both lineups have 9 players
            "home_sp_id":      int | None,
            "home_sp_name":    str | None,
            "away_sp_id":      int | None,
            "away_sp_name":    str | None,
        }}

    Falls back gracefully — games where lineups aren't posted yet still appear
    with lineups_posted=False and the probable-pitcher IDs.
    """
    data = _get("/api/v1/schedule", params={
        "sportId":  1,
        "date":     game_date,
        "gameType": "R",
        "hydrate":  "lineups,probablePitcher,officials",
    })

    result: dict[int, dict] = {}
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            game_pk   = g["gamePk"]
            home_pp   = g["teams"]["home"].get("probablePitcher") or {}
            away_pp   = g["teams"]["away"].get("probablePitcher") or {}
            lineups   = g.get("lineups") or {}
            home_batters = lineups.get("homePlayers") or []
            away_batters = lineups.get("awayPlayers") or []
            lineups_posted = len(home_batters) >= 9 and len(away_batters) >= 9

            hp_ump_name = hp_ump_id = None
            for official in g.get("officials", []):
                if official.get("officialType") == "Home Plate":
                    person = official.get("official", {})
                    hp_ump_name = person.get("fullName")
                    hp_ump_id   = person.get("id")
                    break

            result[game_pk] = {
                "lineups_posted": lineups_posted,
                "home_sp_id":    home_pp.get("id"),
                "home_sp_name":  home_pp.get("fullName"),
                "away_sp_id":    away_pp.get("id"),
                "away_sp_name":  away_pp.get("fullName"),
                "hp_ump_name":   hp_ump_name,
                "hp_ump_id":     hp_ump_id,
            }

    return result


def fetch_date_game_map(game_date: str) -> dict[tuple[str, str], int]:
    """
    Return {(home_team, away_team): game_pk} for regular-season games on a date.

    This is intentionally lighter than fetch_season_schedule() and is useful
    for live/pregame workflows where today's games are not yet in the DB.
    """
    data = _get("/api/v1/schedule", params={
        "sportId":  1,
        "date":     game_date,
        "gameType": "R",
    })

    game_map: dict[tuple[str, str], int] = {}
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            home = g.get("teams", {}).get("home", {}).get("team", {})
            away = g.get("teams", {}).get("away", {}).get("team", {})
            home_abbr = TEAM_ID_TO_ABBR.get(home.get("id"), "")
            away_abbr = TEAM_ID_TO_ABBR.get(away.get("id"), "")
            game_pk = g.get("gamePk")
            if home_abbr and away_abbr and game_pk:
                game_map[(home_abbr, away_abbr)] = int(game_pk)
    return game_map


def fetch_seasons(seasons: list[int], verbose: bool = True) -> list[dict]:
    """Pull schedule for multiple seasons, printing progress."""
    all_games = []
    for season in seasons:
        if season in SKIP_SEASONS:
            if verbose:
                print(f"  Skipping {season} (excluded season)")
            continue
        if verbose:
            print(f"  Pulling {season} schedule...", end=" ", flush=True)
        games = fetch_season_schedule(season)
        if verbose:
            print(f"{len(games)} games")
        all_games.extend(games)
        time.sleep(0.5)  # be polite to the API
    return all_games

"""
Pull pitcher and team stats from the official MLB Stats API.
(FanGraphs is blocked for scraping, so pybaseball is not used for stats.)

Park factors come from a hardcoded Baseball Reference table (stable once
the season is over; updated manually each spring).
"""

import requests
import pandas as pd
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HISTORICAL_SEASONS, SKIP_SEASONS, MIN_SP_IP

BASE_URL = "https://statsapi.mlb.com"
HEADERS = {"User-Agent": "mlb-betting-model/0.1"}

# Import the same mapping we use in mlb_api.py
from ingestion.mlb_api import TEAM_ID_TO_ABBR

ABBR_TO_TEAM_ID = {v: k for k, v in TEAM_ID_TO_ABBR.items()}

# FIP constant (league-average ERA - FIP is roughly 3.10; good enough for relative ranking)
_FIP_CONSTANT = 3.10


def _get(path: str, params: dict = None) -> dict:
    r = requests.get(BASE_URL + path, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Park factors — Baseball Reference 5-year regressed (100 = neutral park)
# https://www.baseball-reference.com/leagues/majors/2024-park-factors.shtml
# ---------------------------------------------------------------------------

PARK_FACTORS: dict[tuple[str, int], float] = {
    # (team_abbr, season): basic run park factor
    # 2018
    ("COL", 2018): 115, ("BOS", 2018): 105, ("CIN", 2018): 104, ("TEX", 2018): 104,
    ("CHC", 2018): 103, ("MIL", 2018): 103, ("AZ",  2018): 102, ("BAL", 2018): 102,
    ("PHI", 2018): 102, ("NYY", 2018): 101, ("ATL", 2018): 100, ("CLE", 2018): 100,
    ("HOU", 2018): 100, ("LAA", 2018): 100, ("LAD", 2018): 100, ("MIN", 2018): 100,
    ("NYM", 2018): 100, ("PIT", 2018): 100, ("STL", 2018): 100, ("TOR", 2018): 100,
    ("CWS", 2018): 99,  ("DET", 2018): 99,  ("MIA", 2018): 99,  ("OAK", 2018): 99,
    ("SEA", 2018): 98,  ("SF",  2018): 97,  ("TB",  2018): 97,  ("WSH", 2018): 97,
    ("KC",  2018): 96,  ("SD",  2018): 96,
    # 2019
    ("COL", 2019): 115, ("CIN", 2019): 108, ("TEX", 2019): 107, ("BOS", 2019): 106,
    ("CHC", 2019): 105, ("MIL", 2019): 104, ("AZ",  2019): 103, ("BAL", 2019): 103,
    ("PHI", 2019): 103, ("NYY", 2019): 102, ("ATL", 2019): 101, ("HOU", 2019): 101,
    ("LAA", 2019): 100, ("LAD", 2019): 100, ("MIN", 2019): 100, ("PIT", 2019): 100,
    ("STL", 2019): 100, ("TOR", 2019): 100, ("CLE", 2019): 99,  ("CWS", 2019): 99,
    ("DET", 2019): 99,  ("MIA", 2019): 99,  ("NYM", 2019): 99,  ("OAK", 2019): 99,
    ("SEA", 2019): 98,  ("SF",  2019): 97,  ("TB",  2019): 97,  ("WSH", 2019): 97,
    ("KC",  2019): 96,  ("SD",  2019): 96,
    # 2021
    ("COL", 2021): 113, ("CIN", 2021): 107, ("BOS", 2021): 106, ("TEX", 2021): 105,
    ("CHC", 2021): 104, ("MIL", 2021): 103, ("AZ",  2021): 102, ("BAL", 2021): 102,
    ("PHI", 2021): 102, ("ATL", 2021): 101, ("HOU", 2021): 101, ("NYY", 2021): 101,
    ("LAA", 2021): 100, ("LAD", 2021): 100, ("MIN", 2021): 100, ("PIT", 2021): 100,
    ("STL", 2021): 100, ("TOR", 2021): 100, ("CLE", 2021): 99,  ("CWS", 2021): 99,
    ("DET", 2021): 99,  ("MIA", 2021): 99,  ("NYM", 2021): 99,  ("OAK", 2021): 98,
    ("SEA", 2021): 98,  ("SF",  2021): 97,  ("TB",  2021): 97,  ("WSH", 2021): 97,
    ("KC",  2021): 96,  ("SD",  2021): 95,
    # 2022
    ("COL", 2022): 114, ("CIN", 2022): 108, ("BOS", 2022): 106, ("CHC", 2022): 105,
    ("TEX", 2022): 105, ("MIL", 2022): 104, ("AZ",  2022): 103, ("BAL", 2022): 103,
    ("PHI", 2022): 103, ("ATL", 2022): 102, ("HOU", 2022): 101, ("NYY", 2022): 101,
    ("LAA", 2022): 100, ("LAD", 2022): 100, ("MIN", 2022): 100, ("PIT", 2022): 100,
    ("STL", 2022): 100, ("TOR", 2022): 100, ("CLE", 2022): 99,  ("CWS", 2022): 99,
    ("DET", 2022): 99,  ("MIA", 2022): 99,  ("NYM", 2022): 99,  ("OAK", 2022): 98,
    ("SEA", 2022): 97,  ("SF",  2022): 97,  ("TB",  2022): 97,  ("WSH", 2022): 97,
    ("KC",  2022): 96,  ("SD",  2022): 95,
    # 2023
    ("COL", 2023): 113, ("CIN", 2023): 107, ("BOS", 2023): 106, ("CHC", 2023): 105,
    ("TEX", 2023): 105, ("MIL", 2023): 104, ("AZ",  2023): 103, ("BAL", 2023): 103,
    ("PHI", 2023): 103, ("ATL", 2023): 102, ("HOU", 2023): 101, ("NYY", 2023): 101,
    ("LAA", 2023): 100, ("LAD", 2023): 100, ("MIN", 2023): 100, ("PIT", 2023): 100,
    ("STL", 2023): 100, ("TOR", 2023): 100, ("CLE", 2023): 99,  ("CWS", 2023): 99,
    ("DET", 2023): 99,  ("MIA", 2023): 99,  ("NYM", 2023): 99,  ("OAK", 2023): 98,
    ("SEA", 2023): 97,  ("SF",  2023): 97,  ("TB",  2023): 97,  ("WSH", 2023): 97,
    ("KC",  2023): 96,  ("SD",  2023): 95,
    # 2024
    ("COL", 2024): 113, ("CIN", 2024): 107, ("BOS", 2024): 106, ("CHC", 2024): 105,
    ("TEX", 2024): 104, ("MIL", 2024): 104, ("AZ",  2024): 103, ("BAL", 2024): 103,
    ("PHI", 2024): 103, ("ATL", 2024): 102, ("HOU", 2024): 101, ("NYY", 2024): 101,
    ("LAA", 2024): 100, ("LAD", 2024): 100, ("MIN", 2024): 100, ("PIT", 2024): 100,
    ("STL", 2024): 100, ("TOR", 2024): 100, ("CLE", 2024): 99,  ("CWS", 2024): 99,
    ("DET", 2024): 99,  ("MIA", 2024): 99,  ("NYM", 2024): 99,  ("OAK", 2024): 98,
    ("SEA", 2024): 97,  ("SF",  2024): 97,  ("TB",  2024): 97,  ("WSH", 2024): 97,
    ("KC",  2024): 96,  ("SD",  2024): 95,
    # 2025 — approximate from 2024 (park factors are stable year-over-year)
    ("COL", 2025): 113, ("CIN", 2025): 107, ("BOS", 2025): 106, ("CHC", 2025): 105,
    ("TEX", 2025): 104, ("MIL", 2025): 104, ("AZ",  2025): 103, ("BAL", 2025): 103,
    ("PHI", 2025): 103, ("ATL", 2025): 102, ("HOU", 2025): 101, ("NYY", 2025): 101,
    ("LAA", 2025): 100, ("LAD", 2025): 100, ("MIN", 2025): 100, ("PIT", 2025): 100,
    ("STL", 2025): 100, ("TOR", 2025): 100, ("CLE", 2025): 99,  ("CWS", 2025): 99,
    ("DET", 2025): 99,  ("MIA", 2025): 99,  ("NYM", 2025): 99,  ("OAK", 2025): 98,
    ("SEA", 2025): 97,  ("SF",  2025): 97,  ("TB",  2025): 97,  ("WSH", 2025): 97,
    ("KC",  2025): 96,  ("SD",  2025): 95,
    # 2026 — carried from 2025 until official BBRef values are available
    ("COL", 2026): 113, ("CIN", 2026): 107, ("BOS", 2026): 106, ("CHC", 2026): 105,
    ("TEX", 2026): 104, ("MIL", 2026): 104, ("AZ",  2026): 103, ("BAL", 2026): 103,
    ("PHI", 2026): 103, ("ATL", 2026): 102, ("HOU", 2026): 101, ("NYY", 2026): 101,
    ("LAA", 2026): 100, ("LAD", 2026): 100, ("MIN", 2026): 100, ("PIT", 2026): 100,
    ("STL", 2026): 100, ("TOR", 2026): 100, ("CLE", 2026): 99,  ("CWS", 2026): 99,
    ("DET", 2026): 99,  ("MIA", 2026): 99,  ("NYM", 2026): 99,  ("OAK", 2026): 98,
    ("SEA", 2026): 97,  ("SF",  2026): 97,  ("TB",  2026): 97,  ("WSH", 2026): 97,
    ("KC",  2026): 96,  ("SD",  2026): 95,
}


def normalise_team(abbr: str) -> str:
    from ingestion.mlb_api import MLB_API_TO_FG
    if abbr is None:
        return abbr
    return MLB_API_TO_FG.get(abbr.upper(), abbr.upper())


# ---------------------------------------------------------------------------
# Pitcher stats from MLB Stats API
# ---------------------------------------------------------------------------

def _ip_to_float(ip_str: str) -> float:
    """Convert '100.2' (MLB notation: whole innings + thirds) to decimal."""
    try:
        parts = str(ip_str).split(".")
        full = int(parts[0])
        thirds = int(parts[1]) if len(parts) > 1 else 0
        return full + thirds / 3.0
    except Exception:
        return 0.0


def _compute_fip(hr: int, bb: int, k: int, ip_float: float) -> float | None:
    if ip_float <= 0:
        return None
    return ((13 * hr) + (3 * bb) - (2 * k)) / ip_float + _FIP_CONSTANT


def fetch_pitcher_stats(season: int) -> pd.DataFrame:
    """
    Pull all pitcher season stats from the MLB Stats API.
    Returns one row per pitcher (filtered to starters with MIN_SP_IP).
    Computed fields: k_pct, bb_pct, fip.
    """
    print(f"    MLB API: pitcher stats {season}...", end=" ", flush=True)
    try:
        data = _get("/api/v1/stats", params={
            "season": season,
            "sportId": 1,
            "group": "pitching",
            "stats": "season",
            "playerPool": "All",
            "limit": 2000,
        })
        splits = data["stats"][0]["splits"]

        rows = []
        for s in splits:
            stat = s["stat"]
            player = s["player"]
            team = s.get("team", {})

            ip_float = _ip_to_float(stat.get("inningsPitched", "0"))
            if ip_float < MIN_SP_IP:
                continue

            hr = stat.get("homeRuns", 0)
            bb = stat.get("baseOnBalls", 0)
            k  = stat.get("strikeOuts", 0)
            bf = stat.get("battersFaced", 1) or 1

            rows.append({
                "mlbam_id":   player["id"],
                "name":       player["fullName"],
                "team":       TEAM_ID_TO_ABBR.get(team.get("id"), team.get("name", "")),
                "season":     season,
                "ip":         ip_float,
                "era":        float(stat.get("era", 0) or 0),
                "whip":       float(stat.get("whip", 0) or 0),
                "k_pct":      k / bf,
                "bb_pct":     bb / bf,
                "fip":        _compute_fip(hr, bb, k, ip_float),
                "xfip":       None,   # not available from MLB API; will stay NULL
                "hr9":        float(stat.get("homeRunsPer9", 0) or 0),
                "fangraphs_id": None,
            })

        df = pd.DataFrame(rows)
        print(f"{len(df)} starters")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return pd.DataFrame()


def fetch_all_pitcher_stats(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        df = fetch_pitcher_stats(s)
        if not df.empty:
            frames.append(df)
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Team batting stats from MLB Stats API
# ---------------------------------------------------------------------------

def _compute_woba(stat: dict) -> float | None:
    """
    Compute wOBA from raw counting stats using approximate linear weights.
    wOBA = (0.69*BB + 0.72*HBP + 0.888*1B + 1.271*2B + 1.616*3B + 2.101*HR) / PA
    """
    try:
        pa  = stat.get("plateAppearances", 0)
        if not pa:
            return None
        bb  = stat.get("baseOnBalls", 0)
        hbp = stat.get("hitByPitch", 0)
        h   = stat.get("hits", 0)
        db  = stat.get("doubles", 0)
        tb  = stat.get("triples", 0)
        hr  = stat.get("homeRuns", 0)
        single = h - db - tb - hr
        return (0.69*bb + 0.72*hbp + 0.888*single + 1.271*db + 1.616*tb + 2.101*hr) / pa
    except Exception:
        return None


def fetch_team_batting_v2(season: int) -> pd.DataFrame:
    print(f"    MLB API: team batting {season}...", end=" ", flush=True)
    try:
        data = _get("/api/v1/teams/stats", params={
            "season": season, "sportId": 1, "group": "hitting", "stats": "season"
        })
        splits = data["stats"][0]["splits"]
        rows = []
        for s in splits:
            stat = s["stat"]
            team_id = s["team"]["id"]
            abbr = TEAM_ID_TO_ABBR.get(team_id, str(team_id))
            rows.append({
                "team_abbr": abbr,
                "season":    season,
                "woba":      _compute_woba(stat),
                "wrc_plus":  None,    # not in MLB API; OPS will proxy
                "ops":       float(stat.get("ops", 0) or 0),
                "avg":       float(stat.get("avg", 0) or 0),
                "obp":       float(stat.get("obp", 0) or 0),
                "slg":       float(stat.get("slg", 0) or 0),
            })
        df = pd.DataFrame(rows)
        print(f"{len(df)} teams")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return pd.DataFrame()


def fetch_all_team_batting(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        df = fetch_team_batting_v2(s)
        if not df.empty:
            frames.append(df)
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Team pitching stats from MLB Stats API
# ---------------------------------------------------------------------------

def fetch_team_pitching(season: int) -> pd.DataFrame:
    print(f"    MLB API: team pitching {season}...", end=" ", flush=True)
    try:
        data = _get("/api/v1/teams/stats", params={
            "season": season, "sportId": 1, "group": "pitching", "stats": "season"
        })
        splits = data["stats"][0]["splits"]
        rows = []
        for s in splits:
            stat = s["stat"]
            team_id = s["team"]["id"]
            abbr = TEAM_ID_TO_ABBR.get(team_id, str(team_id))

            hr = stat.get("homeRuns", 0)
            bb = stat.get("baseOnBalls", 0)
            k  = stat.get("strikeOuts", 0)
            ip = _ip_to_float(stat.get("inningsPitched", "0"))

            rows.append({
                "team_abbr": abbr,
                "season":    season,
                "era":       float(stat.get("era", 0) or 0),
                "fip":       _compute_fip(hr, bb, k, ip),
                "whip":      float(stat.get("whip", 0) or 0),
            })
        df = pd.DataFrame(rows)
        print(f"{len(df)} teams")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return pd.DataFrame()


def fetch_all_team_pitching(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        df = fetch_team_pitching(s)
        if not df.empty:
            frames.append(df)
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Park factors — from hardcoded PARK_FACTORS table
# ---------------------------------------------------------------------------

def fetch_park_factors(season: int) -> pd.DataFrame:
    print(f"    Park factors {season}...", end=" ", flush=True)
    rows = [
        {"team_abbr": abbr, "season": yr, "basic_pf": pf}
        for (abbr, yr), pf in PARK_FACTORS.items()
        if yr == season
    ]
    df = pd.DataFrame(rows)
    print(f"{len(df)} parks")
    return df


def fetch_all_park_factors(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        df = fetch_park_factors(s)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Pitcher game logs (per-start) from MLB Stats API
# ---------------------------------------------------------------------------

def fetch_pitcher_game_logs(season: int, mlbam_ids: list[int]) -> pd.DataFrame:
    """
    Pull per-start game logs for a list of starters in a given season.
    Returns one row per qualified start (IP >= 2.0 and gamesStarted = 1).
    """
    print(f"    MLB API: pitcher game logs {season} ({len(mlbam_ids)} pitchers)...")
    rows = []
    for pid in mlbam_ids:
        try:
            data = _get(f"/api/v1/people/{pid}/stats", params={
                "stats":  "gameLog",
                "group":  "pitching",
                "season": season,
            })
            splits = (data.get("stats") or [{}])[0].get("splits", [])
            for s in splits:
                stat = s.get("stat", {})
                if not stat.get("gamesStarted", 0):
                    continue
                ip = _ip_to_float(stat.get("inningsPitched", "0"))
                if ip < 2.0:
                    continue
                rows.append({
                    "mlbam_id":    pid,
                    "game_pk":     (s.get("game") or {}).get("gamePk"),
                    "game_date":   s.get("date", ""),
                    "season":      season,
                    "home_away":   "home" if s.get("isHome") else "away",
                    "ip":          ip,
                    "strikeouts":  int(stat.get("strikeOuts", 0) or 0),
                    "walks":       int(stat.get("baseOnBalls", 0) or 0),
                    "hits":        int(stat.get("hits", 0) or 0),
                    "earned_runs": int(stat.get("earnedRuns", 0) or 0),
                    "pitches":     int(stat.get("numberOfPitches", 0) or 0),
                    "strikes":     int(stat.get("strikes", 0) or 0),
                })
        except Exception:
            continue
        time.sleep(0.2)
    print(f"      {len(rows)} starts found")
    return pd.DataFrame(rows) if rows else pd.DataFrame()

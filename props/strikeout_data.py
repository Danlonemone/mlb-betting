"""
Pull per-start pitcher data for the strikeout prop model.

Data sources (all free):
  1. MLB Stats API — game logs per pitcher per season (IP, K, BB, H, ER, pitches)
  2. Statcast via pybaseball — pitch-level swinging strike rate, CSW% per start

The strikeout prop market sets a line (e.g. 6.5 Ks). We need to model:
  - How many Ks will this pitcher throw in THIS start?

Key features:
  Pitcher-side:
    - Rolling K/9 over last N starts (3, 5, season)
    - SwStr% (swinging strike rate) — best leading indicator of Ks
    - CSW% (called strike + whiff) — even better
    - Recent trend: is K rate going up or down?
    - Innings pitched per start (determines raw K ceiling)
    - Handedness

  Opponent-side:
    - Opposing lineup K% (average K rate vs this pitcher's handedness)
    - Lineup strength (wOBA vs hand)

  Context:
    - Park factor for strikeouts
    - Umpire historical K rate (some umps have much larger strike zones)
    - Game-time temperature (cold = lower K rates slightly)
    - Line itself (over/under 6.5 — book's implied expectation)
"""

from __future__ import annotations

import sys
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.schema import init_db, get_session, PitcherGameLog, PitcherSeason

MLB_BASE = "https://statsapi.mlb.com"
HEADERS  = {"User-Agent": "mlb-betting-model/0.1"}

# Minimum IP per start to count as a true start (not an opener/bulk guy)
MIN_START_IP = 3.0


# ---------------------------------------------------------------------------
# MLB Stats API: per-start game logs
# ---------------------------------------------------------------------------

def fetch_pitcher_game_logs_mlb(
    player_id: int,
    season: int,
) -> list[dict]:
    """
    Fetch per-start game log for a pitcher from the MLB Stats API.
    Returns list of start dicts with date, IP, K, BB, H, ER, pitches.
    """
    try:
        r = requests.get(
            f"{MLB_BASE}/api/v1/people/{player_id}/stats",
            params={
                "stats":  "gameLog",
                "season": season,
                "group":  "pitching",
            },
            headers=HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
    except Exception as e:
        print(f"  ⚠ Could not fetch game log for player {player_id}: {e}")
        return []

    starts = []
    for s in splits:
        stat = s.get("stat", {})
        ip_str = stat.get("inningsPitched", "0")
        ip = _ip_to_float(ip_str)
        if ip < MIN_START_IP:
            continue   # reliever appearance, skip

        game      = s.get("game", {})
        team      = s.get("team", {})
        opponent  = s.get("opponent", {})
        date_str  = s.get("date", "")

        starts.append({
            "game_pk":     game.get("gamePk"),
            "game_date":   date_str,
            "season":      season,
            "ip":          ip,
            "strikeouts":  stat.get("strikeOuts", 0),
            "walks":       stat.get("baseOnBalls", 0),
            "hits":        stat.get("hits", 0),
            "earned_runs": stat.get("earnedRuns", 0),
            "pitches":     stat.get("numberOfPitches", 0),
            "strikes":     stat.get("strikes", 0),
            "home_away":   "home" if s.get("isHome") else "away",
            "opponent":    opponent.get("abbreviation", ""),
        })

    return starts


def _ip_to_float(ip_str: str) -> float:
    try:
        parts = str(ip_str).split(".")
        return int(parts[0]) + (int(parts[1]) if len(parts) > 1 else 0) / 3.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Statcast: swinging strike rate per pitcher per season
# ---------------------------------------------------------------------------

def fetch_swstr_from_statcast(season: int) -> pd.DataFrame:
    """
    Pull pitcher-level SwStr% and CSW% from Baseball Savant.
    Season-level aggregate used as a pitch-quality signal per pitcher.
    """
    import io
    import warnings
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={season}&type=pitcher&filter=&min=10"
        f"&selections=p_swinging_strike,p_called_strike_plus_whiff"
        f"&chart=false&x=p_swinging_strike&z=p_swinging_strike&csv=true"
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = requests.get(
                url, verify=False,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30,
            )
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df = df.rename(columns={
            "player_id":                   "mlbam_id",
            "p_swinging_strike":           "swstr_pct",
            "p_called_strike_plus_whiff":  "csw_pct",
        })
        df["season"] = season
        keep = [c for c in ["mlbam_id", "swstr_pct", "csw_pct", "season"] if c in df.columns]
        return df[keep].dropna(subset=["mlbam_id"])
    except Exception as e:
        print(f"  ⚠ Could not fetch Statcast SwStr data for {season}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Ingest to DB
# ---------------------------------------------------------------------------

def ingest_pitcher_game_logs(seasons: list[int], verbose: bool = True):
    """
    Pull per-start game logs for all pitchers in those seasons and store in DB.
    Only pulls pitchers already in pitcher_seasons table (starters with ≥30 IP).
    """
    engine  = init_db()
    session = get_session(engine)

    # Load Statcast SwStr data for each season (for quality signal)
    swstr_cache: dict[tuple[int, int], dict] = {}
    for season in seasons:
        df = fetch_swstr_from_statcast(season)
        for _, row in df.iterrows():
            swstr_cache[(int(row["mlbam_id"]), season)] = {
                "swstr_pct": row.get("swstr_pct"),
                "csw_pct":   row.get("csw_pct"),
            }
        time.sleep(1)

    # Get unique pitchers for these seasons
    pitcher_rows = (
        session.query(PitcherSeason.mlbam_id, PitcherSeason.name, PitcherSeason.season)
        .filter(
            PitcherSeason.season.in_(seasons),
            PitcherSeason.mlbam_id.isnot(None),
        )
        .all()
    )

    if verbose:
        print(f"  Fetching game logs for {len(pitcher_rows)} pitcher-seasons...")

    inserted = skipped = 0

    for mlbam_id, name, season in pitcher_rows:
        starts = fetch_pitcher_game_logs_mlb(mlbam_id, season)
        swstr  = swstr_cache.get((mlbam_id, season), {})

        for s in starts:
            existing = (
                session.query(PitcherGameLog)
                .filter(
                    PitcherGameLog.mlbam_id == mlbam_id,
                    PitcherGameLog.game_date == s["game_date"],
                )
                .first()
            )
            if existing:
                skipped += 1
                continue

            session.add(PitcherGameLog(
                mlbam_id=mlbam_id,
                player_name=name,
                game_date=s["game_date"],
                game_pk=s["game_pk"],
                season=season,
                opponent=s["opponent"],
                home_away=s["home_away"],
                ip=s["ip"],
                strikeouts=s["strikeouts"],
                walks=s["walks"],
                hits=s["hits"],
                earned_runs=s["earned_runs"],
                pitches=s["pitches"],
                strikes=s["strikes"],
                swstr_pct=swstr.get("swstr_pct"),
                csw_pct=swstr.get("csw_pct"),
            ))
            inserted += 1

        if inserted % 500 == 0 and inserted > 0:
            session.commit()
        time.sleep(0.2)

    session.commit()
    session.close()

    if verbose:
        print(f"  Pitcher game logs: {inserted} inserted, {skipped} already existed.")
    return inserted

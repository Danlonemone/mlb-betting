"""
Pull per-game batter stats from the MLB Stats API and store in batter_game_logs.

Strategy:
  1. Fetch qualified batters (≥100 PA) for each season from the stats leaderboard
  2. Pull per-game logs for each batter via the gameLog endpoint
  3. Upsert into batter_game_logs (mlbam_id + game_date as unique key)

Season coverage: pull as many seasons as needed; recent seasons (2023+) are
the most relevant for the hits model since batter tendencies evolve.
"""

from __future__ import annotations

import sys
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.schema import init_db, get_session, BatterGameLog

MLB_BASE = "https://statsapi.mlb.com"
HEADERS  = {"User-Agent": "mlb-betting-model/0.1"}
MIN_PA   = 100   # minimum plate appearances to count as a qualified batter


def _get(path: str, params: dict | None = None) -> dict:
    r = requests.get(MLB_BASE + path, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_qualified_batters(season: int, min_pa: int = MIN_PA) -> list[tuple[int, str]]:
    """Return list of (mlbam_id, full_name) for qualified batters."""
    data = _get("/api/v1/stats", params={
        "stats":    "season",
        "group":    "hitting",
        "season":   season,
        "sportId":  1,
        "limit":    500,
        "sortStat": "plateAppearances",
        "order":    "desc",
        "hydrate":  "person",
    })
    splits = data.get("stats", [{}])[0].get("splits", [])
    result = []
    for s in splits:
        pa = int(s.get("stat", {}).get("plateAppearances", 0) or 0)
        if pa < min_pa:
            break
        person = s.get("player") or {}
        pid  = person.get("id")
        name = person.get("fullName", "")
        if pid:
            result.append((int(pid), name))
    return result


def fetch_qualified_batter_ids(season: int, min_pa: int = MIN_PA) -> list[int]:
    """
    Return MLBAM IDs for all batters with >= min_pa plate appearances in season.
    Uses the MLB Stats API season hitting leaderboard.
    """
    data = _get("/api/v1/stats", params={
        "stats":    "season",
        "group":    "hitting",
        "season":   season,
        "sportId":  1,
        "limit":    500,
        "sortStat": "plateAppearances",
        "order":    "desc",
    })
    splits = data.get("stats", [{}])[0].get("splits", [])
    ids = []
    for s in splits:
        pa = int(s.get("stat", {}).get("plateAppearances", 0) or 0)
        if pa < min_pa:
            break
        pid = (s.get("player") or {}).get("id")
        if pid:
            ids.append(int(pid))
    return ids


def fetch_batter_game_logs(mlbam_id: int, season: int) -> list[dict]:
    """
    Fetch per-game hitting stats for a batter in a given season.
    Returns list of game dicts.
    """
    try:
        data = _get(f"/api/v1/people/{mlbam_id}/stats", params={
            "stats":  "gameLog",
            "group":  "hitting",
            "season": season,
        })
        splits = data.get("stats", [{}])[0].get("splits", [])
    except Exception as e:
        print(f"  ⚠ game log fetch failed for {mlbam_id}: {e}")
        return []

    rows = []
    for s in splits:
        st = s.get("stat", {})
        ab = int(st.get("atBats", 0) or 0)
        pa = int(st.get("plateAppearances", 0) or 0)
        if pa < 1:
            continue
        game  = s.get("game", {})
        team  = s.get("team", {})
        opp   = s.get("opponent", {})
        rows.append({
            "mlbam_id":    mlbam_id,
            "game_pk":     game.get("gamePk"),
            "game_date":   s.get("date", ""),
            "season":      season,
            "team":        team.get("abbreviation", ""),
            "opponent":    opp.get("abbreviation", ""),
            "home_away":   "home" if s.get("isHome") else "away",
            "ab":          ab,
            "pa":          pa,
            "hits":        int(st.get("hits", 0) or 0),
            "doubles":     int(st.get("doubles", 0) or 0),
            "triples":     int(st.get("triples", 0) or 0),
            "home_runs":   int(st.get("homeRuns", 0) or 0),
            "total_bases": int(st.get("totalBases", 0) or 0),
            "strikeouts":  int(st.get("strikeOuts", 0) or 0),
            "walks":       int(st.get("baseOnBalls", 0) or 0),
        })
    return rows


def ingest_batter_game_logs(
    seasons: list[int],
    min_pa: int = MIN_PA,
    verbose: bool = True,
) -> int:
    """
    Pull and store batter game logs for given seasons.
    Returns total rows inserted.
    """
    engine  = init_db()
    session = get_session(engine)

    total_inserted = 0

    for season in seasons:
        if verbose:
            print(f"\n[Batters] Fetching {season} qualified batters (min {min_pa} PA)...")
        batters = fetch_qualified_batters(season, min_pa=min_pa)
        if verbose:
            print(f"  {len(batters)} qualified batters")

        # Build name lookup for this season
        id_to_name = {pid: name for pid, name in batters}

        inserted = skipped = 0
        for mlbam_id, _ in batters:
            logs = fetch_batter_game_logs(mlbam_id, season)
            batter_name = id_to_name.get(mlbam_id, "")
            for row in logs:
                existing = (
                    session.query(BatterGameLog)
                    .filter(
                        BatterGameLog.mlbam_id  == mlbam_id,
                        BatterGameLog.game_date == row["game_date"],
                    )
                    .first()
                )
                if existing:
                    skipped += 1
                    continue
                session.add(BatterGameLog(
                    mlbam_id    = mlbam_id,
                    player_name = batter_name,
                    game_pk     = row["game_pk"],
                    game_date   = row["game_date"],
                    season      = row["season"],
                    team        = row["team"],
                    opponent    = row["opponent"],
                    home_away   = row["home_away"],
                    ab          = row["ab"],
                    pa          = row["pa"],
                    hits        = row["hits"],
                    doubles     = row["doubles"],
                    triples     = row["triples"],
                    home_runs   = row["home_runs"],
                    total_bases = row["total_bases"],
                    strikeouts  = row["strikeouts"],
                    walks       = row["walks"],
                ))
                inserted += 1

            if inserted % 2000 == 0 and inserted > 0:
                session.commit()
            time.sleep(0.15)

        session.commit()
        total_inserted += inserted
        if verbose:
            print(f"  {season}: {inserted} rows inserted, {skipped} skipped")

    session.close()
    return total_inserted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    parser.add_argument("--min-pa",  type=int, default=MIN_PA)
    args = parser.parse_args()
    total = ingest_batter_game_logs(args.seasons, min_pa=args.min_pa)
    print(f"\nDone. {total:,} rows inserted total.")

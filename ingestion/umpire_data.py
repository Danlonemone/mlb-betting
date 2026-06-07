"""
Fetch and store home plate umpire assignments and career tendency stats.

Data flow:
  1. ingest_ump_assignments([seasons]) — pulls ump names/IDs from MLB Stats API
     schedule with hydrate=officials; upserts into game_umpires.
  2. compute_ump_stats(engine) — joins game_umpires+games to get career run
     scoring tendency; joins game_umpires+pitcher_game_logs for K tendency.
     Upserts into umpire_stats (one row per ump, career aggregates).

Run once historically to backfill, then daily via update_and_pick.py.
"""

from __future__ import annotations

import sys
import time
import requests
from pathlib import Path
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.schema import init_db, get_session, GameUmpire, UmpireStat
from config import SKIP_SEASONS

MLB_BASE     = "https://statsapi.mlb.com"
HEADERS      = {"User-Agent": "mlb-betting-model/0.1"}
MIN_START_IP = 3.0   # minimum IP to count a start for K-rate computation
MIN_GAMES    = 10    # minimum career games to compute meaningful ump stats


def fetch_ump_assignments(season: int) -> list[dict]:
    """
    Return [{game_pk, ump_name, ump_id}, ...] for all regular-season games
    in the given season that have a home plate umpire recorded.
    """
    if season in SKIP_SEASONS:
        return []
    r = requests.get(
        f"{MLB_BASE}/api/v1/schedule",
        params={"sportId": 1, "season": season, "gameType": "R", "hydrate": "officials"},
        headers=HEADERS,
        timeout=60,
    )
    r.raise_for_status()

    assignments = []
    for date_block in r.json().get("dates", []):
        for g in date_block.get("games", []):
            for official in g.get("officials", []):
                if official.get("officialType") == "Home Plate":
                    person = official.get("official", {})
                    ump_name = person.get("fullName", "").strip()
                    if ump_name:
                        assignments.append({
                            "game_pk":  int(g["gamePk"]),
                            "ump_name": ump_name,
                            "ump_id":   person.get("id"),
                        })
                    break  # one HP ump per game
    return assignments


def ingest_ump_assignments(seasons: list[int]) -> int:
    """
    Fetch and upsert home plate ump assignments for the given seasons.
    Skips game_pks already in the table (PK is game_pk).
    Returns total rows inserted across all seasons.
    """
    engine  = init_db()
    session = get_session(engine)
    total   = 0

    for season in sorted(seasons):
        print(f"\n[Umps] Fetching {season} assignments...", end=" ", flush=True)
        try:
            assignments = fetch_ump_assignments(season)
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        inserted = 0
        for a in assignments:
            if session.get(GameUmpire, a["game_pk"]):
                continue
            session.add(GameUmpire(
                game_pk  = a["game_pk"],
                ump_name = a["ump_name"],
                ump_id   = a["ump_id"],
            ))
            inserted += 1
        session.commit()
        total += inserted
        print(f"{len(assignments)} games found, {inserted} new")
        time.sleep(0.5)

    session.close()
    return total


def compute_ump_stats(engine=None) -> None:
    """
    Compute career ump tendencies and upsert into umpire_stats.

    run_scoring: joins game_umpires + games → avg total runs/game vs league avg
    k_tendency:  joins game_umpires + pitcher_game_logs → avg SP K9 vs league avg
    """
    if engine is None:
        engine = init_db()

    with engine.connect() as conn:
        run_rows = conn.execute(text("""
            SELECT u.ump_name,
                   COUNT(*)                          AS games,
                   AVG(g.home_score + g.away_score)  AS runs_pg
            FROM   game_umpires u
            JOIN   games g ON u.game_pk = g.game_pk
            WHERE  g.home_score IS NOT NULL
               AND g.away_score IS NOT NULL
               AND u.ump_name != ''
            GROUP  BY u.ump_name
            HAVING COUNT(*) >= :min_games
        """), {"min_games": MIN_GAMES}).fetchall()

        league_runs = conn.execute(text(
            "SELECT AVG(home_score + away_score) FROM games WHERE home_score IS NOT NULL"
        )).scalar() or 9.0

        k_rows = conn.execute(text("""
            SELECT u.ump_name,
                   SUM(p.strikeouts) * 9.0 / SUM(p.ip) AS k9
            FROM   game_umpires u
            JOIN   pitcher_game_logs p ON u.game_pk = p.game_pk
            WHERE  p.ip >= :min_ip
               AND p.strikeouts IS NOT NULL
               AND u.ump_name != ''
            GROUP  BY u.ump_name
            HAVING SUM(p.ip) >= 50
        """), {"min_ip": MIN_START_IP}).fetchall()

        league_k9 = conn.execute(text(
            "SELECT SUM(strikeouts)*9.0/SUM(ip) FROM pitcher_game_logs WHERE ip >= :m"
        ), {"m": MIN_START_IP}).scalar() or 9.0

    k9_by_ump = {r.ump_name: float(r.k9) for r in k_rows}

    session = get_session(engine)
    for r in run_rows:
        runs_vs_avg = round(float(r.runs_pg) - float(league_runs), 4)
        k9          = k9_by_ump.get(r.ump_name)
        k9_vs_avg   = round(float(k9) - float(league_k9), 4) if k9 is not None else None

        existing = session.query(UmpireStat).filter_by(ump_name=r.ump_name).first()
        if existing:
            existing.games       = int(r.games)
            existing.runs_pg     = round(float(r.runs_pg), 4)
            existing.runs_vs_avg = runs_vs_avg
            existing.k9          = round(float(k9), 4) if k9 else None
            existing.k9_vs_avg   = k9_vs_avg
        else:
            session.add(UmpireStat(
                ump_name    = r.ump_name,
                games       = int(r.games),
                runs_pg     = round(float(r.runs_pg), 4),
                runs_vs_avg = runs_vs_avg,
                k9          = round(float(k9), 4) if k9 else None,
                k9_vs_avg   = k9_vs_avg,
            ))

    session.commit()
    session.close()
    print(
        f"\n[Umps] Stats computed for {len(run_rows)} umpires  "
        f"(league avg {league_runs:.2f} runs/game, {league_k9:.2f} K9)"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest umpire assignments and compute stats")
    parser.add_argument("--seasons", nargs="+", type=int,
                        default=[2019, 2021, 2022, 2023, 2024, 2025, 2026])
    args = parser.parse_args()

    n = ingest_ump_assignments(args.seasons)
    print(f"\nTotal new assignments inserted: {n}")
    compute_ump_stats()

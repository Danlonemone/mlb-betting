"""
Build the clean per-game table in SQLite.

Strategy for look-ahead bias:
  - For a game in season Y, we join pitcher and team stats from season Y-1.
  - Rest days are computed from each team's prior game in the season schedule.

This is provably look-ahead-bias-free at the cost of using slightly stale signals.
Phase 3 can enhance with rolling season-to-date stats.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import HISTORICAL_SEASONS, SKIP_SEASONS
from db.schema import init_db, get_session, Game, PitcherSeason, TeamSeason, ParkFactor, PitcherGameLog
from ingestion.mlb_api import fetch_seasons, fetch_season_f5_outcomes
from ingestion.pybaseball_pull import (
    fetch_all_pitcher_stats,
    fetch_all_team_batting,
    fetch_all_team_pitching,
    fetch_all_park_factors,
    fetch_pitcher_stats,
    fetch_team_batting_v2,
    fetch_team_pitching,
    fetch_park_factors,
    fetch_pitcher_game_logs,
    PARK_FACTORS,
)


# ---------------------------------------------------------------------------
# Helpers: insert rows into the lookup tables
# ---------------------------------------------------------------------------

def _upsert_pitcher(session, row: dict, overwrite: bool = False):
    existing = (
        session.query(PitcherSeason)
        .filter(
            PitcherSeason.mlbam_id == row["mlbam_id"],
            PitcherSeason.season == row["season"],
        )
        .first()
    )
    if existing:
        if not overwrite:
            return
        existing.ip     = row.get("ip")
        existing.era    = row.get("era")
        existing.fip    = row.get("fip")
        existing.xfip   = row.get("xfip")
        existing.k_pct  = row.get("k_pct")
        existing.bb_pct = row.get("bb_pct")
        existing.whip   = row.get("whip")
        existing.hr9    = row.get("hr9")
        return
    obj = PitcherSeason(
        mlbam_id=row.get("mlbam_id"),
        fangraphs_id=row.get("fangraphs_id"),
        name=row.get("name"),
        season=int(row["season"]),
        team=row.get("team"),
        ip=row.get("ip"),
        era=row.get("era"),
        fip=row.get("fip"),
        xfip=row.get("xfip"),
        k_pct=row.get("k_pct"),
        bb_pct=row.get("bb_pct"),
        whip=row.get("whip"),
        hr9=row.get("hr9"),
    )
    session.add(obj)


def _upsert_team_batting(session, row: dict, overwrite: bool = False):
    existing = (
        session.query(TeamSeason)
        .filter(
            TeamSeason.team_abbr == row["team_abbr"],
            TeamSeason.season == row["season"],
            TeamSeason.stat_type == "batting",
        )
        .first()
    )
    if existing:
        if not overwrite:
            return
        existing.woba    = row.get("woba")
        existing.wrc_plus = row.get("wrc_plus")
        existing.ops     = row.get("ops")
        existing.avg     = row.get("avg")
        return
    obj = TeamSeason(
        team_abbr=row["team_abbr"],
        season=int(row["season"]),
        stat_type="batting",
        woba=row.get("woba"),
        wrc_plus=row.get("wrc_plus"),
        ops=row.get("ops"),
        avg=row.get("avg"),
    )
    session.add(obj)


def _upsert_team_pitching(session, row: dict, overwrite: bool = False):
    existing = (
        session.query(TeamSeason)
        .filter(
            TeamSeason.team_abbr == row["team_abbr"],
            TeamSeason.season == row["season"],
            TeamSeason.stat_type == "pitching",
        )
        .first()
    )
    if existing:
        if not overwrite:
            return
        existing.era  = row.get("era")
        existing.fip  = row.get("fip")
        existing.whip = row.get("whip")
        return
    obj = TeamSeason(
        team_abbr=row["team_abbr"],
        season=int(row["season"]),
        stat_type="pitching",
        era=row.get("era"),
        fip=row.get("fip"),
        whip=row.get("whip"),
    )
    session.add(obj)


def _upsert_game_log(session, row: dict):
    existing = (
        session.query(PitcherGameLog)
        .filter(
            PitcherGameLog.mlbam_id == row["mlbam_id"],
            PitcherGameLog.game_date == row["game_date"],
        )
        .first()
    )
    if existing:
        return
    obj = PitcherGameLog(
        mlbam_id=int(row["mlbam_id"]),
        game_date=row["game_date"],
        game_pk=row.get("game_pk"),
        season=row.get("season"),
        home_away=row.get("home_away"),
        ip=row.get("ip"),
        strikeouts=row.get("strikeouts"),
        walks=row.get("walks"),
        hits=row.get("hits"),
        earned_runs=row.get("earned_runs"),
        pitches=row.get("pitches"),
        strikes=row.get("strikes"),
    )
    session.add(obj)


def _upsert_park_factor(session, row: dict):
    existing = (
        session.query(ParkFactor)
        .filter(
            ParkFactor.team_abbr == row["team_abbr"],
            ParkFactor.season == row["season"],
        )
        .first()
    )
    if existing:
        return
    obj = ParkFactor(
        team_abbr=row["team_abbr"],
        season=int(row["season"]),
        basic_pf=row.get("basic_pf"),
    )
    session.add(obj)


# ---------------------------------------------------------------------------
# Step 1: Ingest raw stats into the DB
# ---------------------------------------------------------------------------

def ingest_pitcher_stats(session, seasons):
    print("\n[1/4] Fetching pitcher stats...")
    df = fetch_all_pitcher_stats(seasons)
    if df.empty:
        print("  No pitcher data returned.")
        return
    for _, row in df.iterrows():
        _upsert_pitcher(session, row.to_dict())
    session.commit()
    print(f"  Saved {len(df)} pitcher-season rows.")


def ingest_team_batting(session, seasons):
    print("\n[2/4] Fetching team batting stats...")
    df = fetch_all_team_batting(seasons)
    if df.empty:
        print("  No team batting data returned.")
        return
    for _, row in df.iterrows():
        _upsert_team_batting(session, row.to_dict())
    session.commit()
    print(f"  Saved {len(df)} team-batting-season rows.")


def ingest_team_pitching(session, seasons):
    print("\n[3/4] Fetching team pitching stats...")
    df = fetch_all_team_pitching(seasons)
    if df.empty:
        print("  No team pitching data returned.")
        return
    for _, row in df.iterrows():
        _upsert_team_pitching(session, row.to_dict())
    session.commit()
    print(f"  Saved {len(df)} team-pitching-season rows.")


def ingest_park_factors(session, seasons):
    print("\n[4/4] Loading park factors...")
    df = fetch_all_park_factors(seasons)
    if df.empty:
        print("  No park factor data.")
        return
    for _, row in df.iterrows():
        _upsert_park_factor(session, row.to_dict())
    session.commit()
    print(f"  Saved {len(df)} park factor rows.")


def ingest_pitcher_game_logs(session, seasons: list[int]) -> None:
    """
    Pull per-start game logs for all starters in the given seasons and upsert
    into pitcher_game_logs. Uses pitcher_seasons as the source of MLBAM IDs.
    """
    print("\n[Game Logs] Fetching SP per-start game logs...")
    for season in seasons:
        mlbam_ids = [
            row[0]
            for row in session.query(PitcherSeason.mlbam_id)
            .filter(PitcherSeason.season == season, PitcherSeason.mlbam_id.isnot(None))
            .all()
        ]
        if not mlbam_ids:
            print(f"  No pitcher IDs for {season}, skipping.")
            continue
        df = fetch_pitcher_game_logs(season, mlbam_ids)
        if df.empty:
            print(f"  No game logs returned for {season}.")
            continue
        for _, row in df.iterrows():
            _upsert_game_log(session, row.to_dict())
        session.commit()
        print(f"  {season}: {len(df)} starts upserted.")


def ingest_current_season_stats(session, season: int) -> None:
    """
    Fetch and upsert current-season (in-progress) pitcher and team stats.

    Uses overwrite=True so mid-season re-runs update accumulating stats rather
    than silently skipping rows that already exist from an earlier run.
    Park factors are static — they are only inserted if not already present.
    """
    print(f"\n[Current season] Refreshing {season} stats for live picks...")

    df_p = fetch_pitcher_stats(season)
    if not df_p.empty:
        for _, row in df_p.iterrows():
            _upsert_pitcher(session, row.to_dict(), overwrite=True)
        session.commit()
        print(f"  Pitcher stats: {len(df_p)} starters upserted.")

    df_bat = fetch_team_batting_v2(season)
    if not df_bat.empty:
        for _, row in df_bat.iterrows():
            _upsert_team_batting(session, row.to_dict(), overwrite=True)
        session.commit()
        print(f"  Team batting:  {len(df_bat)} teams upserted.")

    df_pit = fetch_team_pitching(season)
    if not df_pit.empty:
        for _, row in df_pit.iterrows():
            _upsert_team_pitching(session, row.to_dict(), overwrite=True)
        session.commit()
        print(f"  Team pitching: {len(df_pit)} teams upserted.")

    df_pf = fetch_park_factors(season)
    if not df_pf.empty:
        for _, row in df_pf.iterrows():
            _upsert_park_factor(session, row.to_dict())
        session.commit()
        print(f"  Park factors:  {len(df_pf)} parks (insert-only).")

    ingest_pitcher_game_logs(session, [season])


# ---------------------------------------------------------------------------
# Step 2: Compute rest days
# ---------------------------------------------------------------------------

def compute_rest_days(games: list[dict]) -> dict:
    """
    Returns {game_pk: {home_rest_days, away_rest_days}}.
    Rest = calendar days since each team's last game (capped at 10).
    """
    sorted_games = sorted(games, key=lambda g: (g["game_date"], g["game_pk"]))
    last_date: dict[str, str] = {}
    rest = {}

    for g in sorted_games:
        pk   = g["game_pk"]
        date = g["game_date"]
        home = g["home_team"]
        away = g["away_team"]

        def days_since(team):
            if team not in last_date:
                return None
            delta = datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(last_date[team], "%Y-%m-%d")
            return min(delta.days, 10)

        rest[pk] = {"home_rest_days": days_since(home), "away_rest_days": days_since(away)}
        last_date[home] = date
        last_date[away] = date

    return rest


# ---------------------------------------------------------------------------
# Step 3: Lookup helpers
# ---------------------------------------------------------------------------

def _pitcher_lookup_cache(session, season: int) -> dict[int, dict]:
    """Build a {mlbam_id -> stats_dict} cache for one season."""
    rows = session.query(PitcherSeason).filter(PitcherSeason.season == season).all()
    return {
        r.mlbam_id: {"era": r.era, "fip": r.fip, "xfip": r.xfip,
                     "k_pct": r.k_pct, "bb_pct": r.bb_pct, "ip": r.ip}
        for r in rows if r.mlbam_id is not None
    }


def _team_batting_cache(session, season: int) -> dict[str, dict]:
    rows = (session.query(TeamSeason)
            .filter(TeamSeason.season == season, TeamSeason.stat_type == "batting")
            .all())
    return {r.team_abbr: {"woba": r.woba, "wrc_plus": r.wrc_plus} for r in rows}


def _team_pitching_cache(session, season: int) -> dict[str, dict]:
    rows = (session.query(TeamSeason)
            .filter(TeamSeason.season == season, TeamSeason.stat_type == "pitching")
            .all())
    return {r.team_abbr: {"era": r.era, "fip": r.fip} for r in rows}


def _park_factor_cache(session, season: int) -> dict[str, float]:
    rows = session.query(ParkFactor).filter(ParkFactor.season == season).all()
    return {r.team_abbr: r.basic_pf for r in rows}


def _latest_park_factor_cache(session, season: int) -> dict[str, float]:
    """
    Return park factors for season, or the most recent prior season available.

    Official current-year park factors are usually unavailable during the season,
    but park run environments are stable enough that last year's values are a
    better live fallback than neutral 100s for every team.
    """
    cache = _park_factor_cache(session, season)
    if cache:
        return cache

    prior = (
        session.query(ParkFactor.season)
        .filter(ParkFactor.season < season)
        .order_by(ParkFactor.season.desc())
        .first()
    )
    if not prior:
        return {}
    return _park_factor_cache(session, int(prior[0]))


# ---------------------------------------------------------------------------
# Step 4: Build the games table
# ---------------------------------------------------------------------------

def build_games(session, seasons: list[int]):
    print("\nPulling MLB schedule from Stats API...")
    all_games = fetch_seasons(seasons)

    completed = [
        g for g in all_games
        if g["status"] in ("Final", "Game Over", "Completed Early")
        and g["home_score"] is not None
        and g["season"] not in SKIP_SEASONS
    ]
    print(f"  {len(completed)} completed games across seasons {seasons}")

    print("Computing rest days...")
    rest_map = compute_rest_days(all_games)

    # Pre-load stat caches per season (Y-1)
    prior_seasons = sorted({g["season"] - 1 for g in completed})
    pitcher_caches = {s: _pitcher_lookup_cache(session, s) for s in prior_seasons}
    bat_caches     = {s: _team_batting_cache(session, s) for s in prior_seasons}
    pit_caches     = {s: _team_pitching_cache(session, s) for s in prior_seasons}
    pf_caches      = {s: _latest_park_factor_cache(session, s) for s in set(g["season"] for g in completed)}

    print("Building per-game feature rows...")
    inserted = 0
    skipped  = 0

    for g in tqdm(completed):
        pk     = g["game_pk"]
        season = g["season"]
        prior  = season - 1
        home   = g["home_team"]
        away   = g["away_team"]

        if session.query(Game).filter(Game.game_pk == pk).first():
            skipped += 1
            continue

        rest    = rest_map.get(pk, {})
        p_cache = pitcher_caches.get(prior, {})
        b_cache = bat_caches.get(prior, {})
        pp_cache = pit_caches.get(prior, {})
        pf_cache = pf_caches.get(season, {})

        home_sp = p_cache.get(g["home_sp_id"], {})
        away_sp = p_cache.get(g["away_sp_id"], {})
        home_bat = b_cache.get(home, {})
        away_bat = b_cache.get(away, {})
        home_pit = pp_cache.get(home, {})
        away_pit = pp_cache.get(away, {})
        pf       = pf_cache.get(home)

        h_score = g["home_score"]
        a_score = g["away_score"]

        session.add(Game(
            game_pk=pk,
            game_date=g["game_date"],
            season=season,
            home_team=home,
            away_team=away,
            home_sp_name=g["home_sp_name"],
            away_sp_name=g["away_sp_name"],
            home_sp_mlbam_id=g["home_sp_id"],
            away_sp_mlbam_id=g["away_sp_id"],
            home_sp_era=home_sp.get("era"),
            home_sp_fip=home_sp.get("fip"),
            home_sp_xfip=home_sp.get("xfip"),
            home_sp_k_pct=home_sp.get("k_pct"),
            home_sp_bb_pct=home_sp.get("bb_pct"),
            home_sp_ip=home_sp.get("ip"),
            away_sp_era=away_sp.get("era"),
            away_sp_fip=away_sp.get("fip"),
            away_sp_xfip=away_sp.get("xfip"),
            away_sp_k_pct=away_sp.get("k_pct"),
            away_sp_bb_pct=away_sp.get("bb_pct"),
            away_sp_ip=away_sp.get("ip"),
            home_woba=home_bat.get("woba"),
            home_wrc_plus=home_bat.get("wrc_plus"),
            away_woba=away_bat.get("woba"),
            away_wrc_plus=away_bat.get("wrc_plus"),
            home_team_era=home_pit.get("era"),
            home_team_fip=home_pit.get("fip"),
            away_team_era=away_pit.get("era"),
            away_team_fip=away_pit.get("fip"),
            park_factor=pf,
            home_rest_days=rest.get("home_rest_days"),
            away_rest_days=rest.get("away_rest_days"),
            home_score=h_score,
            away_score=a_score,
            home_win=1 if h_score > a_score else 0,
            data_source="historical",
        ))
        inserted += 1

        if inserted % 500 == 0:
            session.commit()

    session.commit()
    print(f"  Inserted {inserted} games, skipped {skipped} (already in DB).")
    return inserted


# ---------------------------------------------------------------------------
# F5 outcome ingestion
# ---------------------------------------------------------------------------

def ingest_f5_outcomes(session, seasons: list[int]) -> None:
    """
    Fetch per-inning linescores and backfill home_score_f5 / away_score_f5 /
    home_win_f5 on the games table.  Only updates rows where home_win_f5 is
    currently NULL so re-runs are safe (ties stay NULL intentionally).
    """
    print("\n[F5] Ingesting first-5-inning outcomes...")
    for season in seasons:
        print(f"  Fetching {season} linescores...", end=" ", flush=True)
        outcomes = fetch_season_f5_outcomes(season)
        if not outcomes:
            print("no data.")
            continue
        updated = 0
        for game_pk, data in outcomes.items():
            game = session.query(Game).filter(Game.game_pk == game_pk).first()
            if game and game.home_win_f5 is None and game.home_win is not None:
                game.home_score_f5 = data["home_score_f5"]
                game.away_score_f5 = data["away_score_f5"]
                game.home_win_f5   = data["home_win_f5"]
                updated += 1
        session.commit()
        print(f"{updated} games updated.")


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------

def print_summary(session):
    from sqlalchemy import func, text

    total = session.query(func.count(Game.id)).scalar()
    if not total:
        print("No games in DB yet.")
        return

    seasons = session.query(Game.season, func.count(Game.id)).group_by(Game.season).all()
    home_win_rate = session.query(func.avg(Game.home_win)).scalar()

    print(f"\n{'='*55}")
    print(f"Games table: {total:,} rows  |  Home win rate: {home_win_rate:.1%}")
    print(f"{'Season':<8} {'Games':<10} {'Home W%'}")
    for season, count in sorted(seasons):
        hw = session.query(func.avg(Game.home_win)).filter(Game.season == season).scalar()
        print(f"  {season:<6} {count:<10} {hw:.1%}")

    cols = ["home_sp_fip", "home_woba", "park_factor", "home_rest_days"]
    print("\nFeature fill rates:")
    for col in cols:
        n_null = session.execute(
            text(f"SELECT COUNT(*) FROM games WHERE {col} IS NULL")
        ).scalar()
        pct = round((1 - n_null / total) * 100, 1)
        bar = "#" * int(pct / 5)
        print(f"  {col:<22} {pct:>5.1f}%  [{bar:<20}]")
    print("="*55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-2025", action="store_true",
        help="Pull 2025 stats and game results for live 2026 paper trading"
    )
    parser.add_argument(
        "--refresh-season", type=int,
        help=(
            "Pull one season's completed games into the games table. "
            "Also ensures prior-season pitcher/team stats are present."
        ),
    )
    args = parser.parse_args()

    print("Initialising database...")
    engine = init_db()
    session = get_session(engine)

    if args.refresh_season:
        season = args.refresh_season
        prior = season - 1
        print(f"Refreshing {season} games for live paper trading...")
        print(f"Ensuring {prior} prior-season stats are present...")
        ingest_pitcher_stats(session, [prior])
        ingest_team_batting(session, [prior])
        ingest_team_pitching(session, [prior])
        ingest_park_factors(session, [prior])
        build_games(session, [season])
    elif args.refresh_2025:
        print("Refreshing 2025 stats for live paper trading...")
        ingest_pitcher_stats(session, [2025])
        ingest_team_batting(session, [2025])
        ingest_team_pitching(session, [2025])
        ingest_park_factors(session, [2025])
        build_games(session, [2025])
    else:
        # Pull stats for ALL seasons including 2020 — we skip 2020 games but need
        # 2020 stats as the prior-season signal for 2021 games.
        stat_seasons = list(range(2018, 2026))   # includes 2020 and 2025
        game_seasons = [s for s in HISTORICAL_SEASONS if s not in SKIP_SEASONS]

        ingest_pitcher_stats(session, stat_seasons)
        ingest_team_batting(session, stat_seasons)
        ingest_team_pitching(session, stat_seasons)
        ingest_park_factors(session, stat_seasons)

        build_games(session, game_seasons)

    print_summary(session)
    session.close()
    print("\nPhase 0 complete. Database written to data/mlb_betting.db")

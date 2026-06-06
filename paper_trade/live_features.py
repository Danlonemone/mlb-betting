"""
Build model features for TODAY's upcoming games.

For live games we follow the same look-ahead-bias-safe rule as the backtest:
use prior-season stats (most recent completed season in the DB).

Flow:
  1. MLB Stats API → today's schedule with probable starters
  2. DB lookup → prior-season pitcher stats (by MLBAM ID)
  3. DB lookup → prior-season team batting/pitching stats
  4. DB lookup → park factor for home team
  5. Compute rest days from each team's last game in the DB

Returns a list of feature dicts ready for recommender.recommend().
"""

from __future__ import annotations

import sys
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import text, func

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.schema import get_engine, get_session, PitcherSeason, TeamSeason, ParkFactor, Game
from features.engineering import (
    FEATURE_COLS, F5_FEATURE_COLS,
    load_game_log_cache, recent_sp_stats,
    build_bullpen_cache, recent_bullpen_ip,
    load_team_bullpen_cache,
)
from ingestion.mlb_api import fetch_season_schedule, fetch_today_lineups, TEAM_ID_TO_ABBR


def _prior_season(engine) -> int:
    """Return the most recent season in the pitcher_seasons table."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MAX(season) FROM pitcher_seasons")).scalar()
    return int(result) if result else 2024


def _last_game_date(session, team: str, before_date: str) -> str | None:
    """Return the date of the most recent settled game for a team before a given date."""
    row = (
        session.query(Game.game_date)
        .filter(
            ((Game.home_team == team) | (Game.away_team == team)),
            Game.game_date < before_date,
            Game.home_win.isnot(None),
        )
        .order_by(Game.game_date.desc())
        .first()
    )
    return row[0] if row else None


def _rest_days(last_date: str | None, game_date: str) -> int | None:
    if not last_date:
        return None
    d = datetime.strptime(game_date, "%Y-%m-%d") - datetime.strptime(last_date, "%Y-%m-%d")
    return min(d.days, 10)


def _blank_team_state() -> dict:
    return {
        "games": 0,
        "wins": 0,
        "runs_for": 0,
        "runs_against": 0,
        "home_games": 0,
        "home_wins": 0,
        "away_games": 0,
        "away_wins": 0,
        "recent": deque(maxlen=10),
        "recent_scored": deque(maxlen=7),
        "recent_allowed": deque(maxlen=7),
    }


def _shrunken_pct(wins: int, games: int, prior_games: int = 10) -> float:
    return (wins + 0.5 * prior_games) / (games + prior_games)


def _build_team_form_cache(engine, season: int, before_date: str) -> dict[str, dict]:
    """Build team form from settled games before the target date."""
    with engine.connect() as conn:
        games = conn.execute(text(
            "SELECT game_date, home_team, away_team, home_score, away_score, home_win "
            "FROM games "
            "WHERE season = :season "
            "AND game_date < :before_date "
            "AND home_win IS NOT NULL "
            "ORDER BY game_date, game_pk"
        ), {"season": season, "before_date": before_date}).fetchall()

    states: dict[str, dict] = {}
    for g in games:
        home = g.home_team
        away = g.away_team
        home_win = int(g.home_win)
        away_win = 1 - home_win

        home_state = states.setdefault(home, _blank_team_state())
        away_state = states.setdefault(away, _blank_team_state())

        home_state["games"] += 1
        home_state["wins"] += home_win
        home_state["runs_for"] += int(g.home_score)
        home_state["runs_against"] += int(g.away_score)
        home_state["home_games"] += 1
        home_state["home_wins"] += home_win
        home_state["recent"].append(home_win)
        home_state["recent_scored"].append(int(g.home_score))
        home_state["recent_allowed"].append(int(g.away_score))

        away_state["games"] += 1
        away_state["wins"] += away_win
        away_state["runs_for"] += int(g.away_score)
        away_state["runs_against"] += int(g.home_score)
        away_state["away_games"] += 1
        away_state["away_wins"] += away_win
        away_state["recent"].append(away_win)
        away_state["recent_scored"].append(int(g.away_score))
        away_state["recent_allowed"].append(int(g.home_score))

    return states


def _team_form_values(states: dict[str, dict], team: str, side: str) -> dict:
    state = states.get(team) or _blank_team_state()
    games = state["games"]
    run_diff_pg = (
        (state["runs_for"] - state["runs_against"]) / games
        if games
        else 0.0
    )
    recent_games = len(state["recent"])
    recent_wins = sum(state["recent"])

    if side == "home":
        split_wins = state["home_wins"]
        split_games = state["home_games"]
    else:
        split_wins = state["away_wins"]
        split_games = state["away_games"]

    n_scored  = len(state["recent_scored"])
    n_allowed = len(state["recent_allowed"])
    recent_scored_pg  = sum(state["recent_scored"])  / n_scored  if n_scored  else 4.5
    recent_allowed_pg = sum(state["recent_allowed"]) / n_allowed if n_allowed else 4.5

    return {
        "season_win_pct": _shrunken_pct(state["wins"], games),
        "season_run_diff_pg": run_diff_pg,
        "recent_win_pct": _shrunken_pct(recent_wins, recent_games, prior_games=6),
        "split_win_pct": _shrunken_pct(split_wins, split_games, prior_games=8),
        "recent_scored_pg": recent_scored_pg,
        "recent_allowed_pg": recent_allowed_pg,
    }


def build_live_features(
    odds_games: list[dict],
    game_date: str | None = None,
) -> list[dict]:
    """
    Given a list of game dicts from parse_game_odds(), build full feature
    dicts ready for the recommender.

    Parameters
    ----------
    odds_games  : output of paper_trade.odds_api.parse_game_odds()
    game_date   : date string "YYYY-MM-DD" (defaults to today)

    Returns
    -------
    List of dicts with all FEATURE_COLS plus game metadata and odds.
    """
    if game_date is None:
        game_date = datetime.now().strftime("%Y-%m-%d")

    engine  = get_engine()
    session = get_session(engine)
    prior   = _prior_season(engine)
    season  = int(game_date[:4])
    team_form = _build_team_form_cache(engine, season, game_date)
    game_log_cache     = load_game_log_cache(engine)
    bullpen_cache      = build_bullpen_cache(engine, game_log_cache)
    team_bullpen_cache = load_team_bullpen_cache(engine)

    # Pre-load pitcher stats cache for prior season
    pitcher_cache: dict[int, dict] = {}
    for p in session.query(PitcherSeason).filter(PitcherSeason.season == prior).all():
        if p.mlbam_id:
            pitcher_cache[p.mlbam_id] = {
                "era": p.era, "fip": p.fip, "xfip": p.xfip,
                "k_pct": p.k_pct, "bb_pct": p.bb_pct, "ip": p.ip,
            }

    # Team batting and pitching caches
    bat_cache: dict[str, dict] = {}
    for t in session.query(TeamSeason).filter(
        TeamSeason.season == prior, TeamSeason.stat_type == "batting"
    ).all():
        bat_cache[t.team_abbr] = {"woba": t.woba, "wrc_plus": t.wrc_plus}

    pit_cache: dict[str, dict] = {}
    for t in session.query(TeamSeason).filter(
        TeamSeason.season == prior, TeamSeason.stat_type == "pitching"
    ).all():
        pit_cache[t.team_abbr] = {"era": t.era, "fip": t.fip}

    pf_cache: dict[str, float] = {}
    for pf in session.query(ParkFactor).filter(ParkFactor.season == prior).all():
        pf_cache[pf.team_abbr] = pf.basic_pf

    # Fetch today's schedule to get probable pitcher MLBAM IDs
    year = int(game_date[:4])
    print(f"  Fetching today's schedule for probable starters (season {year})...")
    try:
        schedule = fetch_season_schedule(year)
    except Exception as e:
        print(f"  ⚠ Could not fetch schedule: {e}")
        schedule = []

    # Build a map: (home_team, away_team) → schedule game dict
    today_schedule = {
        (g["home_team"], g["away_team"]): g
        for g in schedule
        if g["game_date"] == game_date
    }

    # Overlay confirmed lineups when posted — overrides probable SP IDs
    try:
        confirmed = fetch_today_lineups(game_date)
    except Exception as e:
        print(f"  ⚠ Could not fetch confirmed lineups: {e}")
        confirmed = {}

    n_confirmed = sum(1 for v in confirmed.values() if v["lineups_posted"])
    if confirmed:
        print(f"  Confirmed lineups: {n_confirmed}/{len(confirmed)} games")

    feature_rows = []
    for odds_game in odds_games:
        home = odds_game["home_team"]
        away = odds_game["away_team"]

        schedule_game    = today_schedule.get((home, away), {})
        game_pk          = schedule_game.get("game_pk", 0)
        conf             = confirmed.get(game_pk, {})
        lineup_confirmed = conf.get("lineups_posted", False)

        # Prefer confirmed lineup SP over probable when available
        if lineup_confirmed:
            home_sp_id   = conf.get("home_sp_id")   or schedule_game.get("home_sp_id")
            away_sp_id   = conf.get("away_sp_id")   or schedule_game.get("away_sp_id")
            home_sp_name = conf.get("home_sp_name") or schedule_game.get("home_sp_name") or "TBD"
            away_sp_name = conf.get("away_sp_name") or schedule_game.get("away_sp_name") or "TBD"
            # Warn if confirmed SP differs from probable
            prob_home_id = schedule_game.get("home_sp_id")
            prob_away_id = schedule_game.get("away_sp_id")
            if prob_home_id and home_sp_id != prob_home_id:
                print(f"  ⚠ SP change {home}: probable={schedule_game.get('home_sp_name')} → confirmed={home_sp_name}")
            if prob_away_id and away_sp_id != prob_away_id:
                print(f"  ⚠ SP change {away}: probable={schedule_game.get('away_sp_name')} → confirmed={away_sp_name}")
        else:
            home_sp_id   = schedule_game.get("home_sp_id")
            away_sp_id   = schedule_game.get("away_sp_id")
            home_sp_name = schedule_game.get("home_sp_name") or "TBD"
            away_sp_name = schedule_game.get("away_sp_name") or "TBD"

        home_sp = pitcher_cache.get(home_sp_id, {})
        away_sp = pitcher_cache.get(away_sp_id, {})
        home_bat = bat_cache.get(home, {})
        away_bat = bat_cache.get(away, {})
        home_pit = pit_cache.get(home, {})
        away_pit = pit_cache.get(away, {})
        pf = pf_cache.get(home, 100.0)

        # Rest days from DB
        home_last = _last_game_date(session, home, game_date)
        away_last = _last_game_date(session, away, game_date)
        home_rest = _rest_days(home_last, game_date)
        away_rest = _rest_days(away_last, game_date)
        home_form = _team_form_values(team_form, home, side="home")
        away_form = _team_form_values(team_form, away, side="away")

        # Build feature dict (matching engineering.FEATURE_COLS)
        sp_data_available = float(bool(home_sp and away_sp))

        # Recent SP form (last 5 starts before today)
        h_recent = recent_sp_stats(home_sp_id, game_date, game_log_cache)
        a_recent = recent_sp_stats(away_sp_id, game_date, game_log_cache)
        if h_recent and a_recent:
            sp_recent_era_diff  = h_recent["era"] - a_recent["era"]
            sp_recent_k9_diff   = h_recent["k9"]  - a_recent["k9"]
            sp_recent_form_data = 1.0
        else:
            sp_recent_era_diff  = 0.0
            sp_recent_k9_diff   = 0.0
            sp_recent_form_data = 0.0

        # Bullpen freshness (last 3 calendar days)
        home_bp_ip = recent_bullpen_ip(home, game_date, bullpen_cache)
        away_bp_ip = recent_bullpen_ip(away, game_date, bullpen_cache)
        if home_bp_ip is not None and away_bp_ip is not None:
            bullpen_freshness_diff  = away_bp_ip - home_bp_ip
            bullpen_data_available  = 1.0
        else:
            bullpen_freshness_diff  = 0.0
            bullpen_data_available  = 0.0

        # Bullpen quality (prior-season reliever ERA/FIP)
        home_bp = team_bullpen_cache.get((home, prior))
        away_bp = team_bullpen_cache.get((away, prior))
        if home_bp and away_bp:
            bullpen_era_diff = (home_bp.get("era") or 0.0) - (away_bp.get("era") or 0.0)
            bullpen_fip_diff = (home_bp.get("fip") or 0.0) - (away_bp.get("fip") or 0.0)
        else:
            bullpen_era_diff = 0.0
            bullpen_fip_diff = 0.0

        row = {
            # Features
            "sp_fip_diff":        (home_sp.get("fip", 0) or 0) - (away_sp.get("fip", 0) or 0),
            "sp_k_pct_diff":      (home_sp.get("k_pct", 0) or 0) - (away_sp.get("k_pct", 0) or 0),
            "sp_bb_pct_diff":     (home_sp.get("bb_pct", 0) or 0) - (away_sp.get("bb_pct", 0) or 0),
            "sp_era_diff":        (home_sp.get("era", 0) or 0) - (away_sp.get("era", 0) or 0),
            "sp_data_available":  sp_data_available,
            "woba_diff":          (home_bat.get("woba", 0) or 0) - (away_bat.get("woba", 0) or 0),
            "team_era_diff":      (home_pit.get("era", 0) or 0) - (away_pit.get("era", 0) or 0),
            "team_fip_diff":      (home_pit.get("fip", 0) or 0) - (away_pit.get("fip", 0) or 0),
            "park_factor":        pf or 100.0,
            "rest_diff":          (home_rest or 1) - (away_rest or 1),
            "season_win_pct_diff": home_form["season_win_pct"] - away_form["season_win_pct"],
            "season_run_diff_pg_diff": home_form["season_run_diff_pg"] - away_form["season_run_diff_pg"],
            "recent_win_pct_diff": home_form["recent_win_pct"] - away_form["recent_win_pct"],
            "home_away_win_pct_diff": home_form["split_win_pct"] - away_form["split_win_pct"],
            "recent_scored_pg_diff": home_form["recent_scored_pg"] - away_form["recent_scored_pg"],
            "recent_allowed_pg_diff": home_form["recent_allowed_pg"] - away_form["recent_allowed_pg"],
            "sp_recent_era_diff":       sp_recent_era_diff,
            "sp_recent_k9_diff":        sp_recent_k9_diff,
            "sp_recent_form_data":      sp_recent_form_data,
            "bullpen_freshness_diff":   bullpen_freshness_diff,
            "bullpen_data_available":   bullpen_data_available,
            "bullpen_era_diff":         bullpen_era_diff,
            "bullpen_fip_diff":         bullpen_fip_diff,

            # Metadata
            "game_pk":            game_pk,
            "game_date":          game_date,
            "home_team":          home,
            "away_team":          away,
            "home_sp_name":       home_sp_name,
            "away_sp_name":       away_sp_name,
            "home_sp_id":         home_sp_id,
            "away_sp_id":         away_sp_id,
            "lineup_confirmed":   lineup_confirmed,

            # Odds (from The Odds API)
            "home_american_odds": odds_game["home_american"],
            "away_american_odds": odds_game["away_american"],
            "bookmaker":          odds_game.get("bookmaker", ""),
            "commence_time":      odds_game.get("commence_time", ""),
        }

        feature_rows.append(row)

    session.close()

    if feature_rows:
        print(f"  Built features for {len(feature_rows)} games "
              f"(prior season: {prior})")
        sp_fill = sum(1 for r in feature_rows if r["sp_data_available"]) / len(feature_rows)
        n_conf  = sum(1 for r in feature_rows if r.get("lineup_confirmed"))
        print(f"  SP data available: {sp_fill:.0%}  |  Lineups confirmed: {n_conf}/{len(feature_rows)}")

    return feature_rows


def build_live_f5_features(
    f5_odds_games: list[dict],
    game_date: str | None = None,
) -> list[dict]:
    """
    Same as build_live_features but uses F5 odds and returns rows scoped to
    F5_FEATURE_COLS.  Reuses the full feature dict then drops the bullpen cols.
    """
    full_rows = build_live_features(f5_odds_games, game_date=game_date)
    _f5_drop = {"team_era_diff", "team_fip_diff", "bullpen_freshness_diff", "bullpen_data_available"}
    return [{k: v for k, v in row.items() if k not in _f5_drop} for row in full_rows]

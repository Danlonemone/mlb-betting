"""
Refresh live data, retrain the moneyline model, and run today's picks.

This is the one-command daily loop:
  1. Pull completed games for the current season.
  2. Retrain the saved model using games strictly before the pick date.
  3. Fetch today's odds and run paper-trading recommendations.

The date cutoff matters. If the script runs after an early game has already
finished, that same-date result is excluded from retraining and live rolling
features, preserving the morning-line assumption for the rest of the slate.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MIN_EDGE, DEFAULT_BANKROLL, get_current_bankroll
from db.schema import init_db, get_session, PitcherSeason, TeamSeason, ParkFactor, ShadowPick
from ingestion.build_game_table import (
    ingest_pitcher_stats,
    ingest_team_batting,
    ingest_team_pitching,
    ingest_park_factors,
    ingest_pitcher_game_logs,
    ingest_f5_outcomes,
    ingest_current_season_stats,
    build_games,
    print_summary,
)
from ingestion.batter_data import ingest_batter_game_logs
from ingestion.umpire_data import ingest_ump_assignments, compute_ump_stats
from model.train import train_final_model, train_f5_model
from paper_trade.daily_picks import run_daily_picks


def _has_prior_stats(session, prior_season: int) -> bool:
    pitcher_count = (
        session.query(PitcherSeason)
        .filter(PitcherSeason.season == prior_season)
        .count()
    )
    batting_count = (
        session.query(TeamSeason)
        .filter(TeamSeason.season == prior_season, TeamSeason.stat_type == "batting")
        .count()
    )
    pitching_count = (
        session.query(TeamSeason)
        .filter(TeamSeason.season == prior_season, TeamSeason.stat_type == "pitching")
        .count()
    )
    park_count = (
        session.query(ParkFactor)
        .filter(ParkFactor.season == prior_season)
        .count()
    )
    return (
        pitcher_count > 0
        and batting_count >= 30
        and pitching_count >= 30
        and park_count >= 30
    )


def refresh_current_season(season: int) -> None:
    engine = init_db()
    session = get_session(engine)
    prior = season - 1

    print(f"\n{'='*60}")
    print(f"Refreshing {season} completed games")
    print(f"{'='*60}")

    if _has_prior_stats(session, prior):
        print(f"Prior-season stats for {prior} already present.")
    else:
        print(f"Ensuring {prior} prior-season stats are present...")
        ingest_pitcher_stats(session, [prior])
        ingest_team_batting(session, [prior])
        ingest_team_pitching(session, [prior])
        ingest_park_factors(session, [prior])

    build_games(session, [season])
    ingest_current_season_stats(session, season)
    ingest_f5_outcomes(session, [season])
    ingest_batter_game_logs([season])
    ingest_ump_assignments([season])
    compute_ump_stats(engine)

    # Per-game Statcast quality (exit velo / barrels) for the TB model
    try:
        from ingestion.statcast_batter_data import ingest_recent
        ingest_recent(days=2)
    except Exception as exc:
        print(f"  [Statcast] Skipped: {exc}")
    print_summary(session)
    session.close()


def _log_shadow_picks(picks: list[dict], pick_date: str) -> None:
    engine = init_db()
    session = get_session(engine)
    logged = 0
    for p in picks:
        game_pk = p.get("game_pk")
        if not game_pk:
            continue
        existing = session.query(ShadowPick).filter_by(game_pk=game_pk).first()
        if existing:
            continue
        session.add(ShadowPick(
            game_pk=game_pk,
            game_date=pick_date,
            home_team=p.get("home_team"),
            away_team=p.get("away_team"),
            bet_side=p.get("bet_side"),
            model_prob=p.get("model_prob"),
            fair_prob=p.get("fair_prob"),
            edge=p.get("edge"),
            home_american_open=p.get("home_american_open"),
            away_american_open=p.get("away_american_open"),
            bet_american_odds=p.get("american_odds"),
            bet_decimal_odds=p.get("decimal_odds"),
            stake_dollars=p.get("stake"),
            bookmaker=p.get("bookmaker", ""),
            created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        ))
        logged += 1
    session.commit()
    session.close()
    print(f"  [Shadow] {logged} pick(s) logged to shadow_picks.")


def run_update_and_picks(
    bankroll: float,
    min_edge: float,
    model_type: str,
    shadow: bool,
    skip_refresh: bool,
    skip_retrain: bool,
    pick_date: str,
) -> list[dict]:
    season = int(pick_date[:4])

    if not skip_refresh:
        refresh_current_season(season)

    if not skip_retrain:
        train_final_model(model_type=model_type, before_date=pick_date)
        train_f5_model(model_type="logistic", before_date=pick_date)

    # Log morning odds snapshot for line movement tracking
    try:
        from paper_trade.log_odds_snapshot import log_snapshot
        log_snapshot(label="morning", game_date=pick_date)
    except Exception as exc:
        print(f"  [Snapshot] Skipped: {exc}")

    # Always dry-run paper_bets — user places bets manually via the dashboard.
    ml_picks = run_daily_picks(
        bankroll=bankroll,
        min_edge=min_edge,
        model_type=model_type,
        dry_run=True,
    )

    if shadow and ml_picks:
        _log_shadow_picks(ml_picks, pick_date)

    # Props always dry-run — user places manually.
    try:
        from props.props_picks import run_props_picks
        run_props_picks(
            markets=["pitcher_strikeouts", "batter_total_bases"],
            bankroll=bankroll,
            min_edge=min_edge,
            dry_run=True,
        )
    except Exception as exc:
        print(f"\n  [Props] Skipped: {exc}")

    return ml_picks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refresh MLB data, retrain the model, and run today's picks."
    )
    parser.add_argument("--bankroll", type=float, default=get_current_bankroll())
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE)
    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=["logistic", "xgboost"],
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Pick date used as the retraining cutoff. Defaults to today.",
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="Log model picks to shadow_picks table for tracking (never touches paper_bets).",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Do not pull current-season completed games before retraining.",
    )
    parser.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Use the existing saved model without retraining.",
    )
    args = parser.parse_args()

    run_update_and_picks(
        bankroll=args.bankroll,
        min_edge=args.min_edge,
        model_type=args.model,
        shadow=args.shadow,
        skip_refresh=args.skip_refresh,
        skip_retrain=args.skip_retrain,
        pick_date=args.date,
    )

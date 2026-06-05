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

from config import MIN_EDGE, DEFAULT_BANKROLL
from db.schema import init_db, get_session, PitcherSeason, TeamSeason, ParkFactor
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
    print_summary(session)
    session.close()


def run_update_and_picks(
    bankroll: float,
    min_edge: float,
    model_type: str,
    log_bets: bool,
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

    return run_daily_picks(
        bankroll=bankroll,
        min_edge=min_edge,
        model_type=model_type,
        dry_run=not log_bets,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refresh MLB data, retrain the model, and run today's picks."
    )
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
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
        "--log-bets",
        action="store_true",
        help="Log recommended bets to paper_bets. Default is dry-run.",
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
        log_bets=args.log_bets,
        skip_refresh=args.skip_refresh,
        skip_retrain=args.skip_retrain,
        pick_date=args.date,
    )

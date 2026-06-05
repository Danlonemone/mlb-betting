"""
Walk-forward backtest harness.

Each fold:
  - Train model on seasons [s0 ... sY-1]
  - Price games in sY with the synthetic market (log5, prior-season W%)
  - Generate bets where model_edge >= MIN_EDGE
  - Settle against real game outcomes

The backtest intentionally uses the exact same walk-forward splits as
Phase 1 training so there is zero data leakage: the model has never
seen outcomes for the season it is predicting.
"""

from __future__ import annotations

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MIN_EDGE, KELLY_FRACTION
from db.schema import get_engine
from features.engineering import load_feature_matrix, build_features, load_games, FEATURE_COLS
from model.train import make_logistic, make_xgb, SEASONS_IN_ORDER
from backtest.market import SyntheticMarket
from betting.odds import remove_vig, compute_edge, american_to_decimal, format_american
from betting.kelly import kelly_stake


def _load_closing_odds(engine) -> dict[int, dict]:
    """
    Load real closing odds from the games table.
    Returns {game_pk: {home_american, away_american, home_fair, away_fair}}.
    Only returns rows where closing odds are populated.
    """
    with engine.connect() as conn:
        from sqlalchemy import text
        rows = conn.execute(text(
            "SELECT game_pk, home_close_american, away_close_american, "
            "close_home_fair, close_away_fair "
            "FROM games WHERE home_close_american IS NOT NULL"
        )).fetchall()
    return {
        r[0]: {
            "home_american": r[1], "away_american": r[2],
            "home_fair": r[3],     "away_fair": r[4],
        }
        for r in rows
    }

MODEL_DIR = Path(__file__).parent.parent / "model"


# ---------------------------------------------------------------------------
# Bet record
# ---------------------------------------------------------------------------

@dataclass
class BetRecord:
    game_pk:          int
    game_date:        str
    season:           int
    home_team:        str
    away_team:        str
    bet_side:         str        # "home" or "away"
    model_prob:       float      # model's probability for the bet side
    market_prob:      float      # market's probability for the bet side (pre-vig)
    fair_prob:        float      # market's vig-free probability
    edge:             float      # model_prob - fair_prob
    american_odds:    float
    decimal_odds:     float
    stake:            float      # fraction of bankroll
    outcome:          int        # 1 = won, 0 = lost
    profit:           float      # net P&L (stake-normalised)
    odds_source:      str        # "real" or "synthetic"


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def run_backtest(
    min_edge: float = MIN_EDGE,
    kelly_fraction: float = KELLY_FRACTION,
    model_type: str = "logistic",
    bankroll: float = 1.0,        # normalised; all stakes are fractions
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full walk-forward backtest and return a DataFrame of all bet records.
    """
    engine = get_engine()

    # Load real closing odds if available
    real_odds = _load_closing_odds(engine)
    n_real = len(real_odds)
    if verbose:
        print(f"Real closing odds in DB: {n_real:,} games")
        if n_real == 0:
            print("  → Using synthetic log5 market for all games.")
            print("  → Run ingestion/historical_odds.py after adding your API key")
            print("    to get real closing lines and a trustworthy backtest.")
        else:
            print(f"  → Will use real odds for {n_real:,} games, synthetic for the rest.")

    # Build the synthetic market as fallback
    if verbose:
        print("Building synthetic market fallback (log5 from prior-season W%)...")
    market = SyntheticMarket()
    market.build(engine)

    # Load all features + metadata
    X_all, y_all, meta_all = load_feature_matrix()

    all_records: list[BetRecord] = []

    # Walk-forward splits: train on [s0..sY-1], test on sY
    splits = []
    for i in range(2, len(SEASONS_IN_ORDER)):
        splits.append((SEASONS_IN_ORDER[:i], SEASONS_IN_ORDER[i]))

    for train_seasons, test_season in splits:
        prior_season = train_seasons[-1]   # most recent training season = market's prior season

        if verbose:
            print(f"\nFold: train {train_seasons} → test {test_season}")

        # Train model on training seasons
        train_mask = meta_all["season"].isin(train_seasons)
        test_mask  = meta_all["season"] == test_season

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test  = X_all[test_mask]
        y_test  = y_all[test_mask]
        meta_test = meta_all[test_mask].reset_index(drop=True)

        if model_type == "logistic":
            model = make_logistic()
        else:
            model = make_xgb()

        model.fit(X_train, y_train)
        home_probs = model.predict_proba(X_test)[:, 1]

        bets_this_fold = 0
        for i, (prob, outcome) in enumerate(zip(home_probs, y_test.values)):
            row = meta_test.iloc[i]
            home_team = row["home_team"]
            away_team = row["away_team"]
            game_pk_i = int(row["game_pk"])

            # Use real closing odds if available, else fall back to synthetic
            if game_pk_i in real_odds:
                ro = real_odds[game_pk_i]
                home_fair     = ro["home_fair"]
                away_fair     = ro["away_fair"]
                home_american = ro["home_american"]
                away_american = ro["away_american"]
                market_home_prob = home_fair   # use fair prob as "market" prob
                odds_source = "real"
            else:
                pricing = market.price(home_team, away_team, prior_season)
                home_fair     = pricing["home_fair"]
                away_fair     = pricing["away_fair"]
                home_american = pricing["home_american"]
                away_american = pricing["away_american"]
                market_home_prob = pricing["market_home_prob"]
                odds_source = "synthetic"

            away_prob = 1.0 - prob

            home_edge = compute_edge(prob, home_fair)
            away_edge = compute_edge(away_prob, away_fair)

            # Pick the side with edge above threshold
            candidates = []
            if home_edge >= min_edge:
                dec = american_to_decimal(home_american)
                candidates.append(("home", prob, home_fair,
                                   market_home_prob,
                                   home_edge, home_american, dec))
            if away_edge >= min_edge:
                dec = american_to_decimal(away_american)
                candidates.append(("away", away_prob, away_fair,
                                   1.0 - market_home_prob,
                                   away_edge, away_american, dec))

            if not candidates:
                continue

            side, model_p, fair_p, market_p, edge, american, decimal = max(
                candidates, key=lambda c: c[4]
            )

            stake_frac = kelly_stake(model_p, decimal, bankroll=1.0,
                                     fraction=kelly_fraction)
            if stake_frac <= 0:
                continue

            won = (outcome == 1 and side == "home") or (outcome == 0 and side == "away")
            profit = stake_frac * (decimal - 1) if won else -stake_frac

            all_records.append(BetRecord(
                game_pk=int(row["game_pk"]),
                game_date=row["game_date"],
                season=test_season,
                home_team=home_team,
                away_team=away_team,
                bet_side=side,
                model_prob=model_p,
                market_prob=market_p,
                fair_prob=fair_p,
                edge=edge,
                american_odds=american,
                decimal_odds=decimal,
                stake=stake_frac,
                outcome=int(won),
                profit=profit,
                odds_source=odds_source,
            ))
            bets_this_fold += 1

        if verbose:
            season_bets = [r for r in all_records if r.season == test_season]
            if season_bets:
                total_staked = sum(r.stake for r in season_bets)
                total_profit = sum(r.profit for r in season_bets)
                wins = sum(r.outcome for r in season_bets)
                roi = total_profit / total_staked if total_staked else 0
                print(f"  {bets_this_fold} bets  "
                      f"({wins}W/{bets_this_fold-wins}L  "
                      f"win%={wins/bets_this_fold:.1%})  "
                      f"ROI={roi:+.1%}")
            else:
                print(f"  No bets placed at edge threshold {min_edge:.0%}")

    return pd.DataFrame([vars(r) for r in all_records])

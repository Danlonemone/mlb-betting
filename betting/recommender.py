"""
Recommendation engine: model probabilities + market odds → bet list.

The core function `recommend()` is pure — it takes numbers in and returns
a list of recommended bets. All bookmaker communication and model loading
happens outside this module.

Decision logic:
  1. Load the trained model and build features for the target games.
  2. Get model P(home win) for each game.
  3. Fetch market odds (American moneyline for both sides).
  4. Strip the vig to get the book's fair probability.
  5. Compute edge = model_prob - fair_prob.
  6. Recommend a bet only when edge >= MIN_EDGE.
  7. Size the bet with fractional Kelly, capped at MAX_BET_PCT.
  8. Return a list of bet dicts, one per recommended bet.

A game can produce at most ONE recommended bet (home or away), the side
with the positive edge. If both sides somehow show edge (can happen at
the boundary of MIN_EDGE due to rounding), we take the higher-edge side.
"""

from __future__ import annotations

import pickle
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MIN_EDGE, KELLY_FRACTION, MIN_MARKET_PROB, MAX_AMERICAN_ODDS
from betting.odds import (
    american_to_decimal,
    remove_vig,
    compute_edge,
    expected_value,
    format_american,
    vig_pct,
)
from betting.kelly import kelly_stake, kelly_fraction_from_edge

MODEL_DIR = Path(__file__).parent.parent / "model"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BetRecommendation:
    game_pk:        int
    game_date:      str
    home_team:      str
    away_team:      str
    bet_side:       str    # "home" or "away"
    model_prob:     float  # our estimated win probability for the bet side
    fair_prob:      float  # book's vig-free implied probability for the bet side
    edge:           float  # model_prob - fair_prob
    american_odds:  float  # the odds available for the bet side
    decimal_odds:   float
    ev_per_unit:    float  # expected value per $1 staked
    full_kelly_pct: float  # what unconstrained Kelly would say (diagnostic)
    stake:          float  # recommended dollar stake (fractional Kelly, capped)
    overround:      float  # book's total margin (e.g. 1.046)
    vig_pct:        float  # vig as a percentage (e.g. 4.4)

    def to_dict(self) -> dict:
        return asdict(self)

    def display(self) -> str:
        side_str = f"{self.home_team}" if self.bet_side == "home" else f"{self.away_team}"
        opp_str  = f"{self.away_team}" if self.bet_side == "home" else f"{self.home_team}"
        return (
            f"BET {side_str} (vs {opp_str})  {self.game_date}\n"
            f"  Odds: {format_american(self.american_odds)}  "
            f"Model: {self.model_prob:.1%}  Fair: {self.fair_prob:.1%}  "
            f"Edge: {self.edge:+.1%}  EV: {self.ev_per_unit:+.3f}/unit\n"
            f"  Stake: ${self.stake:.2f}  "
            f"(Full Kelly: {self.full_kelly_pct:.1%}  Vig: {self.vig_pct:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_type: str = "logistic"):
    path = MODEL_DIR / f"model_{model_type}.pkl"
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_cols"]


# ---------------------------------------------------------------------------
# Core recommendation function
# ---------------------------------------------------------------------------

def recommend(
    games: list[dict],
    bankroll: float,
    model_type: str = "logistic",
    min_edge: float = MIN_EDGE,
    kelly_fraction: float = KELLY_FRACTION,
) -> list[BetRecommendation]:
    """
    Given a list of game dicts (each must have feature columns plus
    home_american_odds, away_american_odds, game_pk, game_date,
    home_team, away_team), return a list of BetRecommendations.

    Parameters
    ----------
    games           : list of game feature dicts (pre-built by feature pipeline)
    bankroll        : current bankroll in dollars
    model_type      : "logistic" or "xgboost"
    min_edge        : minimum edge threshold to recommend a bet
    kelly_fraction  : Kelly multiplier (default: 0.125 = eighth-Kelly)

    Returns
    -------
    List of BetRecommendation objects, sorted by edge descending.
    """
    if not games:
        return []

    model, feature_cols = load_model(model_type)

    # Build feature matrix
    X = pd.DataFrame([{col: g.get(col, 0.0) for col in feature_cols} for g in games])
    home_probs = model.predict_proba(X)[:, 1]   # P(home wins)

    recommendations = []

    for i, g in enumerate(games):
        home_prob = float(home_probs[i])
        away_prob = 1.0 - home_prob

        home_american = g.get("home_american_odds")
        away_american = g.get("away_american_odds")

        if home_american is None or away_american is None:
            continue

        home_fair, away_fair, overround = remove_vig(home_american, away_american)
        vig = vig_pct(overround)

        home_edge = compute_edge(home_prob, home_fair)
        away_edge = compute_edge(away_prob, away_fair)

        # Best available odds per side (may differ from consensus pair)
        best_home_am = g.get("best_home_american") or home_american
        best_away_am = g.get("best_away_american") or away_american

        # Pick the side with edge (if any) above threshold. Apply market
        # sanity filters before ranking so a filtered longshot does not hide
        # a valid recommendation on the other side.
        # MAX_AMERICAN_ODDS is checked against BOTH consensus and best odds —
        # the bet is ultimately placed at best odds, so the cap must apply there.
        candidates = []
        if (
            home_edge >= min_edge
            and home_fair >= MIN_MARKET_PROB
            and home_american <= MAX_AMERICAN_ODDS
            and best_home_am  <= MAX_AMERICAN_ODDS
        ):
            dec = american_to_decimal(home_american)
            candidates.append(("home", home_prob, home_fair, home_edge, home_american, dec))
        if (
            away_edge >= min_edge
            and away_fair >= MIN_MARKET_PROB
            and away_american <= MAX_AMERICAN_ODDS
            and best_away_am  <= MAX_AMERICAN_ODDS
        ):
            dec = american_to_decimal(away_american)
            candidates.append(("away", away_prob, away_fair, away_edge, away_american, dec))

        if not candidates:
            continue

        # Take the higher-edge side if both qualify
        side, prob, fair, edge, american, decimal = max(candidates, key=lambda c: c[3])

        stake = kelly_stake(prob, decimal, bankroll, fraction=kelly_fraction)
        if stake <= 0:
            continue

        # Exact full-Kelly fraction (same formula as kelly_stake uses internally)
        b = decimal - 1.0
        full_kelly = max(0.0, (b * prob - (1.0 - prob)) / b) if b > 0 else 0.0

        recommendations.append(BetRecommendation(
            game_pk=g.get("game_pk", 0),
            game_date=g.get("game_date", ""),
            home_team=g.get("home_team", ""),
            away_team=g.get("away_team", ""),
            bet_side=side,
            model_prob=prob,
            fair_prob=fair,
            edge=edge,
            american_odds=american,
            decimal_odds=decimal,
            ev_per_unit=expected_value(prob, decimal),
            full_kelly_pct=full_kelly,
            stake=stake,
            overround=overround,
            vig_pct=vig,
        ))

    return sorted(recommendations, key=lambda r: r.edge, reverse=True)


# ---------------------------------------------------------------------------
# Convenience: build game dicts from DB + external odds
# ---------------------------------------------------------------------------

def build_game_dict(
    feature_row: dict,
    home_american_odds: float,
    away_american_odds: float,
) -> dict:
    """
    Merge a feature-engineered row with odds to create a game dict
    ready for recommend().
    """
    return {
        **feature_row,
        "home_american_odds": home_american_odds,
        "away_american_odds": away_american_odds,
    }

"""
Kelly criterion stake sizing.

The Kelly criterion maximises the expected logarithm of wealth, which
is equivalent to maximising long-run compounded growth rate.

Full Kelly is theoretically optimal but practically dangerous because:
  1. Model probabilities are estimates with error — if our edge is
     overstated, Kelly over-bets and produces huge drawdowns.
  2. It produces large variance in the short run (100s of bets).

We default to eighth-Kelly (KELLY_FRACTION = 0.125), which cuts variance
by ~8× at the cost of ~8× slower growth. This is a conservative fraction
for sports bettors who are uncertain about their edge estimate.

Additional hard caps prevent any single bet from exceeding MAX_BET_PCT
of the bankroll, regardless of what Kelly says.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import KELLY_FRACTION

# Hard cap: never bet more than this fraction of bankroll on one game.
# Reduced from 5% to 3% while paper sample is small (<50 bets).
MAX_BET_PCT = 0.03


def kelly_stake(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    fraction: float = KELLY_FRACTION,
    max_pct: float = MAX_BET_PCT,
) -> float:
    """
    Return the recommended stake in dollars.

    Parameters
    ----------
    model_prob    : our estimated win probability (0–1)
    decimal_odds  : the odds we can get at the book (e.g. 2.10)
    bankroll      : current total bankroll in dollars
    fraction      : Kelly fraction (default: quarter-Kelly = 0.25)
    max_pct       : hard cap as fraction of bankroll (default: 5%)

    Returns
    -------
    stake in dollars, rounded to the nearest dollar. Returns 0 if
    Kelly formula produces a non-positive value (no edge).

    Kelly formula:
        f* = (b*p - q) / b
           = p - q/b
    where
        b = decimal_odds - 1  (net profit per unit if we win)
        p = model_prob
        q = 1 - p
    """
    b = decimal_odds - 1.0
    p = model_prob
    q = 1.0 - p

    if b <= 0:
        return 0.0

    full_kelly = (b * p - q) / b   # equivalent to p - q/b

    if full_kelly <= 0:
        return 0.0

    fractional_kelly = full_kelly * fraction
    capped = min(fractional_kelly, max_pct)

    stake = capped * bankroll
    return round(stake, 2)


def kelly_fraction_from_edge(edge: float, decimal_odds: float) -> float:
    """
    Return the full-Kelly fraction for a given edge and odds.
    Useful for diagnostics — shows what the unconstrained Kelly says.

    edge = model_prob - fair_prob
    We back out model_prob from edge and fair_prob, but here we use
    the simpler form:

        f* = edge / (decimal_odds - 1)

    This is an approximation valid when edge is small relative to prob,
    which is typical in sports betting (edges of 2–6%).
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, edge / b)


def recommended_bets_summary(bets: list[dict], bankroll: float) -> dict:
    """
    Given a list of bet dicts (output of recommender.recommend()),
    return a summary of total exposure for the slate.
    """
    total_stake = sum(b["stake"] for b in bets)
    return {
        "n_bets":        len(bets),
        "total_stake":   total_stake,
        "pct_bankroll":  total_stake / bankroll if bankroll else 0,
        "mean_edge":     sum(b["edge"] for b in bets) / len(bets) if bets else 0,
        "mean_odds":     sum(b["decimal_odds"] for b in bets) / len(bets) if bets else 0,
    }

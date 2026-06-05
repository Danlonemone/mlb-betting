"""
Odds conversion and vig-removal utilities.

All functions are pure (no I/O) so they are easy to unit-test and
compose in the recommendation pipeline.

Terminology used throughout:
  american_odds  : e.g. -150 or +130 (US moneyline format)
  decimal_odds   : e.g. 1.667 or 2.30 (European format, includes stake)
  implied_prob   : raw implied probability including the bookmaker's margin
  fair_prob      : vig-adjusted (no-margin) implied probability
  model_prob     : our model's estimated win probability
  edge           : model_prob - fair_prob (the only reason to bet)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Odds format conversions
# ---------------------------------------------------------------------------

def american_to_decimal(american: float) -> float:
    """
    Convert American odds to decimal odds.
      -150 → 1.667   (risk 150 to win 100)
      +130 → 2.300   (risk 100 to win 130)
    """
    if american >= 0:
        return (american / 100.0) + 1.0
    else:
        return (100.0 / abs(american)) + 1.0


def decimal_to_american(decimal: float) -> float:
    """
    Convert decimal odds to American odds.
      1.667 → -150   2.30 → +130
    """
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    else:
        return -100.0 / (decimal - 1.0)


def decimal_to_implied_prob(decimal: float) -> float:
    """1 / decimal_odds. Includes the bookmaker's overround."""
    if decimal <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {decimal}")
    return 1.0 / decimal


def american_to_implied_prob(american: float) -> float:
    """Direct conversion without going through decimal."""
    if american >= 0:
        return 100.0 / (american + 100.0)
    else:
        return abs(american) / (abs(american) + 100.0)


# ---------------------------------------------------------------------------
# Vig removal
# ---------------------------------------------------------------------------

def remove_vig(
    home_american: float,
    away_american: float,
) -> tuple[float, float, float]:
    """
    Given both sides of a two-way market, return:
      (home_fair_prob, away_fair_prob, overround)

    The overround is the book's total margin:
      e.g. 1.046 means 4.6% vig (they take ~2.3% per side).

    Method: divide each side's implied prob by the sum of both.
    This is the standard "multiplicative" vig-removal method and is the
    most common approach for two-outcome markets like MLB moneylines.
    """
    home_imp = american_to_implied_prob(home_american)
    away_imp = american_to_implied_prob(away_american)
    overround = home_imp + away_imp

    home_fair = home_imp / overround
    away_fair = away_imp / overround

    return home_fair, away_fair, overround


def vig_pct(overround: float) -> float:
    """Convert overround to vig percentage. overround=1.046 → 4.4%"""
    return (1.0 - 1.0 / overround) * 100.0


# ---------------------------------------------------------------------------
# Edge calculation
# ---------------------------------------------------------------------------

def compute_edge(model_prob: float, fair_prob: float) -> float:
    """
    Edge = model_prob - fair_prob.
    Positive edge → we estimate this outcome is more likely than the market does.
    This is the only theoretically sound reason to place a bet.
    """
    return model_prob - fair_prob


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """
    EV per unit staked.
    EV = model_prob * (decimal_odds - 1) - (1 - model_prob)
       = model_prob * decimal_odds - 1
    """
    return model_prob * decimal_odds - 1.0


# ---------------------------------------------------------------------------
# American odds formatter
# ---------------------------------------------------------------------------

def format_american(american: float) -> str:
    """Format American odds with sign: -150, +130."""
    return f"+{int(round(american))}" if american >= 0 else f"{int(round(american))}"

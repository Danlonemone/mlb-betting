"""
Synthetic market odds generator for backtesting.

IMPORTANT LIMITATION: We do not have historical bookmaker odds. The free
tier of The Odds API gives only current prices; historical closing lines
require a paid plan. This module builds a stand-in "market" so Phase 3
backtesting is not meaningless. Real CLV measurement begins in Phase 4.

The synthetic market is built from team win rates using the log5 method
(Bill James, 1981). Log5 only uses prior-season W/L records — it does NOT
use FIP, wOBA, or any of the features our model uses. This makes it a
genuinely independent benchmark: if our model beats log5, we have learned
something beyond what a simple records-based market already knows.

Log5 formula:
    P(A beats B) = (A - A*B) / (A + B - 2*A*B)
where A = home team win rate (prior season), B = away team win rate.
Home field advantage is implicitly captured because we compute win rates
from home + away games combined; the home team naturally wins ~53% overall.

Vig is applied symmetrically to both sides to simulate a typical book.
"""

from __future__ import annotations

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.schema import get_engine
from betting.odds import american_to_decimal

DEFAULT_VIG_HALF = 0.023   # ~4.5% total overround, typical MLB moneyline


# ---------------------------------------------------------------------------
# Team win-rate table from game results
# ---------------------------------------------------------------------------

def compute_team_win_rates(engine=None) -> pd.DataFrame:
    """
    Compute each team's win rate per season from the games table.
    Returns a DataFrame with columns: team, season, win_rate, games_played.
    """
    if engine is None:
        engine = get_engine()

    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT home_team, away_team, season, home_win FROM games "
                 "WHERE home_win IS NOT NULL"),
            conn,
        )

    rows = []
    for season, grp in df.groupby("season"):
        # Each game contributes a win/loss to both teams
        home_records = grp.groupby("home_team")["home_win"].agg(["sum", "count"])
        away_records = grp.groupby("away_team")["home_win"].agg(
            lambda x: ((1 - x).sum(), x.count())
        )

        # Compute per-team wins and games from home perspective
        home_wins   = grp.groupby("home_team")["home_win"].sum()
        home_games  = grp.groupby("home_team")["home_win"].count()
        away_wins   = grp.groupby("away_team")["home_win"].apply(lambda x: (1 - x).sum())
        away_games  = grp.groupby("away_team")["home_win"].count()

        all_teams = set(home_wins.index) | set(away_wins.index)
        for team in all_teams:
            wins  = home_wins.get(team, 0) + away_wins.get(team, 0)
            games = home_games.get(team, 0) + away_games.get(team, 0)
            rows.append({
                "team":         team,
                "season":       season,
                "wins":         wins,
                "games_played": games,
                "win_rate":     wins / games if games > 0 else 0.5,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Log5 probability
# ---------------------------------------------------------------------------

def log5(team_a_winrate: float, team_b_winrate: float) -> float:
    """
    P(A beats B) using the log5 method.
    Returns P(A wins); clipped to [0.15, 0.85] to prevent extreme odds.
    """
    a, b = team_a_winrate, team_b_winrate
    denom = a + b - 2 * a * b
    if denom == 0:
        return 0.5
    prob = (a - a * b) / denom
    return float(np.clip(prob, 0.15, 0.85))


# ---------------------------------------------------------------------------
# Synthetic odds builder
# ---------------------------------------------------------------------------

def prob_to_american_with_vig(prob: float, vig_half: float = DEFAULT_VIG_HALF) -> float:
    """Convert a probability to American odds after embedding half the vig."""
    p = min(prob + vig_half, 0.98)
    if p >= 0.5:
        return -(p / (1 - p)) * 100
    else:
        return ((1 - p) / p) * 100


class SyntheticMarket:
    """
    Prices games using log5 win rates from the prior season.
    Call build(prior_season) before pricing games in that season.
    """

    def __init__(self, vig_half: float = DEFAULT_VIG_HALF):
        self.vig_half = vig_half
        self._win_rates: dict[tuple[str, int], float] = {}
        self._loaded_seasons: set[int] = set()
        self._all_rates: pd.DataFrame | None = None

    def build(self, engine=None):
        """Load all team win rates from the DB (once)."""
        self._all_rates = compute_team_win_rates(engine)
        for _, row in self._all_rates.iterrows():
            self._win_rates[(row["team"], int(row["season"]))] = row["win_rate"]

    def _get_rate(self, team: str, season: int) -> float:
        return self._win_rates.get((team, season), 0.5)

    def price(
        self,
        home_team: str,
        away_team: str,
        prior_season: int,
    ) -> dict:
        """
        Return synthetic odds for one game.
        Uses prior_season's win rates (look-ahead-bias-free).

        Returns dict with:
          home_american, away_american, home_fair, away_fair,
          overround, market_home_prob (raw log5 prob, pre-vig)
        """
        home_rate = self._get_rate(home_team, prior_season)
        away_rate = self._get_rate(away_team, prior_season)

        # log5 gives P(home wins) accounting for each team's overall record
        home_prob = log5(home_rate, away_rate)
        away_prob = 1.0 - home_prob

        home_american = prob_to_american_with_vig(home_prob, self.vig_half)
        away_american = prob_to_american_with_vig(away_prob, self.vig_half)

        # Vig-free fair probs (what the market "really" thinks)
        home_imp = 1.0 / american_to_decimal(home_american)
        away_imp = 1.0 / american_to_decimal(away_american)
        overround = home_imp + away_imp
        home_fair = home_imp / overround
        away_fair = away_imp / overround

        return {
            "home_american":    home_american,
            "away_american":    away_american,
            "home_fair":        home_fair,
            "away_fair":        away_fair,
            "overround":        overround,
            "market_home_prob": home_prob,   # raw log5 (no vig)
        }

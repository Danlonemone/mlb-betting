"""
Core unit tests: odds math, Kelly sizing, over/under push handling,
and train/live feature parity.

Run from the project root:
    python -m pytest tests/ -v
"""

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from betting.odds import (
    american_to_decimal, decimal_to_american,
    american_to_implied_prob, remove_vig, compute_edge, expected_value,
)
from betting.kelly import kelly_stake, MAX_BET_PCT


# ---------------------------------------------------------------------------
# Odds conversions
# ---------------------------------------------------------------------------

class TestOdds:
    def test_american_to_decimal_known_values(self):
        assert american_to_decimal(-150) == pytest.approx(1.6667, abs=1e-3)
        assert american_to_decimal(+130) == pytest.approx(2.30, abs=1e-9)
        assert american_to_decimal(+100) == pytest.approx(2.0)
        assert american_to_decimal(-100) == pytest.approx(2.0)

    def test_round_trip(self):
        for am in [-300, -150, -110, 105, 130, 250, 400]:
            dec = american_to_decimal(am)
            assert decimal_to_american(dec) == pytest.approx(am, abs=1e-6)

    def test_implied_prob(self):
        assert american_to_implied_prob(-150) == pytest.approx(0.6)
        assert american_to_implied_prob(+150) == pytest.approx(0.4)

    def test_remove_vig_sums_to_one(self):
        home_fair, away_fair, overround = remove_vig(-150, +130)
        assert home_fair + away_fair == pytest.approx(1.0)
        assert overround > 1.0  # the book always has margin

    def test_remove_vig_symmetric_market(self):
        home_fair, away_fair, _ = remove_vig(-110, -110)
        assert home_fair == pytest.approx(0.5)
        assert away_fair == pytest.approx(0.5)

    def test_edge_and_ev(self):
        assert compute_edge(0.55, 0.50) == pytest.approx(0.05)
        # EV at fair odds with true prob = implied prob is zero
        assert expected_value(0.5, 2.0) == pytest.approx(0.0)
        assert expected_value(0.55, 2.0) == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------

class TestKelly:
    def test_no_edge_returns_zero(self):
        # model prob equal to break-even prob → no bet
        assert kelly_stake(0.5, 2.0, bankroll=100) == 0.0
        assert kelly_stake(0.4, 2.0, bankroll=100) == 0.0

    def test_positive_edge_positive_stake(self):
        assert kelly_stake(0.6, 2.0, bankroll=100, fraction=1.0) > 0

    def test_max_bet_cap(self):
        # Huge edge: full Kelly would say bet a large fraction; cap applies
        stake = kelly_stake(0.9, 3.0, bankroll=100, fraction=1.0)
        assert stake == pytest.approx(MAX_BET_PCT * 100)

    def test_fraction_scales_linearly_below_cap(self):
        full = kelly_stake(0.55, 2.0, bankroll=1000, fraction=1.0, max_pct=1.0)
        eighth = kelly_stake(0.55, 2.0, bankroll=1000, fraction=0.125, max_pct=1.0)
        assert eighth == pytest.approx(full * 0.125, abs=0.01)

    def test_bad_odds_return_zero(self):
        assert kelly_stake(0.9, 1.0, bankroll=100) == 0.0


# ---------------------------------------------------------------------------
# Over/under probability conversion (push handling)
# ---------------------------------------------------------------------------

class TestOverUnderProbs:
    def _probs(self, expected, line):
        from props.strikeout_model import over_under_probs
        return over_under_probs(expected, line)

    def test_half_line_sums_to_one(self):
        over, under = self._probs(6.0, 5.5)
        assert over + under == pytest.approx(1.0)

    def test_half_line_matches_plain_tail(self):
        from scipy.stats import poisson
        over, _ = self._probs(6.0, 5.5)
        assert over == pytest.approx(1 - poisson.cdf(5, 6.0))

    def test_integer_line_excludes_push(self):
        from scipy.stats import poisson
        mu, line = 6.0, 6.0
        over, under = self._probs(mu, line)
        # Conditional on no push: P(X>6) and P(X<6) renormalised
        p_over, p_under = 1 - poisson.cdf(6, mu), poisson.cdf(5, mu)
        assert over == pytest.approx(p_over / (p_over + p_under))
        assert under == pytest.approx(p_under / (p_over + p_under))
        assert over + under == pytest.approx(1.0)

    def test_integer_line_under_not_inflated(self):
        """The old bug: prob_under included the push probability."""
        from scipy.stats import poisson
        mu, line = 6.0, 6.0
        _, under = self._probs(mu, line)
        old_buggy_under = poisson.cdf(6, mu)  # included P(X == 6)
        assert under < old_buggy_under

    def test_monotone_in_expectation(self):
        over_low, _ = self._probs(4.0, 5.5)
        over_high, _ = self._probs(8.0, 5.5)
        assert over_high > over_low


# ---------------------------------------------------------------------------
# Settlement push/void logic
# ---------------------------------------------------------------------------

class TestSettlementLogic:
    def test_integer_line_exact_hit_is_push(self):
        # mirrors the settle.py logic: actual == line → push
        actual, line = 6, 6.0
        assert actual == line
        went_over = actual > line
        assert not went_over  # would have been scored as an under win pre-fix

    def test_days_old(self):
        from paper_trade.settle import _days_old
        from datetime import date, timedelta
        today = date.today().isoformat()
        old = (date.today() - timedelta(days=3)).isoformat()
        assert _days_old(today) == 0
        assert _days_old(old) == 3

    def test_void_bet_sets_push(self):
        from paper_trade.settle import _void_bet

        class FakeBet:
            outcome = None
            profit_dollars = None
            settled_at = None

        bet = FakeBet()
        _void_bet(bet, "test void")
        assert bet.outcome == -1
        assert bet.profit_dollars == 0.0
        assert bet.settled_at is not None


# ---------------------------------------------------------------------------
# Train/live feature parity
# ---------------------------------------------------------------------------

def _minimal_games_df(**overrides) -> pd.DataFrame:
    """One-row games DataFrame with every column build_features touches."""
    row = {
        "game_pk": 1, "game_date": "2025-06-01", "season": 2025,
        "home_team": "NYY", "away_team": "BOS",
        "home_sp_mlbam_id": 100, "away_sp_mlbam_id": 200,
        "home_sp_fip": 3.5, "away_sp_fip": 4.0,
        "home_sp_k_pct": 0.25, "away_sp_k_pct": 0.20,
        "home_sp_bb_pct": 0.07, "away_sp_bb_pct": 0.08,
        "home_sp_era": 3.2, "away_sp_era": 4.1,
        "home_woba": 0.330, "away_woba": 0.310,
        "home_team_era": 3.9, "away_team_era": 4.2,
        "home_team_fip": 3.8, "away_team_fip": 4.1,
        "park_factor": 102.0,
        "home_rest_days": 1, "away_rest_days": 1,
        "home_score": 5, "away_score": 3, "home_win": 1,
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestFeatureParity:
    def test_training_one_sided_missing_sp_is_zero(self):
        from features.engineering import build_features
        df = _minimal_games_df(home_sp_fip=np.nan, home_sp_era=np.nan,
                               home_sp_k_pct=np.nan, home_sp_bb_pct=np.nan)
        X, _ = build_features(df)
        assert X.loc[0, "sp_fip_diff"] == 0.0
        assert X.loc[0, "sp_era_diff"] == 0.0
        assert X.loc[0, "sp_data_available"] == 0.0

    def test_live_paired_diff_matches_training_convention(self):
        from paper_trade.live_features import _paired_diff
        # one-sided missing → 0, exactly like training fillna(0) on the diff
        assert _paired_diff(None, 3.8) == 0.0
        assert _paired_diff(3.8, None) == 0.0
        assert _paired_diff(None, None) == 0.0
        assert _paired_diff(3.5, 4.0) == pytest.approx(-0.5)

    def test_market_feature_centered_and_flagged(self):
        from features.engineering import build_features
        df = _minimal_games_df()
        df["close_home_fair"] = 0.62
        X, _ = build_features(df)
        assert X.loc[0, "market_fair_prob"] == pytest.approx(0.12)
        assert X.loc[0, "market_data_available"] == 1.0

    def test_market_feature_missing_is_neutral(self):
        from features.engineering import build_features
        df = _minimal_games_df()  # no close_home_fair column
        X, _ = build_features(df)
        assert X.loc[0, "market_fair_prob"] == 0.0
        assert X.loc[0, "market_data_available"] == 0.0

    def test_feature_cols_complete(self):
        from features.engineering import build_features, FEATURE_COLS
        X, _ = build_features(_minimal_games_df())
        assert list(X.columns) == FEATURE_COLS
        assert not X.isna().any().any()

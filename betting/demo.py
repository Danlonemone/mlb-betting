"""
Phase 2 demo: simulate a slate of games with realistic bookmaker odds
and run the full pipeline (model → edge → Kelly stake → recommendations).

Since we don't have historical odds yet (that comes in Phase 4), we:
  1. Load real 2024 games from the DB (real features, real outcomes).
  2. Synthesize plausible moneyline odds by converting the model's probability
     to American odds and adding a realistic vig (4–5%), then perturbing
     slightly to simulate market variation.
  3. Run the recommender and show the output.
  4. Simulate what the P&L would look like if we had bet those games.

This is NOT a valid backtest — the odds are synthetic, not real market prices.
Its purpose is to verify the betting math is correct and that the pipeline
produces sensible output before we wire in real odds in Phase 4.
"""

import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.engineering import load_feature_matrix, FEATURE_COLS
from betting.odds import (
    american_to_decimal, decimal_to_american, remove_vig,
    american_to_implied_prob, format_american, vig_pct
)
from betting.kelly import kelly_stake, recommended_bets_summary
from betting.recommender import recommend, load_model, build_game_dict
from db.schema import get_engine
from sqlalchemy import text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prob_to_american(prob: float, vig_half: float = 0.023) -> float:
    """
    Convert a win probability to American odds with a vig applied to each side.
    vig_half ~= 2.3% per side → ~4.5% total overround, typical for MLB.
    """
    raw_prob = prob + vig_half   # inflate probability to embed vig
    raw_prob = min(raw_prob, 0.98)
    if raw_prob >= 0.5:
        return -(raw_prob / (1 - raw_prob)) * 100
    else:
        return ((1 - raw_prob) / raw_prob) * 100


def synthesize_odds(home_prob: float, noise_std: float = 0.01) -> tuple[float, float]:
    """
    Create synthetic market odds from the true probability.
    Adding noise simulates line shopping / book variation.
    """
    rng = np.random.default_rng(seed=int(home_prob * 1e6))
    noise = rng.normal(0, noise_std)
    market_home_prob = np.clip(home_prob + noise, 0.15, 0.85)
    market_away_prob = 1.0 - market_home_prob

    home_american = prob_to_american(market_home_prob)
    away_american = prob_to_american(market_away_prob)
    return round(home_american, 0), round(away_american, 0)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(bankroll: float = 1000.0, min_edge: float = 0.03, n_games: int = 50):
    print(f"\n{'='*65}")
    print("Phase 2 Demo: Model → Edge → Kelly Stake → Bet Recommendations")
    print(f"{'='*65}")
    print(f"Bankroll: ${bankroll:,.2f}  |  Min edge: {min_edge:.0%}  |  Games: {n_games}")

    # Load features for 2024 games (held-out season)
    X, y, meta = load_feature_matrix(seasons=[2024])
    model, _ = load_model("logistic")
    home_probs = model.predict_proba(X)[:, 1]

    # Sample n_games
    idx = np.random.default_rng(42).choice(len(X), size=min(n_games, len(X)), replace=False)
    sample_X    = X.iloc[idx].reset_index(drop=True)
    sample_y    = y.iloc[idx].reset_index(drop=True)
    sample_meta = meta.iloc[idx].reset_index(drop=True)
    sample_prob = home_probs[idx]

    # Build game dicts with synthetic odds
    games = []
    for i in range(len(sample_X)):
        home_prob = sample_prob[i]
        home_odds, away_odds = synthesize_odds(home_prob)

        row = {col: sample_X.iloc[i][col] for col in FEATURE_COLS}
        row.update({
            "game_pk":    int(sample_meta.iloc[i]["game_pk"]),
            "game_date":  sample_meta.iloc[i]["game_date"],
            "home_team":  sample_meta.iloc[i]["home_team"],
            "away_team":  sample_meta.iloc[i]["away_team"],
            "home_american_odds": home_odds,
            "away_american_odds": away_odds,
            "_true_outcome": int(sample_y.iloc[i]),
        })
        games.append(row)

    # Run recommender
    recs = recommend(games, bankroll=bankroll, min_edge=min_edge)

    print(f"\nRecommended bets: {len(recs)} / {n_games} games")
    print(f"Edge threshold:   {min_edge:.0%} (configurable in config.py)\n")

    if not recs:
        print("No bets recommended at this edge threshold.")
        return

    print(f"{'Date':<12} {'Matchup':<14} {'Side':<5} {'Odds':>6} {'Model':>7} "
          f"{'Fair':>7} {'Edge':>6} {'EV':>7} {'Stake':>8}")
    print("-" * 80)

    for r in recs[:20]:   # show first 20
        matchup = f"{r.away_team}@{r.home_team}"
        side_abbr = r.home_team if r.bet_side == "home" else r.away_team
        print(
            f"{r.game_date:<12} {matchup:<14} {side_abbr:<5} "
            f"{format_american(r.american_odds):>6} "
            f"{r.model_prob:>6.1%} {r.fair_prob:>6.1%} "
            f"{r.edge:>+5.1%} {r.ev_per_unit:>+6.3f} "
            f"${r.stake:>7.2f}"
        )

    if len(recs) > 20:
        print(f"  ... and {len(recs)-20} more")

    # Summary
    summary = recommended_bets_summary([r.to_dict() for r in recs], bankroll)
    print(f"\nSlate summary:")
    print(f"  Bets recommended:  {summary['n_bets']}")
    print(f"  Total at risk:     ${summary['total_stake']:.2f} "
          f"({summary['pct_bankroll']:.1%} of bankroll)")
    print(f"  Mean edge:         {summary['mean_edge']:+.2%}")
    print(f"  Mean decimal odds: {summary['mean_odds']:.3f}")

    # P&L simulation (using real 2024 game outcomes, synthetic odds)
    print(f"\n{'='*65}")
    print("Simulated P&L (real outcomes, synthetic odds — for verification only)")
    print(f"{'='*65}")

    # Find the true outcomes for each recommendation
    game_outcome = {g["game_pk"]: g["_true_outcome"] for g in games}

    pnl = 0.0
    wins = 0
    losses = 0

    for r in recs:
        outcome = game_outcome.get(r.game_pk)
        if outcome is None:
            continue
        won = (outcome == 1 and r.bet_side == "home") or \
              (outcome == 0 and r.bet_side == "away")
        if won:
            profit = r.stake * (r.decimal_odds - 1)
            pnl += profit
            wins += 1
        else:
            pnl -= r.stake
            losses += 1

    total_staked = sum(r.stake for r in recs)
    roi = pnl / total_staked if total_staked else 0

    print(f"  Bets:    {wins+losses}  ({wins}W / {losses}L)")
    print(f"  Staked:  ${total_staked:.2f}")
    print(f"  P&L:     ${pnl:+.2f}")
    print(f"  ROI:     {roi:+.1%}")
    print(f"\nNote: odds are synthetic, not real market prices.")
    print(f"Real-odds backtest comes in Phase 3. Paper trading in Phase 4.")

    # Math verification
    print(f"\n{'='*65}")
    print("Odds math verification (spot check):")
    r = recs[0]
    print(f"  Bet:           {r.home_team} vs {r.away_team} ({r.bet_side} side)")
    print(f"  American odds: {format_american(r.american_odds)}")
    print(f"  Decimal odds:  {r.decimal_odds:.4f}")
    print(f"  Implied prob:  {1/r.decimal_odds:.4f}")
    print(f"  Fair prob:     {r.fair_prob:.4f}  (vig removed, overround={r.overround:.4f})")
    print(f"  Model prob:    {r.model_prob:.4f}")
    print(f"  Edge:          {r.edge:+.4f}  ({r.edge/r.fair_prob:+.1%} of fair prob)")
    print(f"  Full Kelly:    {r.full_kelly_pct:.4f} = {r.full_kelly_pct:.2%} of bankroll")
    print(f"  Quarter Kelly: {r.full_kelly_pct * 0.25:.4f} = "
          f"{r.full_kelly_pct * 0.25:.2%} of bankroll")
    print(f"  Stake:         ${r.stake:.2f}  (capped at 5% of ${bankroll:,.0f})")
    print(f"  EV per unit:   {r.ev_per_unit:+.4f}")


if __name__ == "__main__":
    run_demo(bankroll=1000.0, min_edge=0.03, n_games=100)

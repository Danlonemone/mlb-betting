"""
Morning runner — call this each day before games start.

What it does:
  1. Fetches today's MLB moneyline odds from The Odds API
  2. Builds model features for each game
  3. Runs the recommender
  4. Logs recommended bets to the paper_bets table
  5. Prints the day's picks

Usage:
    python paper_trade/daily_picks.py
    python paper_trade/daily_picks.py --bankroll 75.04 --min-edge 0.10

The script is idempotent: re-running it on the same day will not
duplicate bets (the UniqueConstraint on game_pk + bet_side prevents it).
"""

from __future__ import annotations

import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MIN_EDGE, KELLY_FRACTION, DEFAULT_BANKROLL
from db.schema import init_db, get_session, PaperBet, F5Bet
from paper_trade.odds_api import fetch_today_odds, fetch_today_f5_odds, OddsAPIError
from paper_trade.live_features import build_live_features, build_live_f5_features
from betting.recommender import recommend
from betting.odds import american_to_decimal, remove_vig, format_american


def run_daily_picks(
    bankroll: float = DEFAULT_BANKROLL,
    min_edge: float = MIN_EDGE,
    model_type: str = "logistic",
    dry_run: bool = False,
) -> list[dict]:
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"Daily Picks — {today}")
    print(f"Bankroll: ${bankroll:,.2f}  |  Min edge: {min_edge:.0%}  |  Model: {model_type}")
    print(f"{'='*60}")

    # 1. Fetch live odds
    print("\n[1/3] Fetching odds from The Odds API...")
    try:
        odds_games = fetch_today_odds()
    except OddsAPIError as e:
        print(f"\n  ERROR: {e}")
        return []

    if not odds_games:
        print("  No games with odds found for today. Check back later.")
        return []

    # 2. Build features
    print("\n[2/3] Building model features...")
    feature_rows = build_live_features(odds_games)

    if not feature_rows:
        print("  Could not build features for any games.")
        return []

    # 3. Generate recommendations
    print("\n[3/3] Running model...")
    recs = recommend(
        feature_rows,
        bankroll=bankroll,
        model_type=model_type,
        min_edge=min_edge,
        kelly_fraction=KELLY_FRACTION,
    )

    # Display picks
    print(f"\n{'─'*60}")
    if not recs:
        print(f"  No bets recommended today at {min_edge:.0%} edge threshold.")
    else:
        print(f"  {len(recs)} bet(s) recommended:\n")
        print(f"  {'Matchup':<16} {'Side':<5} {'Odds':>6} {'Model':>7} "
              f"{'Fair':>7} {'Edge':>6} {'Stake':>8}  Starters")
        print(f"  {'─'*72}")
        for r in recs:
            matchup = f"{r.away_team}@{r.home_team}"
            side    = r.home_team if r.bet_side == "home" else r.away_team
            meta    = next((g for g in feature_rows if g["game_pk"] == r.game_pk), {})
            conf_flag = "✓" if meta.get("lineup_confirmed") else "P"
            h_sp  = meta.get("home_sp_name", "TBD")
            a_sp  = meta.get("away_sp_name", "TBD")
            print(
                f"  {matchup:<16} {side:<5} "
                f"{format_american(r.american_odds):>6} "
                f"{r.model_prob:>6.1%} {r.fair_prob:>6.1%} "
                f"{r.edge:>+5.1%} ${r.stake:>7.2f}"
                f"  [{conf_flag}] {a_sp} vs {h_sp}"
            )
        total = sum(r.stake for r in recs)
        print(f"\n  Total at risk today: ${total:.2f} ({total/bankroll:.1%} of bankroll)")
        print(f"  [✓]=confirmed lineup  [P]=probable only")

    if dry_run:
        print("\n  [DRY RUN] Bets not logged to DB.")
        run_f5_picks(bankroll=bankroll, min_edge=min_edge, dry_run=True)
        return [r.to_dict() for r in recs]

    # 4. Log to DB
    engine  = init_db()
    session = get_session(engine)
    logged  = 0
    skipped = 0

    for r in recs:
        # Find matching feature row for metadata
        meta = next(
            (g for g in feature_rows if g["game_pk"] == r.game_pk), {}
        )
        home_open = meta.get("home_american_odds")
        away_open = meta.get("away_american_odds")
        _, _, overround = remove_vig(home_open, away_open) if home_open and away_open else (None, None, None)

        existing = (
            session.query(PaperBet)
            .filter(PaperBet.game_pk == r.game_pk, PaperBet.bet_side == r.bet_side)
            .first()
        )
        if existing:
            skipped += 1
            continue

        bet = PaperBet(
            game_pk             = r.game_pk,
            game_date           = r.game_date,
            home_team           = r.home_team,
            away_team           = r.away_team,
            bet_side            = r.bet_side,
            model_prob          = r.model_prob,
            fair_prob           = r.fair_prob,
            edge                = r.edge,
            home_american_open  = home_open,
            away_american_open  = away_open,
            bet_american_odds   = r.american_odds,
            bet_decimal_odds    = r.decimal_odds,
            overround_open      = overround,
            stake_fraction      = r.stake / bankroll,
            stake_dollars       = r.stake,
            bankroll_at_bet     = bankroll,
            bookmaker           = meta.get("bookmaker", ""),
            created_at          = datetime.now(timezone.utc).isoformat(),
        )
        session.add(bet)
        logged += 1

    session.commit()
    session.close()

    if logged:
        print(f"\n  ✓ {logged} bet(s) logged to paper_bets table.")
    if skipped:
        print(f"  {skipped} already logged (skipped).")

    # Run F5 picks as a second pass
    run_f5_picks(bankroll=bankroll, min_edge=min_edge, dry_run=dry_run)

    return [r.to_dict() for r in recs]


def run_f5_picks(
    bankroll: float = DEFAULT_BANKROLL,
    min_edge: float = MIN_EDGE,
    dry_run: bool = False,
) -> list[dict]:
    """
    Fetch F5 odds, run the F5 model, and log bets to f5_paper_bets.
    Returns list of recommendation dicts.
    """
    print("\n[F5] Fetching first-5-innings odds...")
    try:
        f5_games = fetch_today_f5_odds()
    except OddsAPIError as e:
        print(f"  F5 odds error: {e}")
        return []

    if not f5_games:
        print("  No F5 odds available today.")
        return []

    print("[F5] Building features...")
    feature_rows = build_live_f5_features(f5_games)
    if not feature_rows:
        return []

    recs = recommend(
        feature_rows,
        bankroll=bankroll,
        model_type="f5_logistic",
        min_edge=min_edge,
        kelly_fraction=KELLY_FRACTION,
    )

    print(f"\n{'─'*60}")
    print("  F5 (First 5 Innings) picks:")
    if not recs:
        print(f"  No F5 bets recommended at {min_edge:.0%} edge threshold.")
    else:
        print(f"  {len(recs)} F5 bet(s):\n")
        print(f"  {'Matchup':<16} {'Side':<5} {'Odds':>6} {'Model':>7} "
              f"{'Fair':>7} {'Edge':>6} {'Stake':>8}")
        print(f"  {'─'*60}")
        for r in recs:
            matchup = f"{r.away_team}@{r.home_team}"
            side    = r.home_team if r.bet_side == "home" else r.away_team
            print(
                f"  {matchup:<16} {side:<5} "
                f"{format_american(r.american_odds):>6} "
                f"{r.model_prob:>6.1%} {r.fair_prob:>6.1%} "
                f"{r.edge:>+5.1%} ${r.stake:>7.2f}"
            )
        total = sum(r.stake for r in recs)
        print(f"\n  F5 total at risk: ${total:.2f} ({total/bankroll:.1%} of bankroll)")

    if dry_run:
        print("  [DRY RUN] F5 bets not logged.")
        return [r.to_dict() for r in recs]

    engine  = init_db()
    session = get_session(engine)
    logged  = skipped = 0

    for r in recs:
        meta = next((g for g in feature_rows if g.get("game_pk") == r.game_pk), {})
        home_open = meta.get("home_american_odds")
        away_open = meta.get("away_american_odds")
        _, _, overround = remove_vig(home_open, away_open) if home_open and away_open else (None, None, None)

        existing = (
            session.query(F5Bet)
            .filter(F5Bet.game_pk == r.game_pk, F5Bet.bet_side == r.bet_side)
            .first()
        )
        if existing:
            skipped += 1
            continue

        bet = F5Bet(
            game_pk            = r.game_pk,
            game_date          = r.game_date,
            home_team          = r.home_team,
            away_team          = r.away_team,
            bet_side           = r.bet_side,
            model_prob         = r.model_prob,
            fair_prob          = r.fair_prob,
            edge               = r.edge,
            home_american_open = home_open,
            away_american_open = away_open,
            bet_american_odds  = r.american_odds,
            bet_decimal_odds   = r.decimal_odds,
            overround_open     = overround,
            stake_fraction     = r.stake / bankroll,
            stake_dollars      = r.stake,
            bankroll_at_bet    = bankroll,
            bookmaker          = meta.get("bookmaker", ""),
            created_at         = datetime.now(timezone.utc).isoformat(),
        )
        session.add(bet)
        logged += 1

    session.commit()
    session.close()

    if logged:
        print(f"  ✓ {logged} F5 bet(s) logged to f5_paper_bets.")
    if skipped:
        print(f"  {skipped} F5 bet(s) already logged (skipped).")

    return [r.to_dict() for r in recs]


def record_closing_odds(game_pk: int, home_close: float, away_close: float):
    """
    Call this ~30 minutes before first pitch to record the closing line.
    This enables true CLV calculation after settlement.
    """
    from betting.odds import remove_vig, compute_edge

    engine  = init_db()
    session = get_session(engine)

    for bet in session.query(PaperBet).filter(PaperBet.game_pk == game_pk).all():
        home_fair, away_fair, _ = remove_vig(home_close, away_close)
        close_fair = home_fair if bet.bet_side == "home" else away_fair
        close_american = home_close if bet.bet_side == "home" else away_close

        bet.home_american_close = home_close
        bet.away_american_close = away_close
        bet.bet_american_close  = close_american
        bet.clv = compute_edge(bet.model_prob, close_fair)

    session.commit()
    session.close()
    print(f"  Closing odds recorded for game_pk={game_pk}: "
          f"home {format_american(home_close)} / away {format_american(away_close)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB paper trading daily picks")
    parser.add_argument("--bankroll",  type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--min-edge",  type=float, default=MIN_EDGE)
    parser.add_argument("--model",     type=str,   default="logistic",
                        choices=["logistic", "xgboost"])
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print picks without logging to DB")
    args = parser.parse_args()

    run_daily_picks(
        bankroll  = args.bankroll,
        min_edge  = args.min_edge,
        model_type = args.model,
        dry_run   = args.dry_run,
    )

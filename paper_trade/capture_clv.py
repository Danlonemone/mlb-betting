"""
Capture closing-line odds for pending paper bets and compute CLV.

CLV (Closing Line Value) = fair_close_prob - our_implied_prob
  Positive → we got better odds than where the vig-free market settled (good)
  Negative → market moved against us after we bet (bad)

Best run schedule: ~4 PM PT / 7 PM ET daily, before most evening games start.
Games already in progress will not appear in the Odds API and will be skipped.

Usage:
    python paper_trade/capture_clv.py
    python paper_trade/capture_clv.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.schema import init_db, get_session, PaperBet, F5Bet
from betting.odds import american_to_implied_prob, remove_vig
from paper_trade.odds_api import fetch_mlb_odds, fetch_today_f5_odds, parse_game_odds


def _safe_error(exc: Exception) -> str:
    """Avoid printing API keys embedded in request URLs."""
    import re
    msg = str(exc)
    return re.sub(r"apiKey=[^&\s)]+", "apiKey=redacted", msg)


def _clv(bet_american: float, home_close: float, away_close: float, bet_side: str) -> float:
    """
    Vig-free CLV: fair closing probability for our side minus our bet's implied probability.
    Positive = we beat the closing line.
    """
    home_fair, away_fair, _ = remove_vig(home_close, away_close)
    fair_close = home_fair if bet_side == "home" else away_fair
    our_implied = american_to_implied_prob(bet_american)
    return round(fair_close - our_implied, 6)


def capture_closing_odds(dry_run: bool = False) -> dict:
    """
    Find pending bets without closing odds, match to live Odds API prices,
    and write home/away/bet close + CLV for both ML and F5 bets.

    Returns {"captured": int, "missed": int}.
    """
    # Log closing odds snapshot for line movement tracking (side-effect, non-fatal)
    try:
        from paper_trade.log_odds_snapshot import log_snapshot
        log_snapshot(label="close")
    except Exception as exc:
        print(f"  [Snapshot] close skipped: {_safe_error(exc)}")

    # Fetch odds once; used for both ML and F5 CLV capture
    try:
        raw        = fetch_mlb_odds()
        live_games = parse_game_odds(raw)
    except Exception as exc:
        print(f"  ✗ Odds API error: {_safe_error(exc)}")
        return {"captured": 0, "missed": 0}

    odds_index: dict[tuple, dict] = {
        (g["home_team"], g["away_team"], g["game_date"]): g
        for g in live_games
    }

    engine  = init_db()
    session = get_session(engine)

    pending = (
        session.query(PaperBet)
        .filter(
            PaperBet.outcome.is_(None),
            PaperBet.bet_american_close.is_(None),
        )
        .all()
    )

    if not pending:
        print("  No pending bets need closing odds.")
        session.close()
        f5 = _capture_f5_clv(dry_run=dry_run)
        return {"captured": f5["captured"], "missed": f5.get("missed", 0)}

    print(f"\n  {len(pending)} pending bet(s) missing closing odds...")

    captured = missed = 0

    for bet in pending:
        key  = (bet.home_team, bet.away_team, bet.game_date)
        live = odds_index.get(key)
        side = bet.home_team if bet.bet_side == "home" else bet.away_team

        if live is None:
            print(
                f"  ⏭  {bet.away_team}@{bet.home_team} ({bet.game_date}) — "
                f"not in API (game may have started)"
            )
            missed += 1
            continue

        home_close = live["home_american"]
        away_close = live["away_american"]
        bet_close  = home_close if bet.bet_side == "home" else away_close
        clv_val    = _clv(bet.bet_american_odds, home_close, away_close, bet.bet_side)

        direction = "↑ beat" if clv_val > 0 else "↓ missed"
        print(
            f"  ✓  {bet.away_team}@{bet.home_team}  |  "
            f"Bet {side} {bet.bet_american_odds:+.0f}  |  "
            f"Close {bet_close:+.0f}  |  "
            f"CLV {clv_val:+.3f} ({direction} line)"
        )

        if not dry_run:
            bet.home_american_close = home_close
            bet.away_american_close = away_close
            bet.bet_american_close  = bet_close
            bet.clv                 = clv_val

        captured += 1

    if not dry_run and captured:
        session.commit()

    session.close()

    if dry_run:
        print(f"\n  [dry-run] Would capture {captured}, miss {missed}.")
    else:
        if captured:
            print(f"\n  Saved closing odds for {captured} bet(s).")
        if missed:
            print(f"  {missed} bet(s) could not be captured (already in play).")

    f5 = _capture_f5_clv(dry_run=dry_run)
    captured += f5["captured"]
    missed += f5.get("missed", 0)

    return {"captured": captured, "missed": missed}


def _capture_f5_clv(dry_run: bool = False) -> dict:
    """Capture closing odds and CLV for pending F5 bets from F5 moneyline odds."""
    engine  = init_db()
    session = get_session(engine)

    pending = (
        session.query(F5Bet)
        .filter(F5Bet.outcome.is_(None), F5Bet.bet_american_close.is_(None))
        .all()
    )

    if not pending:
        session.close()
        return {"captured": 0, "missed": 0}

    try:
        f5_games = fetch_today_f5_odds()
    except Exception as exc:
        print(f"  ✗ F5 odds API error: {_safe_error(exc)}")
        session.close()
        return {"captured": 0, "missed": len(pending)}

    odds_index: dict[tuple, dict] = {
        (g["home_team"], g["away_team"], g["game_date"]): g
        for g in f5_games
    }

    captured = missed = 0
    for bet in pending:
        key  = (bet.home_team, bet.away_team, bet.game_date)
        live = odds_index.get(key)
        if live is None:
            missed += 1
            continue

        home_close = live["home_american_odds"]
        away_close = live["away_american_odds"]
        bet_close  = home_close if bet.bet_side == "home" else away_close
        clv_val    = _clv(bet.bet_american_odds, home_close, away_close, bet.bet_side)

        if not dry_run:
            bet.home_american_close = home_close
            bet.away_american_close = away_close
            bet.bet_american_close  = bet_close
            bet.clv                 = clv_val
        captured += 1

    if not dry_run and captured:
        session.commit()
    session.close()
    return {"captured": captured, "missed": missed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture pre-game closing odds and compute CLV for pending paper bets."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing to DB")
    args = parser.parse_args()
    capture_closing_odds(dry_run=args.dry_run)

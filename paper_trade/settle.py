"""
Result settler — runs after games finish (evening or next morning).

Finds all unsettled paper bets, fetches final scores from the MLB Stats API,
marks each bet as won or lost, and calculates profit/loss in dollars.

Usage:
    python paper_trade/settle.py
    python paper_trade/settle.py --date 2025-06-10
"""

from __future__ import annotations

import sys
import argparse
import requests
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.schema import init_db, get_session, PaperBet, F5Bet
from betting.odds import format_american
from paper_trade.capture_clv import capture_closing_odds

MLB_API_BASE = "https://statsapi.mlb.com"
HEADERS = {"User-Agent": "mlb-betting-model/0.1"}


def fetch_f5_result(game_pk: int) -> dict | None:
    """
    Fetch per-inning linescore and return F5 score.
    Returns {"home_score_f5": int, "away_score_f5": int, "home_win_f5": int|None}
    or None if the game has not yet completed at least 5 innings.
    Push (tie after 5) returns home_win_f5=None.
    """
    try:
        r = requests.get(
            f"{MLB_API_BASE}/api/v1.1/game/{game_pk}/feed/live",
            headers=HEADERS, timeout=20
        )
        r.raise_for_status()
        data = r.json()

        status = (
            data.get("gameData", {})
                .get("status", {})
                .get("detailedState", "")
        )
        innings_completed = (
            data.get("liveData", {})
                .get("linescore", {})
                .get("currentInning", 0)
        )
        is_final = status in ("Final", "Game Over", "Completed Early")
        if not is_final and innings_completed < 5:
            return None

        innings = (
            data.get("liveData", {})
                .get("linescore", {})
                .get("innings", [])
        )
        if len(innings) < 5:
            return None

        home_f5 = sum(inn.get("home", {}).get("runs", 0) or 0 for inn in innings[:5])
        away_f5 = sum(inn.get("away", {}).get("runs", 0) or 0 for inn in innings[:5])

        if home_f5 > away_f5:
            win_f5 = 1
        elif away_f5 > home_f5:
            win_f5 = 0
        else:
            win_f5 = None  # tie = push

        return {"home_score_f5": home_f5, "away_score_f5": away_f5, "home_win_f5": win_f5}

    except Exception as e:
        print(f"  ⚠ Could not fetch F5 result for game_pk={game_pk}: {e}")
        return None


def settle_f5_bets(date: str | None = None) -> dict:
    """Settle unsettled F5 bets. Ties are recorded as pushes (profit = 0)."""
    engine  = init_db()
    session = get_session(engine)

    query = session.query(F5Bet).filter(F5Bet.outcome.is_(None))
    if date:
        query = query.filter(F5Bet.game_date == date)
    unsettled = query.all()
    session.close()

    if not unsettled:
        return {"settled": 0, "wins": 0, "losses": 0, "pushes": 0, "pnl": 0.0}

    print(f"\nSettling {len(unsettled)} unsettled F5 bet(s)...")

    game_pks = list({b.game_pk for b in unsettled if b.game_pk})
    results: dict[int, dict] = {}
    for pk in game_pks:
        result = fetch_f5_result(pk)
        if result:
            results[pk] = result
        time.sleep(0.3)

    engine  = init_db()
    session = get_session(engine)
    settled = wins = losses = pushes = 0
    total_pnl = 0.0

    for bet in session.query(F5Bet).filter(F5Bet.outcome.is_(None)).all():
        result = results.get(bet.game_pk)
        if not result:
            print(f"  ⏳ F5 {bet.away_team}@{bet.home_team} ({bet.game_date}) — not yet 5 innings")
            continue

        home_f5 = result["home_score_f5"]
        away_f5 = result["away_score_f5"]
        win_f5  = result["home_win_f5"]

        bet.home_score_f5 = home_f5
        bet.away_score_f5 = away_f5
        bet.settled_at    = datetime.now(timezone.utc).isoformat()

        if win_f5 is None:
            bet.outcome        = -1
            bet.profit_dollars = 0.0
            pushes += 1
            print(f"  ~ PUSH  {bet.away_team}@{bet.home_team}  F5 tied {home_f5}-{away_f5}")
        else:
            bet_won = (win_f5 == 1 and bet.bet_side == "home") or \
                      (win_f5 == 0 and bet.bet_side == "away")
            profit  = (bet.stake_dollars * (bet.bet_decimal_odds - 1)
                       if bet_won else -bet.stake_dollars)
            bet.outcome        = 1 if bet_won else 0
            bet.profit_dollars = round(profit, 2)
            result_str = "✓ WON " if bet_won else "✗ LOST"
            side_team  = bet.home_team if bet.bet_side == "home" else bet.away_team
            print(
                f"  {result_str}  F5 {bet.away_team}@{bet.home_team}  "
                f"(F5: {home_f5}-{away_f5})  Bet {side_team} "
                f"{format_american(bet.bet_american_odds)}  ${profit:+.2f}"
            )
            total_pnl += profit
            if bet_won:
                wins += 1
            else:
                losses += 1

        settled += 1

    session.commit()
    session.close()

    if settled:
        print(f"\n  F5 settled {settled}: {wins}W / {losses}L / {pushes} push  "
              f"P&L: ${total_pnl:+.2f}")

    return {"settled": settled, "wins": wins, "losses": losses,
            "pushes": pushes, "pnl": total_pnl}


def fetch_game_result(game_pk: int) -> dict | None:
    """
    Fetch the final score for a game_pk from the MLB Stats API linescore.
    Returns {"home_score": int, "away_score": int} or None if not final.
    """
    try:
        r = requests.get(
            f"{MLB_API_BASE}/api/v1.1/game/{game_pk}/feed/live",
            headers=HEADERS, timeout=20
        )
        r.raise_for_status()
        data = r.json()

        status = (
            data.get("gameData", {})
                .get("status", {})
                .get("detailedState", "")
        )
        if status not in ("Final", "Game Over", "Completed Early"):
            return None

        linescore = data.get("liveData", {}).get("linescore", {})
        teams     = linescore.get("teams", {})
        home_score = teams.get("home", {}).get("runs")
        away_score = teams.get("away", {}).get("runs")

        if home_score is None or away_score is None:
            return None

        return {"home_score": int(home_score), "away_score": int(away_score)}

    except Exception as e:
        print(f"  ⚠ Could not fetch result for game_pk={game_pk}: {e}")
        return None


def settle_bets(date: str | None = None) -> dict:
    """
    Settle all unsettled bets (optionally filtered by date).
    Returns a summary dict with wins, losses, P&L.
    """
    engine  = init_db()
    session = get_session(engine)

    query = session.query(PaperBet).filter(PaperBet.outcome.is_(None))
    if date:
        query = query.filter(PaperBet.game_date == date)

    unsettled = query.all()

    if not unsettled:
        print("  No unsettled bets found.")
        session.close()
        return {"settled": 0, "wins": 0, "losses": 0, "pnl": 0.0}

    # Attempt last-chance CLV capture for bets still missing closing odds.
    # Usually the pre-game job handles this; this is a fallback for delayed games.
    session.close()
    try:
        capture_closing_odds(dry_run=False)
    except Exception as exc:
        print(f"  ⚠ CLV capture skipped: {exc}")
    engine  = init_db()
    session = get_session(engine)
    query = session.query(PaperBet).filter(PaperBet.outcome.is_(None))
    if date:
        query = query.filter(PaperBet.game_date == date)
    unsettled = query.all()

    print(f"\nSettling {len(unsettled)} unsettled bet(s)...")

    settled = wins = losses = 0
    total_pnl = 0.0

    # Group by game_pk to minimise API calls
    game_pks = list({b.game_pk for b in unsettled if b.game_pk})
    results: dict[int, dict] = {}

    for pk in game_pks:
        result = fetch_game_result(pk)
        if result:
            results[pk] = result
        time.sleep(0.3)

    for bet in unsettled:
        result = results.get(bet.game_pk)
        if not result:
            print(f"  ⏳ {bet.away_team}@{bet.home_team} ({bet.game_date}) — "
                  f"not yet final, skipping")
            continue

        home_score = result["home_score"]
        away_score = result["away_score"]
        home_won   = home_score > away_score
        bet_won    = (home_won and bet.bet_side == "home") or \
                     (not home_won and bet.bet_side == "away")

        profit = (bet.stake_dollars * (bet.bet_decimal_odds - 1)
                  if bet_won else -bet.stake_dollars)

        bet.home_score     = home_score
        bet.away_score     = away_score
        bet.outcome        = 1 if bet_won else 0
        bet.profit_dollars = round(profit, 2)
        bet.settled_at     = datetime.now(timezone.utc).isoformat()

        result_str = "✓ WON " if bet_won else "✗ LOST"
        side_team  = bet.home_team if bet.bet_side == "home" else bet.away_team
        print(
            f"  {result_str}  {bet.away_team}@{bet.home_team}  "
            f"({home_score}-{away_score})  "
            f"Bet {side_team} {format_american(bet.bet_american_odds)}  "
            f"${profit:+.2f}"
        )

        settled += 1
        if bet_won:
            wins += 1
            total_pnl += profit
        else:
            losses += 1
            total_pnl += profit   # negative

    session.commit()
    session.close()

    print(f"\n  Settled {settled} bets: {wins}W / {losses}L  "
          f"P&L: ${total_pnl:+.2f}")

    # Also settle any F5 bets
    settle_f5_bets(date=date)

    return {
        "settled": settled, "wins": wins,
        "losses": losses,   "pnl": total_pnl,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settle paper trading bets")
    parser.add_argument("--date", type=str, default=None,
                        help="Only settle bets from this date (YYYY-MM-DD)")
    args = parser.parse_args()
    settle_bets(date=args.date)

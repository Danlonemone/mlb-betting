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

from db.schema import init_db, get_session, PaperBet, F5Bet, PropBet, PitcherSeason
from betting.odds import format_american, american_to_decimal
from config import DEFAULT_BANKROLL
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

    query2 = session.query(F5Bet).filter(F5Bet.outcome.is_(None))
    if date:
        query2 = query2.filter(F5Bet.game_date == date)
    for bet in query2.all():
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

        dec = bet.bet_decimal_odds or american_to_decimal(float(bet.bet_american_odds or 0))
        if dec is None or dec <= 1:
            print(f"  ⚠ No odds for {bet.away_team}@{bet.home_team}, skipping")
            continue
        if bet.bet_decimal_odds is None:
            bet.bet_decimal_odds = dec
        profit = (bet.stake_dollars * (dec - 1) if bet_won else -bet.stake_dollars)

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

    # Also settle any F5 bets and include their P&L
    f5 = settle_f5_bets(date=date)

    return {
        "settled": settled + f5["settled"],
        "wins":    wins    + f5["wins"],
        "losses":  losses  + f5["losses"],
        "pnl":     total_pnl + f5["pnl"],
    }


def fetch_pitcher_ks(mlbam_id: int, game_date: str) -> int | None:
    """
    Fetch actual strikeout total for a pitcher on a given date from the MLB API.
    Returns strikeout count or None if the game isn't final yet.
    """
    try:
        r = requests.get(
            f"{MLB_API_BASE}/api/v1/people/{mlbam_id}/stats",
            params={"stats": "gameLog", "group": "pitching",
                    "season": game_date[:4]},
            headers=HEADERS, timeout=20,
        )
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
    except Exception as e:
        print(f"  ⚠ Could not fetch game log for pitcher {mlbam_id}: {e}")
        return None

    for s in splits:
        if s.get("date", "") == game_date:
            stat = s.get("stat", {})
            # Only settle if game is marked as started (IP > 0)
            ip_str = stat.get("inningsPitched", "0")
            try:
                parts = str(ip_str).split(".")
                ip = int(parts[0]) + (int(parts[1]) if len(parts) > 1 else 0) / 3.0
            except Exception:
                ip = 0.0
            if ip > 0:
                return int(stat.get("strikeOuts", 0) or 0)
    return None


def settle_prop_bets(date: str | None = None) -> dict:
    """
    Settle unsettled pitcher strikeout prop bets.
    Looks up actual K count from the MLB Stats API game log.
    """
    engine  = init_db()
    session = get_session(engine)

    query = session.query(PropBet).filter(
        PropBet.outcome.is_(None),
        PropBet.market == "pitcher_strikeouts",
    )
    if date:
        query = query.filter(PropBet.game_date == date)
    unsettled = query.all()

    if not unsettled:
        print("  No unsettled prop bets found.")
        session.close()
        return {"settled": 0, "wins": 0, "losses": 0, "pnl": 0.0}

    print(f"\nSettling {len(unsettled)} unsettled prop bet(s)...")

    import unicodedata
    def _norm(s: str) -> str:
        return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()

    # Build pitcher name → mlbam_id lookup
    name_to_id: dict[str, int] = {}
    for row in session.query(PitcherSeason).order_by(PitcherSeason.season.desc()).all():
        key = _norm(row.name or "")
        if row.mlbam_id and key and key not in name_to_id:
            name_to_id[key] = row.mlbam_id

    settled = wins = losses = 0
    total_pnl = 0.0

    for bet in unsettled:
        mlbam_id = name_to_id.get(_norm(bet.player_name or ""))
        if not mlbam_id:
            print(f"  ⚠ No pitcher ID for {bet.player_name} — skipping")
            continue

        actual_ks = fetch_pitcher_ks(mlbam_id, bet.game_date)
        time.sleep(0.3)

        if actual_ks is None:
            print(f"  ⏳ {bet.player_name} ({bet.game_date}) — game not final yet")
            continue

        line      = float(bet.line or 0)
        went_over = actual_ks > line        # e.g. actual=5, line=4.5 → True
        bet_won   = (went_over and bet.bet_side == "over") or \
                    (not went_over and bet.bet_side == "under")

        decimal   = american_to_decimal(float(bet.american_odds or 0))
        profit    = round(bet.stake_dollars * (decimal - 1) if bet_won
                          else -bet.stake_dollars, 2)

        bet.actual_value    = float(actual_ks)
        bet.outcome         = 1 if bet_won else 0
        bet.profit_dollars  = profit
        bet.settled_at      = datetime.now(timezone.utc).isoformat()

        result = "✓ WON " if bet_won else "✗ LOST"
        print(f"  {result}  {bet.player_name}  {bet.bet_side} {line}K  "
              f"actual={actual_ks}K  {format_american(bet.american_odds)}  "
              f"${profit:+.2f}")

        total_pnl += profit
        settled   += 1
        if bet_won:
            wins += 1
        else:
            losses += 1

    session.commit()
    session.close()

    if settled:
        print(f"\n  Props settled {settled}: {wins}W / {losses}L  "
              f"P&L: ${total_pnl:+.2f}")

    return {"settled": settled, "wins": wins, "losses": losses, "pnl": total_pnl}


def settle_batter_hit_bets(date: str | None = None) -> dict:
    """
    Settle unsettled batter hits prop bets using the batter_game_logs table.
    """
    from db.schema import BatterGameLog
    engine  = init_db()
    session = get_session(engine)

    query = session.query(PropBet).filter(
        PropBet.outcome.is_(None),
        PropBet.market == "batter_hits",
    )
    if date:
        query = query.filter(PropBet.game_date == date)
    unsettled = query.all()

    if not unsettled:
        session.close()
        return {"settled": 0, "wins": 0, "losses": 0, "pnl": 0.0}

    print(f"\nSettling {len(unsettled)} unsettled batter hits prop bet(s)...")

    settled = wins = losses = pushes = 0
    total_pnl = 0.0

    from datetime import date as _date
    today_str = _date.today().isoformat()

    for bet in unsettled:
        log = (
            session.query(BatterGameLog)
            .filter(
                BatterGameLog.player_name == bet.player_name,
                BatterGameLog.game_date   == bet.game_date,
                BatterGameLog.ab          >  0,
            )
            .first()
        )
        if not log:
            # Auto-void as push after 2 days: player almost certainly DNP.
            # Game logs appear same day or next day when a player plays;
            # 2+ days with no log means they were scratched from the lineup.
            days_old = (_date.fromisoformat(today_str) - _date.fromisoformat(bet.game_date)).days
            if days_old >= 2:
                bet.outcome        = -1
                bet.profit_dollars = 0.0
                bet.settled_at     = datetime.now(timezone.utc).isoformat()
                print(f"  ~ PUSH  {bet.player_name}  {bet.bet_side} {bet.line}H  "
                      f"({bet.game_date})  DNP — auto-voided after {days_old} days")
                settled += 1
                pushes  += 1
            else:
                print(f"  ⏳ {bet.player_name} ({bet.game_date}) — no game log yet")
            continue

        actual_hits = int(log.hits or 0)
        line        = float(bet.line or 0)
        went_over   = actual_hits > line
        bet_won     = (went_over and bet.bet_side == "over") or \
                      (not went_over and bet.bet_side == "under")

        decimal = american_to_decimal(float(bet.american_odds or 0))
        profit  = round(bet.stake_dollars * (decimal - 1) if bet_won
                        else -bet.stake_dollars, 2)

        bet.actual_value   = float(actual_hits)
        bet.outcome        = 1 if bet_won else 0
        bet.profit_dollars = profit
        bet.settled_at     = datetime.now(timezone.utc).isoformat()

        result = "✓ WON " if bet_won else "✗ LOST"
        print(f"  {result}  {bet.player_name}  {bet.bet_side} {line}H  "
              f"actual={actual_hits}H  {format_american(bet.american_odds)}  "
              f"${profit:+.2f}")

        total_pnl += profit
        settled   += 1
        if bet_won:
            wins += 1
        else:
            losses += 1

    session.commit()
    session.close()

    if settled:
        push_txt = f" / {pushes}P" if pushes else ""
        print(f"\n  Batter hits settled {settled}: {wins}W / {losses}L{push_txt}  "
              f"P&L: ${total_pnl:+.2f}")

    return {"settled": settled, "wins": wins, "losses": losses, "pushes": pushes, "pnl": total_pnl}


def _update_bankroll(pnl_delta: float) -> None:
    """Add pnl_delta to the bankroll in data/settings.json."""
    if pnl_delta == 0:
        return
    import json
    settings_path = Path(__file__).parent.parent / "data" / "settings.json"
    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except Exception:
            pass
    prev = float(settings.get("bankroll") or DEFAULT_BANKROLL)
    settings["bankroll"] = round(prev + pnl_delta, 2)
    settings_path.write_text(json.dumps(settings, indent=2))
    print(f"\n  Bankroll updated: ${prev:.2f} → ${settings['bankroll']:.2f}  "
          f"(session P&L: ${pnl_delta:+.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settle paper trading bets")
    parser.add_argument("--date", type=str, default=None,
                        help="Only settle bets from this date (YYYY-MM-DD)")
    args = parser.parse_args()
    ml_result   = settle_bets(date=args.date)
    prop_result = settle_prop_bets(date=args.date)
    hits_result = settle_batter_hit_bets(date=args.date)
    _update_bankroll(ml_result["pnl"] + prop_result["pnl"] + hits_result["pnl"])

"""
Props daily picks runner.

Fetches player prop odds from The Odds API (pitcher_strikeouts, batter_hits),
runs the relevant model, finds edges, and logs recommendations.

Usage:
    python props/props_picks.py
    python props/props_picks.py --markets pitcher_strikeouts batter_hits --bankroll 1000
"""

from __future__ import annotations

import sys
import argparse
import requests
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ODDS_API_KEY, MIN_EDGE, KELLY_FRACTION, MAX_DAILY_EXPOSURE_PCT,
    get_current_bankroll,
)
from db.schema import init_db, get_session, PropBet
from betting.odds import (
    american_to_decimal, remove_vig, compute_edge,
    expected_value, format_american
)
from betting.kelly import kelly_stake
from paper_trade.odds_api import ODDS_API_TEAM_MAP, OddsAPIError

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT    = "baseball_mlb"


def _norm(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()

# The Odds API market key → our internal name
MARKET_MAP = {
    "pitcher_strikeouts": "pitcher_strikeouts",
    "batter_hits":        "batter_hits",
    "batter_home_runs":   "batter_home_runs",
    "batter_total_bases": "batter_total_bases",
}


# ---------------------------------------------------------------------------
# Fetch props odds from The Odds API
# ---------------------------------------------------------------------------

def fetch_props_odds(markets: list[str]) -> list[dict]:
    """
    Fetch player prop odds for today's MLB games.
    markets: list of Odds API market keys, e.g. ["pitcher_strikeouts", "batter_hits"]
    """
    if not ODDS_API_KEY or ODDS_API_KEY == "your_key_here":
        raise OddsAPIError("ODDS_API_KEY not set in .env")

    # First get the list of event IDs for today
    events_url = f"{BASE_URL}/sports/{SPORT}/events"
    r = requests.get(events_url, params={"apiKey": ODDS_API_KEY}, timeout=30)
    r.raise_for_status()
    events = r.json()

    today = datetime.now().strftime("%Y-%m-%d")
    today_events = [
        e for e in events
        if e.get("commence_time", "")[:10] == today
    ]

    all_props = []

    for event in today_events:
        event_id  = event["id"]
        home_name = event.get("home_team", "")
        away_name = event.get("away_team", "")
        home_abbr = ODDS_API_TEAM_MAP.get(home_name, home_name)
        away_abbr = ODDS_API_TEAM_MAP.get(away_name, away_name)

        odds_url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
        try:
            r2 = requests.get(odds_url, params={
                "apiKey":     ODDS_API_KEY,
                "regions":    "us",
                "markets":    ",".join(markets),
                "oddsFormat": "american",
            }, timeout=30)
            remaining = r2.headers.get("x-requests-remaining", "?")
            r2.raise_for_status()
            event_odds = r2.json()
        except Exception as e:
            print(f"  ⚠ Could not fetch props for {away_abbr}@{home_abbr}: {e}")
            continue

        for bookmaker in event_odds.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                market_key = market["key"]
                if market_key not in MARKET_MAP:
                    continue

                # Each outcome is one player's over or under
                outcomes = market.get("outcomes", [])
                # Group by player name
                players: dict[str, dict] = {}
                for outcome in outcomes:
                    name  = outcome.get("description", outcome.get("name", ""))
                    side  = "over" if "Over" in outcome.get("name", "") else "under"
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if name not in players:
                        players[name] = {"line": point, "over": None, "under": None}
                    players[name][side] = price

                for player_name, odds_dict in players.items():
                    if odds_dict["over"] is None or odds_dict["under"] is None:
                        continue
                    all_props.append({
                        "game_date":    today,
                        "home_team":    home_abbr,
                        "away_team":    away_abbr,
                        "player_name":  player_name,
                        "market":       market_key,
                        "line":         odds_dict["line"],
                        "over_american":  odds_dict["over"],
                        "under_american": odds_dict["under"],
                        "bookmaker":    bookmaker["key"],
                        "event_id":     event_id,
                        "game_pk":      0,   # fill from schedule if needed
                    })

    print(f"  {len(all_props)} player prop lines fetched for {today}")
    return all_props


def build_today_game_pk_map(engine, game_date: str) -> dict[tuple[str, str], int]:
    """
    Build {(home_team, away_team): game_pk} from DB, then MLB's live schedule.

    Today's games usually are not inserted into the completed-games table yet,
    so the live schedule fallback is what keeps prop bets linked to real games.
    """
    from sqlalchemy import text as _text

    with engine.connect() as conn:
        rows = conn.execute(_text(
            "SELECT game_pk, home_team, away_team FROM games WHERE game_date = :d"
        ), {"d": game_date}).fetchall()

    game_pk_map = {(r.home_team, r.away_team): int(r.game_pk) for r in rows}
    missing_today_schedule = not game_pk_map
    if missing_today_schedule:
        try:
            from ingestion.mlb_api import fetch_date_game_map
            game_pk_map.update(fetch_date_game_map(game_date))
        except Exception as exc:
            print(f"  ⚠ Could not resolve live game_pks: {exc}")
    return game_pk_map


# ---------------------------------------------------------------------------
# Score props against models
# ---------------------------------------------------------------------------

def build_batter_kpct_totals(engine, before_date: str) -> dict[int, tuple[int, int]]:
    """{mlbam_id: (pa, strikeouts)} season-to-date strictly before before_date."""
    from sqlalchemy import text as _text
    season = int(before_date[:4])
    with engine.connect() as conn:
        rows = conn.execute(_text(
            "SELECT mlbam_id, SUM(pa) AS pa, SUM(strikeouts) AS k "
            "FROM batter_game_logs "
            "WHERE season = :s AND game_date < :d AND pa > 0 "
            "GROUP BY mlbam_id"
        ), {"s": season, "d": before_date}).fetchall()
    return {int(r.mlbam_id): (int(r.pa or 0), int(r.k or 0)) for r in rows}


def lineup_k_pct(
    player_ids: list[int],
    totals: dict[int, tuple[int, int]],
    min_pa: int = 150,
) -> float | None:
    """PA-weighted K% of a confirmed lineup; None if too little history."""
    tot_pa = sum(totals[p][0] for p in player_ids if p in totals)
    tot_k  = sum(totals[p][1] for p in player_ids if p in totals)
    return tot_k / tot_pa if tot_pa >= min_pa else None


def score_strikeout_props(
    props: list[dict],
    ump_k9_by_game: dict[int, float] | None = None,
) -> list[dict]:
    """Run the K model against pitcher strikeout props."""
    from props.strikeout_model import (
        predict_strikeouts, build_opp_k9_cache, build_prior_swstr_cache,
    )
    from db.schema import get_engine, get_session, PitcherSeason

    engine  = init_db()
    session = get_session(engine)

    # Name → (mlbam_id, team) from the most recent season in DB
    name_to_info: dict[str, tuple[int, str]] = {}
    for row in (
        session.query(PitcherSeason)
        .order_by(PitcherSeason.season.desc())
        .all()
    ):
        key = _norm(row.name) if row.name else ""
        if row.mlbam_id and key and key not in name_to_info:
            name_to_info[key] = (row.mlbam_id, row.team or "")
    session.close()

    today = datetime.now().strftime("%Y-%m-%d")
    game_pk_map = build_today_game_pk_map(engine, today)

    # Pre-build caches once for all pitchers (avoid rebuilding per call)
    opp_k9_cache  = build_opp_k9_cache(engine)
    swstr_cache   = build_prior_swstr_cache(engine)
    batter_totals = build_batter_kpct_totals(engine, today)

    # Confirmed lineups (posted ~2h before first pitch) → opposing lineup K%
    confirmed: dict = {}
    try:
        from ingestion.mlb_api import fetch_today_lineups
        confirmed = fetch_today_lineups(today)
    except Exception as exc:
        print(f"  ⚠ Could not fetch lineups for K props: {exc}")
    n_lineups = sum(1 for v in confirmed.values() if v.get("lineups_posted"))
    if confirmed:
        print(f"  Confirmed lineups for props: {n_lineups}/{len(confirmed)} games")

    scored = []
    no_match = set()
    no_history = set()

    for prop in props:
        if prop["market"] != "pitcher_strikeouts":
            continue
        pitcher_name = prop["player_name"]
        info = name_to_info.get(_norm(pitcher_name))
        if info is None:
            no_match.add(pitcher_name)
            continue
        pitcher_id, pitcher_team = info
        pitcher_is_home = (pitcher_team == prop["home_team"])
        opponent_team   = prop["away_team"] if pitcher_is_home else prop["home_team"]
        prop["game_pk"] = game_pk_map.get((prop["home_team"], prop["away_team"]), 0)

        ump_k9_vs_avg = (ump_k9_by_game or {}).get(prop.get("game_pk", 0), 0.0)

        # Opposing lineup K% from the confirmed batting order, when posted
        opp_lineup_kpct = None
        conf = confirmed.get(prop["game_pk"], {})
        if conf.get("lineups_posted"):
            opp_ids = (conf.get("away_lineup_ids") if pitcher_is_home
                       else conf.get("home_lineup_ids")) or []
            opp_lineup_kpct = lineup_k_pct(opp_ids, batter_totals)

        result = predict_strikeouts(
            pitcher_id       = pitcher_id,
            game_date        = prop["game_date"],
            line             = prop["line"],
            home_team        = prop["home_team"],
            opponent_team    = opponent_team,
            pitcher_is_home  = pitcher_is_home,
            opp_k9_cache     = opp_k9_cache,
            ump_k9_vs_avg    = ump_k9_vs_avg,
            opp_lineup_k_pct = opp_lineup_kpct,
            swstr_cache      = swstr_cache,
        )
        if result is None:
            no_history.add(pitcher_name)
            continue

        prop["pitcher_id"]       = pitcher_id
        prop["player_team"]      = pitcher_team
        prop["player_opponent"]  = opponent_team
        prop["expected_k"]       = result["expected_k"]
        prop["model_prob_over"]  = result["prob_over"]
        prop["model_prob_under"] = result["prob_under"]
        scored.append(prop)

    if no_match:
        print(f"  ⚠ No DB match: {', '.join(sorted(no_match))}")
    if no_history:
        print(f"  ⚠ Insufficient history: {', '.join(sorted(no_history))}")
    print(f"  Scored {len(scored)} strikeout props")
    return scored


def score_hits_props(props: list[dict]) -> list[dict]:
    """Run the hits model against batter_hits props."""
    from props.hits_model import predict_hits, build_opp_sp_era_cache
    from db.schema import get_engine, get_session, BatterGameLog
    from sqlalchemy import text as _text

    engine  = init_db()
    session = get_session(engine)

    # Build name → mlbam_id from batter_game_logs (most recent appearance)
    with engine.connect() as conn:
        rows = conn.execute(_text(
            "SELECT DISTINCT mlbam_id, player_name FROM batter_game_logs "
            "WHERE player_name IS NOT NULL ORDER BY game_date DESC"
        )).fetchall()
    name_to_id: dict[str, int] = {}
    for r in rows:
        key = _norm(r.player_name or "")
        if key and key not in name_to_id:
            name_to_id[key] = r.mlbam_id

    # game_pk → opp SP ERA cache
    opp_era_cache = build_opp_sp_era_cache(engine)

    today = datetime.now().strftime("%Y-%m-%d")
    game_pk_map = build_today_game_pk_map(engine, today)

    session.close()

    scored = []
    no_match = set()
    no_history = set()

    for prop in props:
        if prop["market"] != "batter_hits":
            continue
        batter_name = prop["player_name"]
        batter_id   = name_to_id.get(_norm(batter_name))
        if batter_id is None:
            no_match.add(batter_name)
            continue

        game_pk  = game_pk_map.get((prop["home_team"], prop["away_team"]), 0)
        prop["game_pk"] = game_pk

        # Determine home/away for the batter
        # The Odds API doesn't tell us — use team abbreviation from batter_game_logs
        # Approximate: if batter's most recent team = home_team → home
        with engine.connect() as conn:
            team_row = conn.execute(_text(
                "SELECT team FROM batter_game_logs WHERE mlbam_id=:id "
                "ORDER BY game_date DESC LIMIT 1"
            ), {"id": batter_id}).fetchone()
        batter_team = team_row[0] if team_row else ""
        home_away = "home" if batter_team == prop["home_team"] else "away"

        result = predict_hits(
            batter_id     = batter_id,
            game_date     = prop["game_date"],
            line          = prop["line"],
            game_pk       = game_pk or None,
            home_away     = home_away,
            opp_era_cache = opp_era_cache,
        )
        if result is None:
            no_history.add(batter_name)
            continue

        prop["batter_id"]        = batter_id
        prop["player_team"]      = batter_team
        prop["player_opponent"]  = prop["away_team"] if batter_team == prop["home_team"] else prop["home_team"]
        prop["expected_hits"]    = result["expected_hits"]
        prop["model_prob_over"]  = result["prob_over"]
        prop["model_prob_under"] = result["prob_under"]
        scored.append(prop)

    if no_match:
        print(f"  ⚠ No DB match (hits): {', '.join(sorted(no_match)[:5])}{'...' if len(no_match)>5 else ''}")
    if no_history:
        print(f"  ⚠ Insufficient history (hits): {len(no_history)} batters")
    print(f"  Scored {len(scored)} hits props")
    return scored


def score_total_bases_props(props: list[dict]) -> list[dict]:
    """Run the TB model against batter_total_bases props."""
    from props.total_bases_model import predict_total_bases
    from props.hits_model import build_opp_sp_era_cache
    from sqlalchemy import text as _text

    engine = init_db()

    with engine.connect() as conn:
        rows = conn.execute(_text(
            "SELECT DISTINCT mlbam_id, player_name FROM batter_game_logs "
            "WHERE player_name IS NOT NULL ORDER BY game_date DESC"
        )).fetchall()
    name_to_id: dict[str, int] = {}
    for r in rows:
        key = _norm(r.player_name or "")
        if key and key not in name_to_id:
            name_to_id[key] = r.mlbam_id

    opp_era_cache = build_opp_sp_era_cache(engine)
    today         = datetime.now().strftime("%Y-%m-%d")
    game_pk_map   = build_today_game_pk_map(engine, today)

    scored = []
    no_match = set()
    no_history = set()

    for prop in props:
        if prop["market"] != "batter_total_bases":
            continue
        batter_id = name_to_id.get(_norm(prop["player_name"]))
        if batter_id is None:
            no_match.add(prop["player_name"])
            continue

        game_pk = game_pk_map.get((prop["home_team"], prop["away_team"]), 0)
        prop["game_pk"] = game_pk

        with engine.connect() as conn:
            team_row = conn.execute(_text(
                "SELECT team FROM batter_game_logs WHERE mlbam_id=:id "
                "ORDER BY game_date DESC LIMIT 1"
            ), {"id": batter_id}).fetchone()
        batter_team = team_row[0] if team_row else ""
        home_away = "home" if batter_team == prop["home_team"] else "away"

        result = predict_total_bases(
            batter_id     = batter_id,
            game_date     = prop["game_date"],
            line          = prop["line"],
            game_pk       = game_pk or None,
            home_away     = home_away,
            opp_era_cache = opp_era_cache,
        )
        if result is None:
            no_history.add(prop["player_name"])
            continue

        prop["batter_id"]        = batter_id
        prop["player_team"]      = batter_team
        prop["player_opponent"]  = prop["away_team"] if batter_team == prop["home_team"] else prop["home_team"]
        prop["expected_tb"]      = result["expected_tb"]
        prop["model_prob_over"]  = result["prob_over"]
        prop["model_prob_under"] = result["prob_under"]
        scored.append(prop)

    if no_match:
        print(f"  ⚠ No DB match (TB): {len(no_match)} batters")
    if no_history:
        print(f"  ⚠ Insufficient history (TB): {len(no_history)} batters")
    print(f"  Scored {len(scored)} total-bases props")
    return scored


def find_prop_edges(
    props: list[dict],
    bankroll: float,
    min_edge: float = MIN_EDGE,
) -> list[dict]:
    """
    Compare model probability vs the CONSENSUS vig-free probability, then
    bet at the best available price across books.

    Why consensus: scoring each book's quote against its own two-sided
    market lets a single soft book both inflate the measured edge AND
    receive the bet — exactly the quotes most likely to be stale or wrong.
    Using the median fair probability across all books quoting the same
    (player, market, line) means an edge only registers when the model
    disagrees with the MARKET, and the soft book's good price is then
    captured at execution via best-price selection.
    """
    import statistics

    # Group identical propositions across books
    groups: dict[tuple, list[dict]] = {}
    for prop in props:
        if prop.get("over_american") is None or prop.get("under_american") is None:
            continue
        key = (_norm(prop["player_name"]), prop["market"], float(prop["line"] or 0))
        groups.setdefault(key, []).append(prop)

    recs = []
    for key, quotes in groups.items():
        # Consensus vig-free over-probability = median across books
        fair_overs = []
        for q in quotes:
            over_imp  = 1.0 / american_to_decimal(q["over_american"])
            under_imp = 1.0 / american_to_decimal(q["under_american"])
            fair_overs.append(over_imp / (over_imp + under_imp))
        fair_over  = statistics.median(fair_overs)
        fair_under = 1.0 - fair_over

        # Model probs are identical across quotes of the same prop
        base        = quotes[0]
        model_over  = base.get("model_prob_over",  0.5)
        model_under = base.get("model_prob_under", 0.5)

        edge_over  = compute_edge(model_over,  fair_over)
        edge_under = compute_edge(model_under, fair_under)

        if edge_over >= min_edge and edge_over >= edge_under:
            side, edge, model_p, fair_p = "over", edge_over, model_over, fair_over
            best_q   = max(quotes, key=lambda q: american_to_decimal(q["over_american"]))
            american = best_q["over_american"]
        elif edge_under >= min_edge:
            side, edge, model_p, fair_p = "under", edge_under, model_under, fair_under
            best_q   = max(quotes, key=lambda q: american_to_decimal(q["under_american"]))
            american = best_q["under_american"]
        else:
            continue

        decimal = american_to_decimal(american)
        stake   = kelly_stake(model_p, decimal, bankroll, fraction=KELLY_FRACTION)
        if stake <= 0:
            continue

        over_imp  = 1.0 / american_to_decimal(best_q["over_american"])
        under_imp = 1.0 / american_to_decimal(best_q["under_american"])

        recs.append({
            **best_q,                      # keeps line/teams + the BEST book's name
            "bet_side":     side,
            "model_prob":   model_p,
            "fair_prob":    fair_p,        # consensus, not the soft book's
            "edge":         edge,
            "american_odds": american,
            "decimal_odds": decimal,
            "ev_per_unit":  expected_value(model_p, decimal),
            "stake":        stake,
            "overround":    over_imp + under_imp,
            "n_books":      len(quotes),
        })

    # Deduplicate. Batter hits and total bases on the same player are nearly
    # the same position (a 2-hit night usually clears both), so batter
    # markets collapse to ONE bet per player; pitcher markets stay per-market.
    _BATTER_MARKETS = {"batter_hits", "batter_total_bases"}
    seen: dict[tuple, dict] = {}
    for r in sorted(recs, key=lambda r: r["edge"], reverse=True):
        market_key = "batter_core" if r["market"] in _BATTER_MARKETS else r["market"]
        k = (r["player_name"], market_key)
        if k not in seen:
            seen[k] = r
    return list(seen.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_props_picks(
    markets: list[str] | None = None,
    bankroll: float | None = None,
    min_edge: float = MIN_EDGE,
    dry_run: bool = False,
):
    if bankroll is None:
        bankroll = get_current_bankroll()
    markets = markets or ["pitcher_strikeouts", "batter_hits", "batter_total_bases"]
    # Drop markets with no model yet
    active = {"pitcher_strikeouts", "batter_hits", "batter_total_bases"}
    markets = [m for m in markets if m in active]
    today   = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"Props Picks — {today}")
    print(f"Markets: {', '.join(markets)}")
    print(f"{'='*60}")

    print("\n[1/3] Fetching props odds...")
    try:
        all_props = fetch_props_odds(markets)
    except OddsAPIError as e:
        print(f"  ERROR: {e}")
        return []

    print("\n[2/3] Scoring props against models...")

    # Build ump K9 lookup for today (game_pk → ump_k9_vs_avg)
    ump_k9_by_game: dict[int, float] = {}
    try:
        from ingestion.mlb_api import fetch_today_lineups
        from sqlalchemy import text as _text
        lineups = fetch_today_lineups(today)
        _engine = init_db()
        with _engine.connect() as _conn:
            ump_k_stats = {
                r.ump_name: float(r.k9_vs_avg)
                for r in _conn.execute(_text(
                    "SELECT ump_name, k9_vs_avg FROM umpire_stats WHERE k9_vs_avg IS NOT NULL"
                )).fetchall()
            }
        for gp, info in lineups.items():
            ump = info.get("hp_ump_name")
            if ump and ump in ump_k_stats:
                ump_k9_by_game[gp] = ump_k_stats[ump]
    except Exception:
        pass

    scored = []
    if "pitcher_strikeouts" in markets:
        scored += score_strikeout_props(all_props, ump_k9_by_game=ump_k9_by_game)
    if "batter_hits" in markets:
        scored += score_hits_props(all_props)
    if "batter_total_bases" in markets:
        scored += score_total_bases_props(all_props)

    print("\n[3/3] Finding edges...")
    recs = find_prop_edges(scored, bankroll=bankroll, min_edge=min_edge)

    print(f"\n{'─'*60}")
    if not recs:
        print(f"  No prop bets recommended today at {min_edge:.0%} edge.")
    else:
        print(f"  {len(recs)} prop bet(s) recommended:\n")
        print(f"  {'Player':<22} {'Line':>5} {'Side':>6} {'Odds':>6} "
              f"{'Exp K':>6} {'Model':>7} {'Edge':>6} {'Stake':>8}")
        print(f"  {'─'*68}")
        for r in recs:
            if r["market"] == "pitcher_strikeouts":
                exp_str = f"{r.get('expected_k', 0):>4.1f}K "
            elif r["market"] == "batter_total_bases":
                exp_str = f"{r.get('expected_tb', 0):>4.2f}TB"
            else:
                exp_str = f"{r.get('expected_hits', 0):>4.2f}H"
            print(
                f"  {r['player_name']:<22} {r['line']:>5.1f} {r['bet_side']:>6} "
                f"{format_american(r['american_odds']):>6} "
                f"{exp_str} "
                f"{r['model_prob']:>6.1%} {r['edge']:>+5.1%} "
                f"${r['stake']:>7.2f}"
            )

    if not dry_run and recs:
        from paper_trade.daily_picks import todays_logged_exposure
        engine  = init_db()
        session = get_session(engine)
        logged  = 0
        capped  = 0
        exposure_cap = MAX_DAILY_EXPOSURE_PCT * bankroll
        exposure     = todays_logged_exposure(session, today)
        for r in sorted(recs, key=lambda x: x["edge"], reverse=True):
            existing = (
                session.query(PropBet)
                .filter(
                    PropBet.game_pk    == r.get("game_pk", 0),
                    PropBet.player_name == r["player_name"],
                    PropBet.market     == r["market"],
                )
                .first()
            )
            if existing:
                continue
            if exposure + r["stake"] > exposure_cap:
                capped += 1
                continue
            exposure += r["stake"]
            session.add(PropBet(
                game_pk        = r.get("game_pk", 0),
                game_date      = r["game_date"],
                player_id      = r.get("pitcher_id") or r.get("batter_id"),
                player_name    = r["player_name"],
                team           = r.get("player_team") or r.get("home_team", ""),
                opponent       = r.get("player_opponent") or r.get("away_team", ""),
                market         = r["market"],
                line           = r["line"],
                bet_side       = r["bet_side"],
                american_odds  = r["american_odds"],
                decimal_odds   = r["decimal_odds"],
                fair_prob      = r["fair_prob"],
                model_prob     = r["model_prob"],
                edge           = r["edge"],
                stake_fraction = r["stake"] / bankroll,
                stake_dollars  = r["stake"],
                bankroll_at_bet = bankroll,
                bookmaker      = r.get("bookmaker", ""),
                is_paper       = 1,
                created_at     = datetime.now(timezone.utc).isoformat(),
            ))
            logged += 1
        session.commit()
        session.close()
        if logged:
            print(f"\n  ✓ {logged} prop bet(s) logged.")
        if capped:
            print(f"  {capped} prop bet(s) skipped by the "
                  f"{MAX_DAILY_EXPOSURE_PCT:.0%} daily exposure cap.")

    return recs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+",
                        default=["pitcher_strikeouts", "batter_hits", "batter_total_bases"])
    parser.add_argument("--bankroll", type=float, default=get_current_bankroll())
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()
    run_props_picks(
        markets  = args.markets,
        bankroll = args.bankroll,
        min_edge = args.min_edge,
        dry_run  = args.dry_run,
    )

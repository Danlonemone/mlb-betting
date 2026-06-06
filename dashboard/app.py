"""
Local web dashboard for the MLB moneyline paper-trading system.

Run:
    python dashboard/app.py --bankroll 75.04 --port 8765
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from betting.odds import format_american
from betting.recommender import recommend
import base64
import hashlib
import os

from config import DB_PATH, MIN_EDGE, DEFAULT_BANKROLL as CONFIG_DEFAULT_BANKROLL
from paper_trade.live_features import build_live_features
from paper_trade.odds_api import fetch_today_odds

# Set via --password flag or DASHBOARD_PASSWORD env var.
# Empty string = no auth (local-only use).
DASHBOARD_PASSWORD: str = ""


def _check_auth(handler) -> bool:
    """Return True if the request is authenticated (or no password set)."""
    if not DASHBOARD_PASSWORD:
        return True
    auth = handler.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth[6:]).decode("utf-8")
        _, pwd   = decoded.split(":", 1)
        return hashlib.sha256(pwd.encode()).hexdigest() == \
               hashlib.sha256(DASHBOARD_PASSWORD.encode()).hexdigest()
    except Exception:
        return False


def _send_auth_challenge(handler) -> None:
    handler.send_response(401)
    handler.send_header("WWW-Authenticate", 'Basic realm="MLB Betting"')
    handler.send_header("Content-Length", "0")
    handler.end_headers()


LIVE_CACHE: dict[str, object] = {
    "ts": 0.0, "bankroll": None, "min_edge": None, "payload": None,
}
PROPS_CACHE: dict[str, object] = {
    "ts": 0.0, "bankroll": None, "min_edge": None, "payload": None,
}
CACHE_SECONDS = 300
DEFAULT_BANKROLL = CONFIG_DEFAULT_BANKROLL
DEFAULT_MIN_EDGE = MIN_EDGE


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _money(value: float | None) -> str:
    value = float(value or 0)
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{float(value) * 100:+.{digits}f}%"


def _american(value: float | None) -> str:
    if value is None:
        return "--"
    return format_american(value)


def _safe_error(exc: Exception) -> str:
    msg = str(exc)
    msg = re.sub(r"apiKey=[^&\s)]+", "apiKey=redacted", msg)
    msg = re.sub(r"ODDS_API_KEY=[^&\s)]+", "ODDS_API_KEY=redacted", msg)
    if "api.the-odds-api.com" in msg and (
        "Failed to resolve" in msg or "NameResolutionError" in msg
    ):
        return "Could not fetch live odds. Check network access or Odds API availability."
    return msg


def _row_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def _paper_metrics(conn: sqlite3.Connection, bankroll: float) -> dict:
    rows = [
        _row_dict(r)
        for r in conn.execute(
            "SELECT * FROM paper_bets ORDER BY game_date, game_pk, id"
        ).fetchall()
    ]
    settled = [r for r in rows if r["outcome"] is not None]
    pending = [r for r in rows if r["outcome"] is None]

    # Include prop and F5 P&L so all bet types flow into every metric
    prop_rows = conn.execute(
        "SELECT outcome, stake_dollars, profit_dollars FROM prop_bets"
    ).fetchall()
    f5_rows = conn.execute(
        "SELECT outcome, stake_dollars, profit_dollars FROM f5_paper_bets"
    ).fetchall()
    prop_settled_rows = [r for r in prop_rows if r["outcome"] is not None]
    f5_settled_rows   = [r for r in f5_rows   if r["outcome"] is not None]
    prop_profit       = sum(float(r["profit_dollars"] or 0) for r in prop_settled_rows)
    f5_profit         = sum(float(r["profit_dollars"] or 0) for r in f5_settled_rows)
    prop_staked       = sum(float(r["stake_dollars"] or 0) for r in prop_settled_rows)
    f5_staked         = sum(float(r["stake_dollars"] or 0) for r in f5_settled_rows)
    prop_pending_risk = sum(float(r["stake_dollars"] or 0) for r in prop_rows if r["outcome"] is None)
    f5_pending_risk   = sum(float(r["stake_dollars"] or 0) for r in f5_rows   if r["outcome"] is None)
    prop_wins  = sum(int(r["outcome"]) for r in prop_settled_rows)
    f5_wins    = sum(1 for r in f5_settled_rows if int(r["outcome"]) == 1)
    f5_losses  = sum(1 for r in f5_settled_rows if int(r["outcome"]) == 0)

    starting_bankroll = bankroll

    ml_wins   = sum(int(r["outcome"] or 0) for r in settled)
    ml_losses = len(settled) - ml_wins
    total_staked    = (sum(float(r["stake_dollars"] or 0) for r in settled)
                       + prop_staked + f5_staked)
    total_profit    = (sum(float(r["profit_dollars"] or 0) for r in settled)
                       + prop_profit + f5_profit)
    pending_at_risk = (sum(float(r["stake_dollars"] or 0) for r in pending)
                       + prop_pending_risk + f5_pending_risk)
    wins   = ml_wins   + prop_wins + f5_wins
    losses = ml_losses + (len(prop_settled_rows) - prop_wins) + f5_losses
    roi    = total_profit / total_staked if total_staked else 0.0
    current_bankroll   = starting_bankroll + total_profit
    available_bankroll = current_bankroll - pending_at_risk

    clv_rows = [r for r in rows if r["clv"] is not None]
    mean_clv = (
        sum(float(r["clv"]) for r in clv_rows) / len(clv_rows)
        if clv_rows
        else None
    )
    positive_clv = (
        sum(1 for r in clv_rows if float(r["clv"]) > 0) / len(clv_rows)
        if clv_rows
        else None
    )

    curve = []
    cum_profit = 0.0
    cum_staked = 0.0
    for i, r in enumerate(settled, start=1):
        cum_profit += float(r["profit_dollars"] or 0)
        cum_staked += float(r["stake_dollars"] or 0)
        curve.append({
            "x": i,
            "date": r["game_date"],
            "bankroll": round(starting_bankroll + cum_profit, 2),
            "roi": round(cum_profit / cum_staked, 4) if cum_staked else 0,
        })

    latest = []
    for r in rows[-8:][::-1]:
        side_team = r["home_team"] if r["bet_side"] == "home" else r["away_team"]
        latest.append({
            "game_date": r["game_date"],
            "matchup": f"{r['away_team']} @ {r['home_team']}",
            "side": side_team,
            "odds": _american(r["bet_american_odds"]),
            "edge": float(r["edge"] or 0),
            "stake": float(r["stake_dollars"] or 0),
            "outcome": (
                "Pending"
                if r["outcome"] is None
                else "Won"
                if int(r["outcome"]) == 1
                else "Lost"
            ),
            "profit": r["profit_dollars"],
            "clv": r["clv"],
        })

    return {
        "starting_bankroll": starting_bankroll,
        "current_bankroll": current_bankroll,
        "available_bankroll": available_bankroll,
        "pending_at_risk": pending_at_risk,
        "total_logged": len(rows) + len(prop_rows) + len(f5_rows),
        "settled": len(settled) + len(prop_settled_rows) + len(f5_settled_rows),
        "pending": (len(pending)
                    + sum(1 for r in prop_rows if r["outcome"] is None)
                    + sum(1 for r in f5_rows   if r["outcome"] is None)),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / (wins + losses) if (wins + losses) else 0.0,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi,
        "mean_edge": (
            sum(float(r["edge"] or 0) for r in rows) / len(rows)
            if rows
            else 0.0
        ),
        "mean_clv": mean_clv,
        "positive_clv": positive_clv,
        "clv_count": len(clv_rows),
        "curve": curve,
        "latest_bets": latest,
    }


def _analytics_payload() -> dict:
    """Analytics tab: CLV series, weekly P&L, edge buckets, automation status, sample progress."""
    import os
    from datetime import datetime as dt

    with _connect() as conn:
        clv_rows = conn.execute(
            "SELECT game_date, clv, home_team, away_team, bet_side "
            "FROM paper_bets WHERE clv IS NOT NULL AND outcome IS NOT NULL "
            "ORDER BY game_date, id"
        ).fetchall()
        clv_series = []
        for i, r in enumerate(clv_rows, 1):
            side = r["home_team"] if r["bet_side"] == "home" else r["away_team"]
            clv_series.append({
                "x": i, "date": r["game_date"],
                "clv": round(float(r["clv"]), 4),
                "label": f"{r['away_team']}@{r['home_team']} {side}",
            })

        week_rows = conn.execute(
            "SELECT strftime('%Y-W%W', game_date) AS week, "
            "MIN(game_date) AS week_start, "
            "ROUND(SUM(profit_dollars),2) AS pnl, COUNT(*) AS count, "
            "SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END) AS wins "
            "FROM paper_bets WHERE outcome IS NOT NULL "
            "GROUP BY week ORDER BY week"
        ).fetchall()
        cum = 0.0
        pnl_curve = []
        for i, r in enumerate(week_rows, 1):
            wk_pnl = float(r["pnl"] or 0)
            cum += wk_pnl
            pnl_curve.append({
                "x": i, "date": r["week_start"], "label": r["week"],
                "pnl": round(wk_pnl, 2), "cum_pnl": round(cum, 2),
            })

        settled = conn.execute(
            "SELECT edge, profit_dollars, stake_dollars, outcome "
            "FROM paper_bets WHERE outcome IS NOT NULL"
        ).fetchall()
        edge_buckets = []
        for label, lo, hi in [
            ("10–15%", 0.10, 0.15), ("15–20%", 0.15, 0.20),
            ("20–25%", 0.20, 0.25), ("25%+",   0.25, 1.00),
        ]:
            rows = [r for r in settled if lo <= float(r["edge"] or 0) < hi]
            if not rows:
                continue
            wins = sum(1 for r in rows if int(r["outcome"]) == 1)
            total_stake  = sum(float(r["stake_dollars"] or 0) for r in rows)
            total_profit = sum(float(r["profit_dollars"] or 0) for r in rows)
            edge_buckets.append({
                "bucket": label, "count": len(rows),
                "wins": wins, "losses": len(rows) - wins,
                "roi": round(total_profit / total_stake, 4) if total_stake else 0,
            })

        settled_count = len(settled)

    log_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    )
    jobs = [
        ("Morning picks",    "morning.log",      "morning_error.log"),
        ("Morning settle",   "morningsettle.log", "morningsettle_error.log"),
        ("CLV capture",      "clvcapture.log",    "clvcapture_error.log"),
        ("Evening settle",   "evening.log",       "evening_error.log"),
    ]
    auto_status = []
    for name, fname, efname in jobs:
        path  = os.path.join(log_dir, fname)
        epath = os.path.join(log_dir, efname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            mtime = os.path.getmtime(path)
            last_run = dt.fromtimestamp(mtime).strftime("%m/%d %H:%M")
            has_error = (os.path.exists(epath) and os.path.getsize(epath) > 0
                         and os.path.getmtime(epath) >= mtime)
            auto_status.append({"name": name, "last_run": last_run, "ok": not has_error})
        else:
            auto_status.append({"name": name, "last_run": "Never", "ok": None})

    return {
        "clv_series":      clv_series,
        "pnl_curve":       pnl_curve,
        "edge_buckets":    edge_buckets,
        "auto_status":     auto_status,
        "sample_progress": {"settled": settled_count, "target": 30},
    }


def _data_freshness(conn: sqlite3.Connection) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    season = int(today[:4])
    game_summary = conn.execute(
        "SELECT COUNT(*) AS n, MAX(game_date) AS max_date "
        "FROM games WHERE season = ?",
        (season,),
    ).fetchone()
    total_games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    close_2025 = conn.execute(
        "SELECT COUNT(*) FROM games "
        "WHERE season = 2025 AND home_close_american IS NOT NULL"
    ).fetchone()[0]
    return {
        "today": today,
        "season": season,
        "season_games": int(game_summary["n"] or 0),
        "latest_game_date": game_summary["max_date"],
        "total_games": int(total_games or 0),
        "close_2025": int(close_2025 or 0),
    }


def _logged_today(conn: sqlite3.Connection, today: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM paper_bets WHERE game_date = ? ORDER BY edge DESC",
        (today,),
    ).fetchall()
    out = []
    for r in rows:
        side_team = r["home_team"] if r["bet_side"] == "home" else r["away_team"]
        out.append({
            "game_pk":  r["game_pk"],
            "bet_side": r["bet_side"],
            "matchup": f"{r['away_team']} @ {r['home_team']}",
            "side": side_team,
            "odds": _american(r["bet_american_odds"]),
            "model_prob": float(r["model_prob"] or 0),
            "fair_prob": float(r["fair_prob"] or 0),
            "edge": float(r["edge"] or 0),
            "stake": float(r["stake_dollars"] or 0),
            "bookmaker": r["bookmaker"] or "",
            "status": "Pending" if r["outcome"] is None else "Settled",
        })
    return out


def _live_recommendations(bankroll: float, min_edge: float, refresh: bool) -> dict:
    now = time.time()
    cached = LIVE_CACHE["payload"]
    if (
        not refresh
        and cached is not None
        and now - float(LIVE_CACHE["ts"] or 0) < CACHE_SECONDS
        and LIVE_CACHE["bankroll"] == bankroll
    ):
        payload = dict(cached)
        payload["cached"] = True
        return payload

    try:
        odds_games = fetch_today_odds()
        feature_rows = build_live_features(odds_games)
        recs = recommend(
            feature_rows,
            bankroll=bankroll,
            min_edge=0.03,
            model_type="logistic",
        )
        picks = []
        for r in recs:
            _fr = next((g for g in feature_rows if g.get("game_pk") == r.game_pk), {})
            picks.append({
                "matchup": f"{r.away_team} @ {r.home_team}",
                "side": r.home_team if r.bet_side == "home" else r.away_team,
                "bet_side": r.bet_side,
                "game_pk": r.game_pk,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "odds": _american(r.american_odds),
                "american_odds_raw": r.american_odds,
                "home_odds": _american(_fr.get("home_american_odds")),
                "away_odds": _american(_fr.get("away_american_odds")),
                "model_prob": r.model_prob,
                "fair_prob": r.fair_prob,
                "edge": r.edge,
                "stake": r.stake,
                "ev_per_unit": r.ev_per_unit,
                "bookmaker": _fr.get("bookmaker", ""),
                "tier": "strong" if r.edge >= min_edge else "watchlist",
            })
        payload = {
            "error": None,
            "games_with_odds": len(odds_games),
            "strong": [p for p in picks if p["tier"] == "strong"],
            "watchlist": [p for p in picks if p["tier"] == "watchlist"],
            "cached": False,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        payload = {
            "error": _safe_error(exc),
            "games_with_odds": 0,
            "strong": [],
            "watchlist": [],
            "cached": False,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
        }

    LIVE_CACHE.update({
        "ts": now,
        "bankroll": bankroll,
        "min_edge": min_edge,
        "payload": payload,
    })
    return payload


def _fetch_live_scores(game_pks: list[int]) -> dict[int, dict]:
    """Fetch live/final game scores from the MLB Stats API."""
    valid = [p for p in game_pks if p]
    if not valid:
        return {}
    try:
        import requests as _req
        r = _req.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"gamePks": ",".join(str(p) for p in valid), "hydrate": "linescore"},
            headers={"User-Agent": "mlb-betting-model/0.1"},
            timeout=10,
        )
        r.raise_for_status()
        scores: dict[int, dict] = {}
        for date in r.json().get("dates", []):
            for game in date.get("games", []):
                pk  = game["gamePk"]
                st  = game.get("status", {})
                ls  = game.get("linescore", {})
                teams = ls.get("teams", {})
                scores[pk] = {
                    "status":         st.get("detailedState", "Scheduled"),
                    "abstract_state": st.get("abstractGameState", "Preview"),
                    "inning":         ls.get("currentInningOrdinal", ""),
                    "inning_half":    ls.get("inningHalf", ""),
                    "home_runs":      (teams.get("home") or {}).get("runs"),
                    "away_runs":      (teams.get("away") or {}).get("runs"),
                }
        return scores
    except Exception:
        return {}


def _live_bets_payload() -> dict:
    """Pending ML, F5, and prop bets enriched with live scores."""
    with _connect() as conn:
        ml_rows = conn.execute(
            "SELECT * FROM paper_bets WHERE outcome IS NULL ORDER BY game_date, game_pk"
        ).fetchall()
        f5_rows = conn.execute(
            "SELECT * FROM f5_paper_bets WHERE outcome IS NULL ORDER BY game_date, game_pk"
        ).fetchall()
        prop_rows = conn.execute(
            "SELECT * FROM prop_bets WHERE outcome IS NULL ORDER BY game_date, id"
        ).fetchall()

    bets = []

    for r in ml_rows:
        side_team = r["home_team"] if r["bet_side"] == "home" else r["away_team"]
        bets.append({
            "id":        r["id"],
            "bet_type":  "ML",
            "game_pk":   r["game_pk"],
            "game_date": r["game_date"],
            "matchup":   f"{r['away_team']} @ {r['home_team']}",
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "side":      side_team,
            "bet_side":  r["bet_side"],
            "odds":      _american(r["bet_american_odds"]),
            "edge":      float(r["edge"] or 0),
            "stake":     float(r["stake_dollars"] or 0),
            "bookmaker": r["bookmaker"] or "",
            "clv":       r["clv"],
            "home_close": _american(r["home_american_close"]),
            "away_close": _american(r["away_american_close"]),
        })

    for r in f5_rows:
        side_team = r["home_team"] if r["bet_side"] == "home" else r["away_team"]
        bets.append({
            "id":        r["id"],
            "bet_type":  "F5",
            "game_pk":   r["game_pk"],
            "game_date": r["game_date"],
            "matchup":   f"{r['away_team']} @ {r['home_team']}",
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "side":      side_team,
            "bet_side":  r["bet_side"],
            "odds":      _american(r["bet_american_odds"]),
            "edge":      float(r["edge"] or 0),
            "stake":     float(r["stake_dollars"] or 0),
            "bookmaker": r["bookmaker"] or "",
            "clv":       r["clv"],
            "home_close": _american(r["home_american_close"]),
            "away_close": _american(r["away_american_close"]),
        })

    # Build a lookup: (home_team, away_team, game_date) → game_pk
    # Priority: ML/F5 bets we already have, then games table, then MLB schedule API.
    game_pk_lookup: dict[tuple, int] = {}

    for b in bets:
        if b.get("game_pk") and b.get("home_team") and b.get("away_team"):
            game_pk_lookup[(b["home_team"], b["away_team"], b["game_date"])] = b["game_pk"]

    # Games table covers historical dates
    with _connect() as conn:
        for date in {r["game_date"] for r in prop_rows}:
            for g in conn.execute(
                "SELECT game_pk, home_team, away_team FROM games WHERE game_date=?", (date,)
            ).fetchall():
                game_pk_lookup.setdefault((g["home_team"], g["away_team"], date), g["game_pk"])

    # For any prop date still missing entries, fetch the MLB schedule API
    unresolved_dates = {
        r["game_date"] for r in prop_rows
        if not game_pk_lookup.get((r["team"] or "", r["opponent"] or "", r["game_date"]))
           and not game_pk_lookup.get((r["opponent"] or "", r["team"] or "", r["game_date"]))
    }
    for date in unresolved_dates:
        try:
            import requests as _req
            resp = _req.get(
                "https://statsapi.mlb.com/api/v1/schedule",
                params={"sportId": 1, "date": date, "hydrate": "team"},
                headers={"User-Agent": "mlb-betting-model/0.1"},
                timeout=8,
            )
            for d in resp.json().get("dates", []):
                for g in d.get("games", []):
                    home = g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
                    away = g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
                    pk   = g.get("gamePk")
                    if home and away and pk:
                        game_pk_lookup.setdefault((home, away, date), pk)
        except Exception:
            pass

    for r in prop_rows:
        mkt = r["market"] or ""
        mkt_label = mkt.replace("pitcher_", "").replace("batter_", "").replace("_", " ")
        side_label = f"{r['bet_side'].title()} {r['line']}"
        # team/opponent stored as home/away — try both orderings to be safe
        t, o, d = r["team"] or "", r["opponent"] or "", r["game_date"]
        game_pk = (game_pk_lookup.get((t, o, d))
                   or game_pk_lookup.get((o, t, d))
                   or 0)
        bets.append({
            "id":        r["id"],
            "bet_type":  "Prop",
            "game_pk":   game_pk,
            "game_date": r["game_date"],
            "matchup":   r["player_name"] or "",
            "side":      side_label,
            "bet_side":  r["bet_side"],
            "odds":      _american(r["american_odds"]),
            "edge":      float(r["edge"] or 0),
            "stake":     float(r["stake_dollars"] or 0),
            "bookmaker": r["bookmaker"] or "",
            "market":    mkt_label,
            "clv":       None,
            "home_close": "--",
            "away_close": "--",
        })

    # Fetch live scores only for ML/F5 bets with real game_pks
    real_pks = [b["game_pk"] for b in bets if b.get("game_pk")]
    scores = _fetch_live_scores(real_pks)
    for bet in bets:
        bet["score"] = scores.get(bet["game_pk"], {})

    # Current odds from cache for ML bets
    cached_payload = LIVE_CACHE.get("payload") or {}
    all_picks = cached_payload.get("strong", []) + cached_payload.get("watchlist", [])
    odds_by_pk = {p["game_pk"]: p for p in all_picks if p.get("game_pk")}
    for bet in bets:
        cur = odds_by_pk.get(bet["game_pk"], {})
        bet["current_home_odds"] = cur.get("home_odds", "--")
        bet["current_away_odds"] = cur.get("away_odds", "--")

    return {"bets": bets, "fetched_at": datetime.now().isoformat(timespec="seconds")}


def _suggestions(metrics: dict, freshness: dict, live: dict, logged_today: list[dict]) -> list[dict]:
    suggestions = []
    logged_sides = {row["matchup"]: row["side"] for row in logged_today}
    live_picks = live["strong"] + live["watchlist"]
    conflicts = [
        p for p in live_picks
        if p["matchup"] in logged_sides and p["side"] != logged_sides[p["matchup"]]
    ]
    if conflicts:
        suggestions.append({
            "title": "Review today's logged side",
            "body": "The current live model/market view differs from an already logged paper bet on the same matchup.",
            "priority": "High",
        })
    from datetime import date as _date, timedelta as _td
    _latest = freshness["latest_game_date"]
    _today  = freshness["today"]
    _stale  = _latest and (_date.fromisoformat(_today) - _date.fromisoformat(_latest)).days > 2
    if _stale:
        suggestions.append({
            "title": "Refresh 2026 results",
            "body": "Run the updater before picks so rolling team form uses the newest completed games.",
            "priority": "High",
        })
    if metrics["settled"] < 30:
        suggestions.append({
            "title": "Build a paper-trading sample",
            "body": f"Only {metrics['settled']} settled bets so far. Aim for 30–50 before trusting ROI.",
            "priority": "Medium",
        })
    if freshness["close_2025"] == 0:
        suggestions.append({
            "title": "Ingest 2025 closing odds",
            "body": "Adding 2025 closes gives the backtest a newer season and a better model check.",
            "priority": "Medium",
        })
    if not live["strong"] and live["watchlist"]:
        suggestions.append({
            "title": "Watchlist only today",
            "body": "The model sees marginal prices today, but none clear the strong-pick threshold.",
            "priority": "Medium",
        })
    suggestions.append({
        "title": "Next model upgrades",
        "body": "Add bullpen last-7-days, confirmed lineups, SP recent form, and automated CLV capture.",
        "priority": "Next",
    })
    return suggestions[:6]


def _logged_today_props(conn: sqlite3.Connection, today: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM prop_bets WHERE game_date = ? ORDER BY edge DESC",
        (today,),
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "game_pk":     r["game_pk"],
            "market":      r["market"],
            "player_name": r["player_name"],
            "line":        float(r["line"] or 0),
            "bet_side":    r["bet_side"],
            "odds":        _american(r["american_odds"]),
            "model_prob":  float(r["model_prob"] or 0),
            "fair_prob":   float(r["fair_prob"] or 0),
            "edge":        float(r["edge"] or 0),
            "stake":       float(r["stake_dollars"] or 0),
            "expected_k":  None,
            "bookmaker":   r["bookmaker"] or "",
            "status":      "Pending" if r["outcome"] is None else "Settled",
        })
    return out


def _live_prop_recommendations(bankroll: float, min_edge: float, refresh: bool) -> dict:
    now = time.time()
    cached = PROPS_CACHE["payload"]
    if (
        not refresh
        and cached is not None
        and now - float(PROPS_CACHE["ts"] or 0) < CACHE_SECONDS
        and PROPS_CACHE["bankroll"] == bankroll
        and PROPS_CACHE["min_edge"] == min_edge
    ):
        payload = dict(cached)
        payload["cached"] = True
        return payload

    try:
        from props.props_picks import fetch_props_odds, score_strikeout_props, find_prop_edges
        all_props  = fetch_props_odds(["pitcher_strikeouts"])
        scored     = score_strikeout_props(all_props)
        recs       = find_prop_edges(scored, bankroll=bankroll, min_edge=0.03)
        picks = []
        for r in recs:
            picks.append({
                "player_name":    r["player_name"],
                "matchup":        f"{r['away_team']} @ {r['home_team']}",
                "home_team":      r["home_team"],
                "away_team":      r["away_team"],
                "game_pk":        r.get("game_pk", 0),
                "market":         r["market"],
                "line":           r["line"],
                "bet_side":       r["bet_side"],
                "over_american":  r.get("over_american"),
                "under_american": r.get("under_american"),
                "american_odds":  r["american_odds"],
                "american_odds_raw": r["american_odds"],
                "model_prob":     r["model_prob"],
                "fair_prob":      r["fair_prob"],
                "edge":           r["edge"],
                "expected_k":     r.get("expected_k"),
                "expected_hits":  r.get("expected_hits"),
                "stake":          r["stake"],
                "bookmaker":      r.get("bookmaker", ""),
                "tier":           "strong" if r["edge"] >= min_edge else "watchlist",
            })
        payload = {
            "error": None,
            "strong":    [p for p in picks if p["tier"] == "strong"],
            "watchlist": [p for p in picks if p["tier"] == "watchlist"],
            "total_props": len(all_props),
            "scored":    len(scored),
            "cached":    False,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        payload = {
            "error":     _safe_error(exc),
            "strong": [], "watchlist": [],
            "total_props": 0, "scored": 0,
            "cached": False,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
        }

    PROPS_CACHE.update({"ts": now, "bankroll": bankroll, "min_edge": min_edge, "payload": payload})
    return payload


def dashboard_payload(bankroll: float, min_edge: float, refresh: bool) -> dict:
    with _connect() as conn:
        metrics = _paper_metrics(conn, bankroll)
        freshness = _data_freshness(conn)
        logged_today = _logged_today(conn, freshness["today"])
        logged_props_today = _logged_today_props(conn, freshness["today"])
    live = _live_recommendations(bankroll, min_edge, refresh)
    return {
        "settings": {
            "bankroll": bankroll,
            "min_edge": min_edge,
        },
        "metrics": metrics,
        "freshness": freshness,
        "logged_today": logged_today,
        "logged_props_today": logged_props_today,
        "live": live,
        "suggestions": _suggestions(metrics, freshness, live, logged_today),
        "display": {
            "current_bankroll": _money(metrics["current_bankroll"]),
            "available_bankroll": _money(metrics["available_bankroll"]),
            "pending_at_risk": _money(metrics["pending_at_risk"]),
            "total_profit": _money(metrics["total_profit"]),
            "roi": _pct(metrics["roi"], 2),
            "win_rate": f"{metrics['win_rate'] * 100:.1f}%",
            "mean_clv": _pct(metrics["mean_clv"], 2),
            "min_edge": f"{min_edge * 100:.0f}%",
        },
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MLB Betting</title>
  <style>
    :root {
      --bg:        #09090c;
      --surface:   #101014;
      --surface2:  #18181d;
      --surface3:  #222228;
      --border:    #27272e;
      --border2:   #35353f;
      --text:      #f0f0f5;
      --muted:     #6b7080;
      --subtle:    #3d3d4a;
      --gold:      #eab308;
      --gold2:     #fbbf24;
      --gold-dim:  rgba(234,179,8,.25);
      --gold-bg:   rgba(234,179,8,.1);
      --green:     #22c55e;
      --green-bg:  rgba(34,197,94,.12);
      --red:       #ef4444;
      --red-bg:    rgba(239,68,68,.12);
      --blue:      #3b82f6;
      --blue-bg:   rgba(59,130,246,.12);
      --amber:     #f59e0b;
      --amber-bg:  rgba(245,158,11,.12);
      --radius:    12px;
      --radius-sm: 8px;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.5;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }

    /* ── Header ── */
    header {
      position: sticky; top: 0; z-index: 200;
      background: rgba(9,9,12,.85);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
    }
    .hdr {
      max-width: 1440px; margin: 0 auto;
      padding: 0 24px; height: 60px;
      display: flex; align-items: center; justify-content: space-between; gap: 16px;
    }
    .hdr-left { display: flex; align-items: center; gap: 24px; }

    /* Brand */
    .brand { display: flex; align-items: center; gap: 10px; }
    .brand-icon {
      width: 32px; height: 32px; border-radius: var(--radius-sm);
      background: linear-gradient(135deg, var(--gold), var(--gold2));
      display: flex; align-items: center; justify-content: center;
      font-size: 15px; flex-shrink: 0;
      box-shadow: 0 0 0 1px var(--gold-dim), 0 2px 8px rgba(234,179,8,.2);
    }
    .brand-name { font-size: 15px; font-weight: 800; letter-spacing: -.4px; color: var(--text); }
    .brand-sub  { font-size: 10.5px; color: var(--muted); margin-top: 1px; letter-spacing: .1px; }

    /* Nav */
    .nav { display: flex; gap: 1px; background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; padding: 3px; }
    .nav-btn {
      height: 30px; padding: 0 13px; border: none; border-radius: 7px;
      background: transparent; color: var(--muted);
      font-size: 12.5px; font-weight: 600; font-family: inherit;
      cursor: pointer; display: flex; align-items: center; gap: 6px;
      transition: all .18s; white-space: nowrap; letter-spacing: -.1px;
    }
    .nav-btn:hover { color: var(--text); background: var(--surface3); }
    .nav-btn.active { background: var(--gold); color: #09090c; font-weight: 700; }
    .nav-badge {
      background: var(--red); color: #fff;
      border-radius: 10px; font-size: 10px; font-weight: 800;
      padding: 1px 5px; min-width: 16px; text-align: center; line-height: 1.5;
    }
    .nav-btn.active .nav-badge { background: rgba(0,0,0,.3); }

    /* Controls */
    .hdr-controls { display: flex; gap: 6px; align-items: center; }
    .ctl {
      height: 32px; border: 1px solid var(--border2); background: var(--surface2); color: var(--text);
      border-radius: var(--radius-sm); padding: 0 10px; font-size: 13px; font-family: inherit; outline: none;
      transition: border-color .15s;
    }
    .ctl:focus { border-color: var(--gold); box-shadow: 0 0 0 2px var(--gold-dim); }
    select.ctl { cursor: pointer; }
    .btn { cursor: pointer; font-weight: 700; display: flex; align-items: center; gap: 5px; }
    .btn-ghost { background: transparent; color: var(--muted); }
    .btn-ghost:hover { color: var(--text); }
    .btn-gold { background: var(--gold); color: #09090c; border-color: var(--gold); font-weight: 700; }
    .btn-gold:hover { background: var(--gold2); border-color: var(--gold2); }
    .refresh-ts { font-size: 11px; color: var(--subtle); white-space: nowrap; }

    /* ── Shell ── */
    .shell { max-width: 1440px; margin: 0 auto; padding: 20px 24px; }

    /* ── Status strip ── */
    .status-strip {
      background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
      padding: 9px 16px; font-size: 12px; color: var(--muted);
      display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 18px;
    }
    .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); display: inline-block; flex-shrink: 0; box-shadow: 0 0 6px var(--green); }
    .dot.stale { background: var(--amber); box-shadow: 0 0 6px var(--amber); }

    /* ── Two-col layout ── */
    .overview-grid { display: grid; grid-template-columns: minmax(0,1fr) 340px; gap: 18px; }
    .col { display: flex; flex-direction: column; gap: 16px; }

    /* ── Metrics row ── */
    .metrics { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }
    .m-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 18px 16px;
      position: relative; overflow: hidden;
      transition: border-color .2s;
    }
    .m-card:hover { border-color: var(--border2); }
    .m-card::before {
      content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
      background: var(--border2); border-radius: var(--radius) var(--radius) 0 0;
    }
    .m-card.pos-accent::before  { background: linear-gradient(90deg, var(--green), rgba(34,197,94,.3)); }
    .m-card.neg-accent::before  { background: linear-gradient(90deg, var(--red),   rgba(239,68,68,.3)); }
    .m-card.gold-accent::before { background: linear-gradient(90deg, var(--gold),  rgba(234,179,8,.3)); }
    .m-label { font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing: .8px; color: var(--muted); }
    .m-val { font-size: 28px; font-weight: 900; line-height: 1.1; margin-top: 10px; letter-spacing: -.8px; }
    .m-val.pos  { color: var(--green); }
    .m-val.neg  { color: var(--red); }
    .m-val.gold { color: var(--gold); }
    .m-sub { font-size: 11.5px; color: var(--muted); margin-top: 6px; }

    /* ── Card ── */
    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); overflow: hidden;
    }
    .card-hd {
      padding: 14px 18px 13px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
    }
    .card-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; color: var(--muted); }
    .card-body  { padding: 16px 18px; }

    /* ── Chips / pills ── */
    .chip {
      display: inline-flex; align-items: center; height: 22px; padding: 0 9px;
      border-radius: 6px; font-size: 11px; font-weight: 600; white-space: nowrap; gap: 3px;
      letter-spacing: -.1px;
    }
    .chip-default { background: var(--surface2); color: var(--muted); border: 1px solid var(--border2); }
    .chip-gold    { background: var(--gold-bg);  color: var(--gold);   border: 1px solid rgba(234,179,8,.2); }
    .chip-green   { background: var(--green-bg); color: var(--green);  border: 1px solid rgba(34,197,94,.2); }
    .chip-red     { background: var(--red-bg);   color: var(--red);    border: 1px solid rgba(239,68,68,.2); }
    .chip-amber   { background: var(--amber-bg); color: var(--amber);  border: 1px solid rgba(245,158,11,.2); }
    .chip-blue    { background: var(--blue-bg);  color: var(--blue);   border: 1px solid rgba(59,130,246,.2); }

    /* ── Pick cards (sportsbook tile) ── */
    .pick-list { display: flex; flex-direction: column; }
    .pick-tile {
      padding: 16px 18px;
      border-bottom: 1px solid var(--border);
      transition: background .15s;
    }
    .pick-tile:last-child { border-bottom: none; }
    .pick-tile:hover { background: rgba(255,255,255,.015); }

    /* Pick tile header */
    .pick-header {
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
      margin-bottom: 13px;
    }
    .pick-matchup {
      font-size: 14px; font-weight: 700; color: var(--text); letter-spacing: -.2px;
    }
    .pick-time {
      font-size: 11px; color: var(--muted); font-weight: 500; white-space: nowrap; flex-shrink: 0;
    }
    .pick-market {
      font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .8px;
      color: var(--subtle); background: var(--surface3); padding: 2px 7px;
      border-radius: 4px; border: 1px solid var(--border2); flex-shrink: 0;
    }

    /* Odds buttons */
    .odds-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }
    .odds-btn {
      background: var(--surface2); border: 1.5px solid var(--border2);
      border-radius: 10px; padding: 12px 14px; text-align: center;
      transition: all .15s; cursor: default;
    }
    .odds-btn.selected {
      background: linear-gradient(145deg, rgba(234,179,8,.12), rgba(234,179,8,.05));
      border-color: var(--gold);
      box-shadow: 0 0 0 1px var(--gold-dim) inset;
    }
    .odds-team  { font-size: 11.5px; color: var(--muted); margin-bottom: 5px; font-weight: 600; letter-spacing: .2px; }
    .odds-price { font-size: 22px; font-weight: 900; color: var(--text); letter-spacing: -.5px; line-height: 1; }
    .odds-btn.selected .odds-team  { color: rgba(234,179,8,.8); }
    .odds-btn.selected .odds-price { color: var(--gold); }

    /* Pick stat line */
    .pick-stat-line {
      display: flex; align-items: center; gap: 10px;
      font-size: 11.5px; color: var(--muted);
      margin-bottom: 12px; flex-wrap: wrap;
    }
    .pick-stat-line .edge-pos { color: var(--green); font-weight: 700; }
    .pick-stat-line .edge-neg { color: var(--red);   font-weight: 700; }
    .pick-stat-line b { color: var(--text); font-weight: 700; }
    .pick-stat-sep { color: var(--border2); }

    .pick-footer { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
    .pick-chips  { display: flex; gap: 5px; flex-wrap: wrap; align-items: center; }

    /* ── Stake input row ── */
    .stake-row {
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: 10px; padding: 10px 14px; margin-bottom: 12px;
    }
    .stake-rec { display: flex; flex-direction: column; gap: 2px; }
    .stake-rec-lbl { font-size: 9.5px; font-weight: 600; text-transform: uppercase; letter-spacing: .7px; color: var(--muted); }
    .stake-rec-amt { font-size: 15px; font-weight: 800; color: var(--gold); letter-spacing: -.3px; }
    .stake-input-wrap {
      display: flex; align-items: center; gap: 4px;
      background: var(--surface3); border: 1.5px solid var(--border2);
      border-radius: 8px; padding: 0 10px; height: 36px;
      transition: all .15s;
    }
    .stake-input-wrap:focus-within { border-color: var(--gold); box-shadow: 0 0 0 2px var(--gold-dim); }
    .stake-prefix { font-size: 14px; font-weight: 700; color: var(--muted); }
    .stake-input {
      width: 76px; background: transparent; border: none; outline: none;
      color: var(--text); font-size: 16px; font-weight: 800; font-family: inherit;
      text-align: right; letter-spacing: -.3px;
    }
    .stake-input::-webkit-inner-spin-button { opacity: .4; }

    .log-btn {
      height: 32px; padding: 0 16px; border: 1.5px solid var(--gold); border-radius: 8px;
      background: transparent; color: var(--gold);
      font-size: 12px; font-weight: 700; font-family: inherit; cursor: pointer;
      transition: all .18s; flex-shrink: 0; letter-spacing: -.1px;
    }
    .log-btn:hover:not(:disabled) {
      background: var(--gold); color: #09090c;
      box-shadow: 0 0 12px rgba(234,179,8,.25);
    }
    .log-btn:disabled { opacity: .3; cursor: default; }
    .logged-badge {
      display: inline-flex; align-items: center; height: 32px; padding: 0 14px;
      border-radius: 8px; font-size: 12px; font-weight: 700;
      background: var(--green-bg); color: var(--green);
      border: 1px solid rgba(34,197,94,.2); flex-shrink: 0;
    }

    /* ── Logged today (sidebar) ── */
    .logged-list { display: flex; flex-direction: column; }
    .logged-row {
      padding: 12px 18px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
      transition: background .12s;
    }
    .logged-row:hover { background: rgba(255,255,255,.015); }
    .logged-row:last-child { border-bottom: none; }
    .logged-team { font-size: 14px; font-weight: 800; color: var(--text); letter-spacing: -.2px; }
    .logged-meta { font-size: 11px; color: var(--muted); margin-top: 2px; }
    .logged-right { display: flex; flex-direction: column; align-items: flex-end; gap: 5px; flex-shrink: 0; }
    .logged-odds  { font-size: 18px; font-weight: 900; color: var(--gold); letter-spacing: -.3px; }

    /* ── Charts ── */
    .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    canvas { width: 100% !important; display: block; height: 180px; }

    /* ── Table ── */
    .tbl-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; min-width: 460px; }
    th, td { padding: 11px 18px; border-bottom: 1px solid var(--border); vertical-align: middle; text-align: left; }
    th { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .7px; color: var(--subtle); background: var(--surface2); }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: rgba(255,255,255,.02); }
    td:nth-child(5), th:nth-child(5), td:nth-child(6), th:nth-child(6),
    td:last-child,  th:last-child { text-align: right; }

    /* ── Freshness pills ── */
    .pill-row { display: flex; flex-wrap: wrap; gap: 6px; padding: 12px 16px; }

    /* ── Suggestions ── */
    .suggest-list { display: flex; flex-direction: column; }
    .suggest-row {
      padding: 11px 16px 11px 20px; border-bottom: 1px solid var(--border);
      position: relative;
    }
    .suggest-row:last-child { border-bottom: none; }
    .suggest-row::before {
      content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    }
    .suggest-row.high::before   { background: var(--red); }
    .suggest-row.medium::before { background: var(--amber); }
    .suggest-row.next::before   { background: var(--blue); }
    .suggest-title { font-size: 12px; font-weight: 700; color: var(--text); margin-bottom: 3px; }
    .suggest-body  { font-size: 11px; color: var(--muted); line-height: 1.4; }

    /* ── Live Bets tab ── */
    .live-section { display: flex; flex-direction: column; gap: 16px; }
    .live-hdr { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .live-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px,1fr)); gap: 14px; }
    /* ── Analytics tab ── */
    .analytics-grid { display: grid; grid-template-columns: 300px 1fr; gap: 18px; }
    .progress-wrap { padding: 16px 18px; }
    .progress-label { display: flex; justify-content: space-between; align-items: baseline;
      font-size: 12px; color: var(--muted); margin-bottom: 8px; }
    .progress-label b { color: var(--text); font-size: 22px; }
    .progress-track { background: var(--surface3); border-radius: 6px; height: 8px; overflow: hidden; }
    .progress-fill  { background: linear-gradient(90deg, var(--gold), var(--gold2));
      height: 100%; border-radius: 6px; transition: width .6s ease; }
    .progress-sub { font-size: 11px; color: var(--subtle); margin-top: 6px; }
    .auto-list { display: flex; flex-direction: column; gap: 0; }
    .auto-row { display: flex; align-items: center; justify-content: space-between;
      padding: 10px 18px; border-bottom: 1px solid var(--border); font-size: 13px; }
    .auto-row:last-child { border-bottom: none; }
    .auto-name { color: var(--text); font-weight: 600; }
    .auto-time { color: var(--muted); font-size: 12px; }
    .auto-dot  { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .auto-dot.ok    { background: var(--green);  box-shadow: 0 0 5px var(--green); }
    .auto-dot.err   { background: var(--red);    box-shadow: 0 0 5px var(--red); }
    .auto-dot.never { background: var(--subtle); }
    .bucket-tbl td, .bucket-tbl th { padding: 9px 14px; }
    .bucket-tbl th { font-size: 10px; text-transform: uppercase; letter-spacing: .7px;
      color: var(--muted); font-weight: 700; border-bottom: 1px solid var(--border); }
    @media(max-width:900px){ .analytics-grid { grid-template-columns: 1fr; } }

    .live-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; overflow: hidden;
    }
    .live-card-hdr {
      padding: 11px 16px; background: var(--surface2); border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
    }
    .live-matchup-txt { font-size: 12px; color: var(--muted); font-weight: 500; }
    .live-card-body   { padding: 14px 16px; }
    .live-main-row    {
      display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 12px;
    }
    .live-bet-lbl  { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .5px; color: var(--subtle); margin-bottom: 4px; }
    .live-bet-team { font-size: 22px; font-weight: 900; color: var(--text); letter-spacing: -.3px; }
    .live-score-box { text-align: right; flex-shrink: 0; }
    .live-score { font-size: 28px; font-weight: 900; letter-spacing: -.5px; line-height: 1; color: var(--subtle); }
    .live-score.live  { color: var(--green); }
    .live-score.final { color: var(--muted); }
    .live-inning { font-size: 11px; color: var(--muted); margin-top: 3px; text-align: right; }
    .live-inning.live { color: var(--green); font-weight: 700; }
    .live-chips  { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px; }
    .live-odds-row {
      font-size: 11px; color: var(--muted); padding-top: 10px;
      border-top: 1px solid var(--border); display: flex; flex-wrap: wrap; gap: 12px;
    }
    .live-odds-row b { color: var(--text); }
    .live-empty {
      text-align: center; padding: 60px 20px; color: var(--subtle);
      font-size: 14px; grid-column: 1 / -1;
    }

    /* ── Utility ── */
    .pos { color: var(--green); }
    .neg { color: var(--red);   }
    .empty-state {
      padding: 28px 16px; text-align: center; color: var(--subtle); font-size: 13px;
    }
    .loading { opacity: .35; pointer-events: none; }

    /* ── Bet History tab ── */
    .history-section { display: flex; flex-direction: column; gap: 18px; }
    .history-toolbar {
      display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
    }
    .history-summary {
      display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
    }
    .filter-row { display: flex; gap: 6px; flex-wrap: wrap; }
    .filter-btn {
      height: 30px; padding: 0 12px; border: 1px solid var(--border2);
      background: var(--surface2); color: var(--muted);
      border-radius: 6px; font-size: 12px; font-weight: 600; font-family: inherit;
      cursor: pointer; transition: all .15s;
    }
    .filter-btn:hover { color: var(--text); border-color: var(--border2); }
    .filter-btn.active { background: var(--gold); color: #0b0d11; border-color: var(--gold); }

    /* Slip grid */
    .slip-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 12px;
    }

    /* Individual bet slip */
    .slip {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; overflow: hidden; display: flex; flex-direction: column;
    }
    .slip.won  { border-color: rgba(39,196,122,.35); }
    .slip.lost { border-color: rgba(232,80,58,.25);  }
    .slip.push { border-color: rgba(240,165,0,.25);  }

    .slip-hdr {
      padding: 10px 14px; background: var(--surface2);
      border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; gap: 8px;
    }
    .slip-matchup { font-size: 12px; color: var(--muted); font-weight: 500; }
    .slip-date    { font-size: 11px; color: var(--subtle); }

    .slip-odds-row {
      display: grid; grid-template-columns: 1fr 1fr; gap: 1px;
      background: var(--border); border-bottom: 1px solid var(--border);
    }
    .slip-team-btn {
      background: var(--surface); padding: 10px 14px; text-align: center;
    }
    .slip-team-btn.selected { background: var(--gold-bg); }
    .slip-team-name  { font-size: 10px; color: var(--muted); margin-bottom: 3px; font-weight: 500; }
    .slip-team-odds  { font-size: 17px; font-weight: 900; color: var(--text); letter-spacing: -.2px; }
    .slip-team-btn.selected .slip-team-name { color: var(--gold); }
    .slip-team-btn.selected .slip-team-odds { color: var(--gold2); }

    .slip-body { padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; flex: 1; }

    .slip-stats {
      display: flex; flex-wrap: wrap; gap: 5px;
    }

    .slip-result-row {
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
      padding-top: 8px; border-top: 1px solid var(--border);
      margin-top: auto;
    }
    .slip-outcome { font-size: 13px; font-weight: 800; }
    .slip-outcome.won  { color: var(--green); }
    .slip-outcome.lost { color: var(--red);   }
    .slip-outcome.push { color: var(--amber); }
    .slip-outcome.pending { color: var(--subtle); }
    .slip-pnl { font-size: 15px; font-weight: 900; letter-spacing: -.2px; }
    .slip-pnl.pos { color: var(--green); }
    .slip-pnl.neg { color: var(--red);   }

    /* ── Sub-nav (Bet History) ── */
    .sub-nav {
      display: flex; gap: 3px; background: var(--surface2);
      border: 1px solid var(--border); border-radius: 10px; padding: 3px;
    }
    .sub-btn {
      height: 30px; padding: 0 16px; border: none; border-radius: 7px;
      background: transparent; color: var(--muted);
      font-size: 12.5px; font-weight: 600; font-family: inherit;
      cursor: pointer; display: flex; align-items: center; gap: 6px;
      transition: all .18s;
    }
    .sub-btn:hover { color: var(--text); background: var(--surface3); }
    .sub-btn.active { background: var(--surface3); color: var(--text); font-weight: 700; }
    .sub-btn.active.live { background: rgba(34,197,94,.15); color: var(--green); }
    .live-dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: var(--green); box-shadow: 0 0 5px var(--green);
      display: inline-block; flex-shrink: 0;
    }

    /* ── Finished bet rows ── */
    .finished-list { display: flex; flex-direction: column; }
    .finished-row {
      display: grid;
      grid-template-columns: 80px 1fr auto auto auto auto;
      align-items: center; gap: 12px;
      padding: 12px 18px; border-bottom: 1px solid var(--border);
      transition: background .12s;
    }
    .finished-row:hover { background: rgba(255,255,255,.018); }
    .finished-row:last-child { border-bottom: none; }
    .finished-row.won  { border-left: 2px solid var(--green); }
    .finished-row.lost { border-left: 2px solid var(--red); }
    .finished-row.push { border-left: 2px solid var(--amber); }
    .finished-row.pending { border-left: 2px solid var(--subtle); }
    .fr-date { font-size: 11px; color: var(--muted); white-space: nowrap; }
    .fr-main { min-width: 0; }
    .fr-bet  { font-size: 14px; font-weight: 800; color: var(--text); letter-spacing: -.2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .fr-sub  { font-size: 11px; color: var(--muted); margin-top: 2px; }
    .fr-odds { font-size: 15px; font-weight: 800; color: var(--gold); letter-spacing: -.2px; text-align: right; white-space: nowrap; }
    .fr-stake { font-size: 13px; color: var(--muted); text-align: right; white-space: nowrap; }
    .fr-result { font-size: 12px; font-weight: 700; text-align: right; white-space: nowrap; }
    .fr-result.won  { color: var(--green); }
    .fr-result.lost { color: var(--red); }
    .fr-result.push { color: var(--amber); }
    .fr-result.pending { color: var(--subtle); }
    .fr-pnl { font-size: 15px; font-weight: 900; letter-spacing: -.2px; text-align: right; white-space: nowrap; min-width: 64px; }
    .fr-pnl.pos { color: var(--green); }
    .fr-pnl.neg { color: var(--red); }
    @media (max-width: 700px) {
      .finished-row { grid-template-columns: 70px 1fr auto auto; }
      .fr-stake, .fr-odds { display: none; }
    }

    /* ── Responsive ── */
    @media (max-width: 1080px) { .overview-grid { grid-template-columns: 1fr; } }
    @media (max-width: 700px) {
      .metrics { grid-template-columns: 1fr 1fr; }
      .charts-grid { grid-template-columns: 1fr; }
      th:nth-child(1), td:nth-child(1) { display: none; }
    }
    @media (max-width: 520px) {
      .hdr { height: auto; padding: 12px 16px; flex-wrap: wrap; }
      .hdr-controls .refresh-ts { display: none; }
    }
  </style>
</head>
<body>

<header>
  <div class="hdr">
    <div class="hdr-left">
      <div class="brand">
        <div class="brand-icon">⚾</div>
        <div>
          <div class="brand-name">MLB Betting</div>
          <div class="brand-sub" id="subtitle">Loading…</div>
        </div>
      </div>
      <div class="nav">
        <button class="nav-btn active" id="tabOverview"   onclick="switchTab('overview')">Overview</button>
        <button class="nav-btn"        id="tabMoneyline" onclick="switchTab('moneyline')">Moneyline</button>
        <button class="nav-btn"        id="tabProps"     onclick="switchTab('props')">Props</button>
        <button class="nav-btn"        id="tabHistory"   onclick="switchTab('history')">
          Bet History&nbsp;<span class="nav-badge" id="pendingBadge">0</span>
        </button>
      </div>
    </div>
    <div class="hdr-controls">
      <input class="ctl" id="bankrollInput" type="number" min="1" step="10"
             value="__DEFAULT_BANKROLL__" aria-label="Bankroll" placeholder="Bankroll">
      <select class="ctl" id="edgeSelect" aria-label="Min edge">
        <option value="0.10" __EDGE_010__>10% edge</option>
        <option value="0.08" __EDGE_008__>8% edge</option>
        <option value="0.05" __EDGE_005__>5% edge</option>
        <option value="0.03" __EDGE_003__>3% edge</option>
      </select>
      <button class="ctl btn btn-ghost" id="refreshBtn">Refresh</button>
      <button class="ctl btn btn-gold" id="refreshLiveBtn">↻ Live Odds</button>
      <span class="refresh-ts" id="refreshTs"></span>
    </div>
  </div>
</header>

<!-- ── Overview Tab ── -->
<div id="tab-overview">
  <div class="shell">
    <div class="status-strip" id="statusStrip">
      <span style="display:flex;align-items:center;gap:7px">
        <span class="dot" id="freshDot"></span>
        <span id="freshLabel">--</span>
      </span>
      <span id="statusExtra"></span>
    </div>

    <div id="app" class="overview-grid">
      <!-- Left col -->
      <div class="col">
        <div class="metrics">
          <div class="m-card gold-accent">
            <div class="m-label">Bankroll</div>
            <div class="m-val gold" id="mBankroll">--</div>
            <div class="m-sub" id="mAvail">--</div>
          </div>
          <div class="m-card" id="mcRoi">
            <div class="m-label">ROI</div>
            <div class="m-val" id="mRoi">--</div>
            <div class="m-sub" id="mProfit">--</div>
          </div>
          <div class="m-card" id="mcRecord">
            <div class="m-label">Record</div>
            <div class="m-val" id="mRecord">--</div>
            <div class="m-sub" id="mRecordSub">--</div>
          </div>
          <div class="m-card" id="mcClv">
            <div class="m-label">Avg CLV</div>
            <div class="m-val" id="mClv">--</div>
            <div class="m-sub" id="mClvSub">--</div>
          </div>
        </div>

        <div class="charts-grid">
          <div class="card">
            <div class="card-hd"><span class="card-title">Bankroll</span></div>
            <div class="card-body" style="padding-top:6px"><canvas id="bankrollChart"></canvas></div>
          </div>
          <div class="card">
            <div class="card-hd"><span class="card-title">Cumulative ROI</span></div>
            <div class="card-body" style="padding-top:6px"><canvas id="roiChart"></canvas></div>
          </div>
        </div>

        <div class="card">
          <div class="card-hd">
            <span class="card-title">Recent Bets</span>
            <span class="chip chip-default" id="pendingPill">--</span>
          </div>
          <div class="tbl-wrap">
            <table>
              <thead><tr>
                <th>Date</th><th>Matchup</th><th>Side</th>
                <th>Odds</th><th>Edge</th><th>Stake</th><th>CLV</th><th>Result</th>
              </tr></thead>
              <tbody id="recentBets"></tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Right col -->
      <div class="col">
        <div class="card">
          <div class="card-hd">
            <span class="card-title">Data Freshness</span>
            <span class="chip chip-default" id="freshPill">--</span>
          </div>
          <div class="pill-row" id="freshPills"></div>
        </div>
        <div class="card">
          <div class="card-hd"><span class="card-title">Actions</span></div>
          <div class="suggest-list" id="suggestions"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ── Moneyline Tab ── -->
<div id="tab-moneyline" style="display:none">
  <div class="shell" style="padding-top:18px">
    <div class="overview-grid">
      <!-- Left col: picks -->
      <div class="col">
        <div class="card">
          <div class="card-hd">
            <span class="card-title">Best Picks Today</span>
            <span class="chip chip-default" id="liveStatus">--</span>
          </div>
          <div id="strongPicks"></div>
        </div>
        <div class="card">
          <div class="card-hd">
            <span class="card-title">Watchlist</span>
            <span class="chip chip-default">3–<span id="watchlistThresh">--</span>% edge</span>
          </div>
          <div id="watchlist"></div>
        </div>
      </div>
      <!-- Right col: what you've logged today -->
      <div class="col">
        <div class="card">
          <div class="card-hd">
            <span class="card-title">Logged Today</span>
            <span class="chip chip-default" id="loggedPill">0</span>
          </div>
          <div id="loggedToday"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ── Props Tab ── -->
<div id="tab-props" style="display:none">
  <div class="shell" style="padding-top:18px">
    <div class="overview-grid">
      <!-- Left col: prop picks -->
      <div class="col">
        <div class="card">
          <div class="card-hd">
            <span class="card-title">K Model — Strong Picks</span>
            <span class="chip chip-default" id="propsStatus">--</span>
          </div>
          <div id="propStrongPicks"></div>
        </div>
        <div class="card">
          <div class="card-hd">
            <span class="card-title">Watchlist</span>
            <span class="chip chip-default">5–<span id="propsWatchlistThresh">10</span>% edge</span>
          </div>
          <div id="propWatchlist"></div>
        </div>
      </div>
      <!-- Right col -->
      <div class="col">
        <div class="card">
          <div class="card-hd">
            <span class="card-title">Logged Props Today</span>
            <span class="chip chip-default" id="propLoggedPill">0</span>
          </div>
          <div id="propLoggedToday"></div>
        </div>
        <div class="card">
          <div class="card-hd"><span class="card-title">Model Info</span></div>
          <div class="card-body" style="font-size:12px;color:var(--muted);line-height:1.6">
            <b style="color:var(--text)">Pitcher Strikeout Model</b><br>
            Ridge regression · Walk-forward MAE 1.78 Ks<br>
            Over/under accuracy: ~65% @5.5 line · ~75% @6.5 line<br>
            Features: K/9 (season + L5), K/pitch, IP/start, pitches/start, opp K/9<br>
            Training data: 22,000 starts · 2021–2026
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ── Bet History Tab ── -->
<div id="tab-history" style="display:none">
  <div class="shell" style="padding-top:18px">

    <!-- Sub-nav -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap;gap:12px">
      <div class="sub-nav">
        <button class="sub-btn active" id="subUpcoming" onclick="switchHistory('upcoming')">
          Upcoming&nbsp;<span class="nav-badge" id="upcomingBadge" style="background:var(--subtle)">0</span>
        </button>
        <button class="sub-btn" id="subLive" onclick="switchHistory('live')">
          <span class="live-dot"></span>Live&nbsp;<span class="nav-badge" id="liveBadge" style="background:var(--green)">0</span>
        </button>
        <button class="sub-btn" id="subFinished" onclick="switchHistory('finished')">Finished</button>
      </div>
      <button class="ctl btn btn-ghost" id="histRefreshBtn" onclick="loadBetHistory()">Refresh</button>
    </div>

    <!-- Upcoming -->
    <div id="hist-upcoming">
      <div id="upcomingGrid" class="live-grid"></div>
    </div>

    <!-- Live -->
    <div id="hist-live" style="display:none">
      <div id="histLiveGrid" class="live-grid"></div>
    </div>

    <!-- Finished -->
    <div id="hist-finished" style="display:none">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:14px">
        <div class="history-summary" id="historySummary"></div>
        <div class="filter-row">
          <button class="filter-btn active" onclick="setFilter('all',this)">All</button>
          <button class="filter-btn" onclick="setFilter('Won',this)">Won</button>
          <button class="filter-btn" onclick="setFilter('Lost',this)">Lost</button>
          <button class="filter-btn" onclick="setFilter('ML',this)">ML</button>
          <button class="filter-btn" onclick="setFilter('F5',this)">F5</button>
          <button class="filter-btn" onclick="setFilter('Prop',this)">Props</button>
        </div>
      </div>
      <div class="card">
        <div id="finishedList" class="finished-list"></div>
      </div>
    </div>

  </div>
</div>

<!-- ── Analytics Tab ── -->
<div id="tab-analytics" style="display:none">
  <div class="shell" style="padding-top:18px">
    <div class="analytics-grid">
      <!-- Left col -->
      <div class="col">
        <div class="card">
          <div class="card-hd"><span class="card-title">Sample Progress</span></div>
          <div class="progress-wrap" id="sampleProgressWrap">
            <div class="progress-label">
              <span>Settled bets toward real money</span>
              <b id="sampleCount">--</b>
            </div>
            <div class="progress-track">
              <div class="progress-fill" id="sampleFill" style="width:0%"></div>
            </div>
            <div class="progress-sub" id="sampleSub"></div>
          </div>
        </div>
        <div class="card">
          <div class="card-hd"><span class="card-title">Automation Status</span></div>
          <div class="auto-list" id="autoStatus"></div>
        </div>
        <div class="card">
          <div class="card-hd"><span class="card-title">ROI by Edge Bucket</span></div>
          <div class="tbl-wrap">
            <table class="bucket-tbl">
              <thead><tr><th>Edge</th><th>Bets</th><th>W–L</th><th>ROI</th></tr></thead>
              <tbody id="edgeBuckets"></tbody>
            </table>
          </div>
        </div>
      </div>
      <!-- Right col -->
      <div class="col">
        <div class="card">
          <div class="card-hd"><span class="card-title">CLV Per Bet</span><span class="chip chip-default" id="clvChartPill"></span></div>
          <div class="card-body" style="padding-top:6px"><canvas id="clvChart" style="height:180px"></canvas></div>
        </div>
        <div class="card">
          <div class="card-hd"><span class="card-title">Cumulative P&amp;L by Week</span></div>
          <div class="card-body" style="padding-top:6px"><canvas id="weeklyPnlChart" style="height:180px"></canvas></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
/* ─── Helpers ─── */
const E = v => String(v ?? '').replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const $ = id => document.getElementById(id);
const money = v => {
  const n = Number(v || 0);
  return (n < 0 ? '-' : '') + '$' + Math.abs(n).toLocaleString(undefined,
    {minimumFractionDigits:2, maximumFractionDigits:2});
};
const pct = (v, signed = true) => {
  const n = Number(v || 0) * 100;
  return (signed && n > 0 ? '+' : '') + n.toFixed(1) + '%';
};
const posNeg = v => Number(v || 0) >= 0 ? 'pos' : 'neg';
const edgeChipCls = v => Number(v || 0) >= 0 ? 'chip chip-green' : 'chip chip-red';
const emptyState = txt => `<div class="empty-state">${E(txt)}</div>`;

/* ─── Tab switching ─── */
let _activeTab = 'overview', _liveTimer = null, _activeHistSub = 'upcoming';

function switchTab(tab) {
  _activeTab = tab;
  $('tab-overview').style.display   = tab === 'overview'   ? '' : 'none';
  $('tab-moneyline').style.display  = tab === 'moneyline'  ? '' : 'none';
  $('tab-props').style.display      = tab === 'props'      ? '' : 'none';
  $('tab-history').style.display    = tab === 'history'    ? '' : 'none';
  $('tabOverview').classList.toggle('active',   tab === 'overview');
  $('tabMoneyline').classList.toggle('active',  tab === 'moneyline');
  $('tabProps').classList.toggle('active',      tab === 'props');
  $('tabHistory').classList.toggle('active',    tab === 'history');
  if (tab === 'history')  { loadBetHistory(); if (!_liveTimer) _liveTimer = setInterval(loadBetHistory, 60000); }
  else                    { if (_liveTimer) { clearInterval(_liveTimer); _liveTimer = null; } }
  if (tab === 'props')    loadProps(false);
  if (tab === 'moneyline' && !$('strongPicks').innerHTML) load(false);
}

function switchHistory(sub) {
  _activeHistSub = sub;
  $('hist-upcoming').style.display = sub === 'upcoming' ? '' : 'none';
  $('hist-live').style.display     = sub === 'live'     ? '' : 'none';
  $('hist-finished').style.display = sub === 'finished' ? '' : 'none';
  $('subUpcoming').classList.toggle('active', sub === 'upcoming');
  $('subLive').classList.toggle('active',     sub === 'live');
  $('subLive').classList.toggle('live',       sub === 'live');
  $('subFinished').classList.toggle('active', sub === 'finished');
}

/* ─── Prop tiles ─── */
let _currentProps = [], _loggedPropKeys = new Set();

function propTile(p, idx) {
  const key      = `${p.player_name}|${p.market}|${p.line}|${p.bet_side}`;
  const isLogged = _loggedPropKeys.has(key);
  const isOver   = p.bet_side === 'over';
  const bkChip   = p.bookmaker ? `<span class="chip chip-default" style="font-size:10px">${E(p.bookmaker)}</span>` : '';
  const expChip  = p.expected_k != null
    ? `<span class="chip chip-blue">Exp ${Number(p.expected_k).toFixed(1)} K</span>`
    : p.expected_hits != null
      ? `<span class="chip chip-blue">Exp ${Number(p.expected_hits).toFixed(2)} H</span>`
      : '';

  const edgeCls  = Number(p.edge || 0) >= 0 ? 'edge-pos' : 'edge-neg';
  const mktLabel = (p.market || '').replace('pitcher_','').replace('batter_','').replace('_',' ');
  const propUnit = p.market === 'pitcher_strikeouts' ? 'K'
                 : p.market === 'batter_hits'        ? 'H'
                 : mktLabel;
  const expStr   = p.expected_k != null
    ? `Exp <b>${Number(p.expected_k).toFixed(1)}K</b>`
    : p.expected_hits != null
      ? `Exp <b>${Number(p.expected_hits).toFixed(2)}H</b>`
      : '';
  const bk = p.bookmaker ? ` <span class="pick-stat-sep">·</span> ${E(p.bookmaker)}` : '';

  const stakeRow = isLogged ? '' : `
    <div class="stake-row">
      <div class="stake-rec">
        <span class="stake-rec-lbl">Kelly rec.</span>
        <span class="stake-rec-amt">${money(p.stake)}</span>
      </div>
      <div class="stake-input-wrap">
        <span class="stake-prefix">$</span>
        <input type="number" class="stake-input" id="pstake${idx}"
               value="${Number(p.stake).toFixed(2)}" min="0.01" step="0.50"
               aria-label="Your stake">
      </div>
    </div>`;

  const logArea = isLogged
    ? `<span class="logged-badge">&#10003; Logged</span>`
    : `<button class="log-btn" id="plb${idx}" onclick="logPropBet(${idx})">Log Bet</button>`;

  return `<div class="pick-tile">
    <div class="pick-header">
      <div>
        <div class="pick-matchup">${E(p.player_name)}</div>
        <div style="font-size:11px;color:var(--muted);margin-top:2px">${E(p.matchup)}</div>
      </div>
      <span class="pick-market">${E(mktLabel)}</span>
    </div>
    <div class="odds-grid">
      <div class="odds-btn ${!isOver ? 'selected' : ''}">
        <div class="odds-team">Under ${E(String(p.line))} ${E(propUnit)}</div>
        <div class="odds-price">${p.under_american != null ? (p.under_american > 0 ? '+' : '') + p.under_american : '--'}</div>
      </div>
      <div class="odds-btn ${isOver ? 'selected' : ''}">
        <div class="odds-team">Over ${E(String(p.line))} ${E(propUnit)}</div>
        <div class="odds-price">${p.over_american != null ? (p.over_american > 0 ? '+' : '') + p.over_american : '--'}</div>
      </div>
    </div>
    <div class="pick-stat-line">
      ${expStr ? expStr + ' <span class="pick-stat-sep">·</span>' : ''}
      <b>${Math.round((p.model_prob||0)*100)}%</b> model
      <span class="pick-stat-sep">·</span>
      <span class="${edgeCls}">${pct(p.edge)} edge</span>${bk}
    </div>
    ${stakeRow}
    <div class="pick-footer">
      <div></div>
      ${logArea}
    </div>
  </div>`;
}

async function logPropBet(idx) {
  const p = _currentProps[idx];
  if (!p) return;
  const stakeInput = $(`pstake${idx}`);
  const userStake  = stakeInput ? parseFloat(stakeInput.value) : p.stake;
  if (!userStake || userStake <= 0) { alert('Enter a valid stake amount.'); return; }
  const btn = $(`plb${idx}`);
  if (btn) { btn.disabled = true; btn.textContent = 'Logging…'; }
  try {
    const res = await fetch('/api/log_prop_bet', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        game_pk: p.game_pk, home_team: p.home_team, away_team: p.away_team,
        player_name: p.player_name, market: p.market, line: p.line,
        bet_side: p.bet_side, american_odds_raw: p.american_odds_raw,
        model_prob: p.model_prob, fair_prob: p.fair_prob,
        edge: p.edge, expected_k: p.expected_k,
        stake: userStake, bookmaker: p.bookmaker,
        bankroll: Number($('bankrollInput').value) || DEFAULT_BANKROLL,
      }),
    });
    const result = await res.json();
    if (result.ok) {
      loadProps(false);
    } else {
      if (btn) { btn.disabled = false; btn.textContent = 'Log Bet'; }
      alert('Could not log: ' + result.error);
    }
  } catch (err) {
    if (btn) { btn.disabled = false; btn.textContent = 'Log Bet'; }
    alert('Error: ' + err.message);
  }
}

async function loadProps(forceRefresh = false) {
  const bankroll = Number($('bankrollInput').value) || DEFAULT_BANKROLL;
  const edge     = Number($('edgeSelect').value)    || 0.10;
  const url = `/api/props?bankroll=${bankroll}&min_edge=${edge}&refresh=${forceRefresh?1:0}`;
  let data, loggedData;
  try {
    const [res, lRes] = await Promise.all([fetch(url), fetch('/api/logged_props_today')]);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
    loggedData = lRes.ok ? await lRes.json() : null;
  } catch (err) {
    $('propStrongPicks').innerHTML = emptyState('Error loading props: ' + err.message);
    return;
  }

  if (loggedData && loggedData.props) {
    _loggedPropsToday = loggedData.props;
    $('propLoggedPill').textContent = _loggedPropsToday.length;
    $('propLoggedToday').innerHTML = _loggedPropsToday.length
      ? `<div class="logged-list">${_loggedPropsToday.map(p => {
          const edgeCls = Number(p.edge||0)>=0?'pos':'neg';
          const stCls = p.status==='Pending'?'chip chip-amber':'chip chip-default';
          return `<div class="logged-row">
            <div>
              <div class="logged-team">${E(p.player_name)}</div>
              <div class="logged-meta">${E(p.bet_side)} ${E(String(p.line))} &middot;
                <span class="${edgeCls}">${pct(p.edge)} edge</span> &middot; ${money(p.stake)}
              </div>
            </div>
            <div class="logged-right">
              <span class="logged-odds">${E(p.odds)}</span>
              <span class="${stCls}">${E(p.status)}</span>
            </div>
          </div>`;
        }).join('')}</div>`
      : emptyState('No props logged today.');
  }

  const {strong, watchlist, total_props, scored, error, fetched_at} = data;
  $('propsStatus').textContent = error ? 'Error' : `${scored}/${total_props} scored${data.cached?' · cached':''}`;
  $('propsStatus').className   = `chip ${error ? 'chip-red' : 'chip-default'}`;
  $('propsWatchlistThresh').textContent = Math.round(edge * 100);

  _loggedPropKeys = new Set(
    (_loggedPropsToday || []).map(p=>`${p.player_name}|${p.market}|${p.line}|${p.bet_side}`)
  );
  _currentProps = [...(strong||[]), ...(watchlist||[])];

  $('propStrongPicks').innerHTML = error
    ? emptyState(error)
    : strong.length
      ? `<div class="pick-list">${strong.map((p,i)=>propTile(p,i)).join('')}</div>`
      : emptyState('No strong prop picks at current edge threshold.');

  const off = (strong||[]).length;
  $('propWatchlist').innerHTML = watchlist.length
    ? `<div class="pick-list">${watchlist.map((p,i)=>propTile(p,off+i)).join('')}</div>`
    : emptyState('No watchlist props above 5% edge right now.');
}

/* ─── Pick tiles ─── */
let _currentPicks = [], _loggedKeys = new Set();

function fmtGameTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString('en-US', {hour: 'numeric', minute: '2-digit', timeZoneName: 'short'});
  } catch(e) { return ''; }
}

function pickTile(p, idx) {
  const key      = p.game_pk ? `${p.game_pk}|${p.bet_side}` : null;
  const isLogged = key ? _loggedKeys.has(key) : false;
  const isHome   = p.bet_side === 'home';
  const timeStr  = fmtGameTime(p.commence_time);
  const edgeCls  = Number(p.edge || 0) >= 0 ? 'edge-pos' : 'edge-neg';
  const bk       = p.bookmaker ? ` <span class="pick-stat-sep">·</span> ${E(p.bookmaker)}` : '';
  const rawOdds  = Number(p.american_odds_raw || 0);
  const highOdds = rawOdds > 300;
  const oddsWarn = highOdds
    ? `<span class="chip chip-amber" style="font-size:10px" title="Big underdog — model edge may be an artifact">+${rawOdds} ⚠</span>`
    : '';

  const stakeRow = isLogged ? '' : `
    <div class="stake-row">
      <div class="stake-rec">
        <span class="stake-rec-lbl">Kelly rec.</span>
        <span class="stake-rec-amt">${money(p.stake)}</span>
      </div>
      <div class="stake-input-wrap">
        <span class="stake-prefix">$</span>
        <input type="number" class="stake-input" id="stake${idx}"
               value="${Number(p.stake).toFixed(2)}" min="0.01" step="0.50"
               aria-label="Your stake">
      </div>
    </div>`;

  const logArea = isLogged
    ? `<span class="logged-badge">&#10003; Logged</span>`
    : `<button class="log-btn" id="lb${idx}" onclick="logBet(${idx})">Log Bet</button>`;

  return `<div class="pick-tile">
    <div class="pick-header">
      <span class="pick-matchup">${E(p.matchup)}</span>
      ${timeStr ? `<span class="pick-time">${E(timeStr)}</span>` : ''}
    </div>
    <div class="odds-grid">
      <div class="odds-btn ${!isHome ? 'selected' : ''}">
        <div class="odds-team">${E(p.away_team || '')}</div>
        <div class="odds-price">${E(p.away_odds || '--')}</div>
      </div>
      <div class="odds-btn ${isHome ? 'selected' : ''}">
        <div class="odds-team">${E(p.home_team || '')}</div>
        <div class="odds-price">${E(p.home_odds || '--')}</div>
      </div>
    </div>
    <div class="pick-stat-line">
      <b>${Math.round((p.model_prob||0)*100)}%</b> model
      <span class="pick-stat-sep">·</span>
      <span class="${edgeCls}">${pct(p.edge)} edge</span>${bk}
    </div>
    ${oddsWarn ? `<div style="padding:0 14px 8px">${oddsWarn}</div>` : ''}
    ${stakeRow}
    <div class="pick-footer">
      <div></div>
      ${logArea}
    </div>
  </div>`;
}

async function logBet(idx) {
  const p = _currentPicks[idx];
  if (!p) return;
  if (!p.game_pk) { alert('Cannot log: no game ID available for this pick.'); return; }
  const stakeInput = $(`stake${idx}`);
  const userStake  = stakeInput ? parseFloat(stakeInput.value) : p.stake;
  if (!userStake || userStake <= 0) { alert('Enter a valid stake amount greater than $0.'); return; }
  const btn = $(`lb${idx}`);
  if (btn) { btn.disabled = true; btn.textContent = 'Logging…'; }
  try {
    const res = await fetch('/api/log_bet', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        game_pk: p.game_pk, home_team: p.home_team, away_team: p.away_team,
        bet_side: p.bet_side, american_odds_raw: p.american_odds_raw,
        model_prob: p.model_prob, fair_prob: p.fair_prob,
        edge: p.edge, stake: userStake, bookmaker: p.bookmaker,
        bankroll: Number($('bankrollInput').value) || DEFAULT_BANKROLL,
      }),
    });
    const result = await res.json();
    if (result.ok) {
      _loggedKeys.add(`${p.game_pk}|${p.bet_side}`);
      if (btn) btn.outerHTML = `<span class="logged-badge">&#10003; Logged</span>`;
      load(false);
    } else {
      if (btn) { btn.disabled = false; btn.textContent = 'Log Bet'; }
      alert('Could not log bet: ' + result.error);
    }
  } catch (err) {
    if (btn) { btn.disabled = false; btn.textContent = 'Log Bet'; }
    alert('Error: ' + err.message);
  }
}

/* ─── Logged today sidebar ─── */
function loggedCard(p) {
  const edgeCls = Number(p.edge || 0) >= 0 ? 'pos' : 'neg';
  const stCls   = p.status === 'Pending' ? 'chip chip-amber' : 'chip chip-default';
  return `<div class="logged-row">
    <div>
      <div class="logged-team">${E(p.side)}</div>
      <div class="logged-meta">
        ${E(p.matchup)} &middot; <span class="${edgeCls}">${pct(p.edge)} edge</span> &middot; ${money(p.stake)}
      </div>
    </div>
    <div class="logged-right">
      <span class="logged-odds">${E(p.odds)}</span>
      <span class="${stCls}">${E(p.status)}</span>
    </div>
  </div>`;
}

/* ─── Live bet card ─── */
function liveBetCard(bet) {
  const sc = bet.score || {};
  const isLive  = sc.abstract_state === 'Live';
  const isFinal = sc.abstract_state === 'Final';
  const hasScore = sc.home_runs != null && sc.away_runs != null;
  const scoreTxt = hasScore ? `${sc.away_runs}–${sc.home_runs}` : '–';
  const scoreCls = isLive ? 'live' : isFinal ? 'final' : '';

  let inningTxt = sc.status || 'Scheduled';
  let inningCls = '';
  if (isLive && sc.inning) { inningTxt = `${sc.inning_half} ${sc.inning}`; inningCls = 'live'; }
  else if (isFinal) { inningTxt = 'Final'; }

  const clvChip = bet.clv != null
    ? `<span class="${bet.clv >= 0 ? 'chip chip-green' : 'chip chip-red'}">${pct(bet.clv)} CLV</span>`
    : `<span class="chip chip-default">CLV pending</span>`;

  const hasCurr  = bet.current_home_odds && bet.current_home_odds !== '--';
  const hasClose = bet.home_close && bet.home_close !== '--';
  let oddsRow = '';
  if (hasCurr || hasClose) {
    oddsRow = `<div class="live-odds-row">`;
    if (hasCurr)  oddsRow += `<span><b>Now:</b> ${E(bet.home_team)} ${E(bet.current_home_odds)} / ${E(bet.away_team)} ${E(bet.current_away_odds)}</span>`;
    if (hasClose) oddsRow += `<span><b>Close:</b> ${E(bet.home_team)} ${E(bet.home_close)} / ${E(bet.away_team)} ${E(bet.away_close)}</span>`;
    oddsRow += `</div>`;
  }

  return `<div class="live-card">
    <div class="live-card-hdr">
      <span class="live-matchup-txt">${E(bet.matchup)}</span>
      <span class="chip chip-default" style="font-size:10px">${E(bet.game_date)}</span>
    </div>
    <div class="live-card-body">
      <div class="live-main-row">
        <div>
          <div class="live-bet-lbl">Bet</div>
          <div class="live-bet-team">${E(bet.side)}</div>
        </div>
        <div class="live-score-box">
          <div class="live-score ${scoreCls}">${scoreTxt}</div>
          <div class="live-inning ${inningCls}">${E(inningTxt)}</div>
        </div>
      </div>
      <div class="live-chips">
        <span class="chip chip-gold">${E(bet.odds)}</span>
        <span class="${edgeChipCls(bet.edge)}">${pct(bet.edge)} edge</span>
        <span class="chip chip-default">${money(bet.stake)}</span>
        ${clvChip}
      </div>
      ${oddsRow}
    </div>
  </div>`;
}

/* ─── Canvas chart (dark theme) ─── */
function drawChart(el, points, key, color, isMoney) {
  const dpr = window.devicePixelRatio || 1;
  const rect = el.getBoundingClientRect();
  const W = rect.width || 380, H = rect.height || 180;
  el.width  = Math.round(W * dpr);
  el.height = Math.round(H * dpr);
  const ctx = el.getContext('2d');
  ctx.scale(dpr, dpr);
  const PAD = {l:52, r:12, t:10, b:24};
  const gW = W - PAD.l - PAD.r, gH = H - PAD.t - PAD.b;
  ctx.fillStyle = '#12141a'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle = '#252833'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l, PAD.t+gH); ctx.lineTo(PAD.l+gW, PAD.t+gH);
  ctx.stroke();
  if (!points.length) {
    ctx.fillStyle = '#42475a'; ctx.font = '12px system-ui'; ctx.textAlign = 'center';
    ctx.fillText('No settled bets yet', PAD.l+gW/2, PAD.t+gH/2); return;
  }
  const vals = points.map(p => Number(p[key] || 0));
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (!isMoney) { lo = Math.min(lo,0); hi = Math.max(hi,0); }
  if (lo === hi) { lo -= 0.01; hi += 0.01; }
  const xP = i => PAD.l + (i / Math.max(points.length-1,1)) * gW;
  const yP = v => PAD.t + gH - ((v-lo)/(hi-lo)) * gH;
  if (!isMoney) {
    const y0 = yP(0);
    ctx.save(); ctx.strokeStyle = '#31354a'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(PAD.l,y0); ctx.lineTo(PAD.l+gW,y0); ctx.stroke(); ctx.restore();
  }
  ctx.beginPath();
  vals.forEach((v,i) => i ? ctx.lineTo(xP(i),yP(v)) : ctx.moveTo(xP(i),yP(v)));
  ctx.lineTo(xP(vals.length-1),PAD.t+gH); ctx.lineTo(PAD.l,PAD.t+gH); ctx.closePath();
  ctx.fillStyle = color+'1a'; ctx.fill();
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.setLineDash([]);
  ctx.beginPath();
  vals.forEach((v,i) => i ? ctx.lineTo(xP(i),yP(v)) : ctx.moveTo(xP(i),yP(v)));
  ctx.stroke();
  ctx.fillStyle = '#7a7f96'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
  const fmt = v => isMoney ? '$'+Number(v).toFixed(0) : (v>=0?'+':'')+(v*100).toFixed(1)+'%';
  ctx.fillText(fmt(hi), PAD.l-5, PAD.t+8);
  ctx.fillText(fmt(lo), PAD.l-5, PAD.t+gH+4);
}

/* ─── Auto-refresh ─── */
let _timerInterval = null, _nextRefresh = 0;
function startCountdown() {
  if (_timerInterval) clearInterval(_timerInterval);
  _nextRefresh = Date.now() + 5*60*1000;
  _timerInterval = setInterval(() => {
    const rem = Math.max(0, _nextRefresh - Date.now());
    const m = Math.floor(rem/60000), s = Math.floor((rem%60000)/1000);
    $('refreshTs').textContent = `Next refresh ${m}:${String(s).padStart(2,'0')}`;
    if (rem === 0) { _nextRefresh = Date.now()+5*60*1000; load(false); }
  }, 1000);
}

/* ─── Upcoming bet card ─── */
function upcomingCard(bet) {
  const typeLabel = bet.bet_type === 'Prop'
    ? `<span class="pick-market">${E(bet.market||'')}</span>`
    : bet.bet_type === 'F5' ? `<span class="pick-market">F5</span>` : '';
  const sc = bet.score || {};
  const isFinal = sc.abstract_state === 'Final' || sc.status === 'Final' || sc.status === 'Game Over';
  const statusTxt  = isFinal ? 'Game Over' : (sc.status || 'Scheduled');
  const statusClr  = isFinal ? 'var(--muted)' : 'var(--subtle)';
  const hasScore   = sc.home_runs != null && sc.away_runs != null;
  const scoreTxt   = hasScore ? `${sc.away_runs}–${sc.home_runs}` : '';
  return `<div class="live-card">
    <div class="live-card-hdr">
      <span class="live-matchup-txt">${E(bet.matchup)}</span>
      <div style="display:flex;gap:5px;align-items:center">
        ${typeLabel}
        <span class="chip chip-default" style="font-size:10px">${E(bet.game_date)}</span>
      </div>
    </div>
    <div class="live-card-body">
      <div class="live-main-row" style="margin-bottom:10px">
        <div>
          <div class="live-bet-lbl">Bet</div>
          <div class="live-bet-team">${E(bet.side)}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--subtle);margin-bottom:4px">Status</div>
          ${hasScore ? `<div style="font-size:18px;font-weight:900;color:${statusClr};letter-spacing:-.3px">${scoreTxt}</div>` : ''}
          <div style="font-size:${hasScore?'11':'15'}px;font-weight:700;color:${statusClr}">${statusTxt}</div>
        </div>
      </div>
      <div class="live-chips">
        <span class="chip chip-gold">${E(bet.odds)}</span>
        <span class="${edgeChipCls(bet.edge)}">${pct(bet.edge)} edge</span>
        <span class="chip chip-default">${money(bet.stake)}</span>
        ${bet.bookmaker ? `<span class="chip chip-default" style="font-size:10px">${E(bet.bookmaker)}</span>` : ''}
      </div>
    </div>
  </div>`;
}

/* ─── Finished bet row ─── */
function finishedRow(b) {
  const outLower = b.outcome.toLowerCase();
  const pnlNum   = Number(b.profit || 0);
  const pnlStr   = outLower === 'pending' ? '--'
    : (pnlNum >= 0 ? '+' : '') + '$' + Math.abs(pnlNum).toFixed(2);
  const pnlCls   = pnlNum >= 0 ? 'pos' : 'neg';
  const dateStr  = b.game_date ? new Date(b.game_date + 'T12:00:00').toLocaleDateString('en-US',{month:'short',day:'numeric'}) : '--';

  let betLabel, subLabel;
  if (b.bet_type === 'Prop') {
    betLabel = `${E(b.matchup)}`;
    subLabel = `${E(b.side)} · ${E((b.market||'').replace('pitcher_','').replace('batter_','').replace('_',' '))}`;
  } else {
    betLabel = `${E(b.side)}`;
    subLabel = `${E(b.matchup)}${b.bet_type === 'F5' ? ' · F5' : ''}`;
  }

  return `<div class="finished-row ${outLower}">
    <div class="fr-date">${dateStr}</div>
    <div class="fr-main">
      <div class="fr-bet">${betLabel}</div>
      <div class="fr-sub">${subLabel}</div>
    </div>
    <div class="fr-odds">${E(b.odds)}</div>
    <div class="fr-stake">${money(b.stake)}</div>
    <div class="fr-result ${outLower}">${E(b.outcome)}</div>
    <div class="fr-pnl ${outLower === 'pending' ? '' : pnlCls}">${pnlStr}</div>
  </div>`;
}

/* ─── Bet History (Upcoming / Live / Finished) ─── */
let _allBets = [], _historyFilter = 'all';

function setFilter(f, btn) {
  _historyFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderFinished();
}

function renderFinished() {
  const f = _historyFilter;
  const settled = _allBets.filter(b => b.outcome !== 'Pending');
  const filtered = settled.filter(b => {
    if (f === 'all') return true;
    if (f === 'ML' || f === 'F5' || f === 'Prop') return b.bet_type === f;
    return b.outcome === f;
  });
  $('finishedList').innerHTML = filtered.length
    ? filtered.map(finishedRow).join('')
    : `<div class="empty-state">No bets match this filter.</div>`;
}

async function loadBetHistory() {
  $('upcomingGrid').innerHTML  = `<div class="live-empty" style="opacity:.4">Loading…</div>`;
  $('histLiveGrid').innerHTML  = `<div class="live-empty" style="opacity:.4">Loading…</div>`;
  $('finishedList').innerHTML  = `<div class="empty-state" style="opacity:.4">Loading…</div>`;
  try {
    const [liveRes, histRes] = await Promise.all([
      fetch('/api/live_bets'),
      fetch('/api/bet_history'),
    ]);
    if (!liveRes.ok || !histRes.ok) throw new Error('API error');
    const {bets: pendingBets}         = await liveRes.json();
    const {bets: allBets, summary}    = await histRes.json();

    // Split pending by game state
    const liveBets     = pendingBets.filter(b => (b.score?.abstract_state || '') === 'Live');
    const upcomingBets = pendingBets.filter(b => (b.score?.abstract_state || '') !== 'Live');

    $('pendingBadge').textContent  = pendingBets.length;
    $('upcomingBadge').textContent = upcomingBets.length;
    $('liveBadge').textContent     = liveBets.length;

    // Upcoming
    $('upcomingGrid').innerHTML = upcomingBets.length
      ? upcomingBets.map(b => { try { return upcomingCard(b); } catch(e) { return ''; } }).join('')
      : `<div class="live-empty">No upcoming bets.</div>`;

    // Live
    $('histLiveGrid').innerHTML = liveBets.length
      ? liveBets.map(b => { try { return liveBetCard(b); } catch(e) { return ''; } }).join('')
      : `<div class="live-empty">No games currently live.</div>`;

    // Auto-switch to Live when there are live games and user is on Upcoming (default)
    if (liveBets.length > 0 && _activeHistSub === 'upcoming') switchHistory('live');

    // Finished
    _allBets = allBets;
    const wins   = summary.wins, losses = summary.losses;
    const wr     = (wins + losses) > 0 ? ((wins / (wins + losses)) * 100).toFixed(1) + '%' : '--';
    const pnlNum = summary.pnl;
    const pnlStr = (pnlNum >= 0 ? '+' : '') + '$' + Math.abs(pnlNum).toFixed(2);
    $('historySummary').innerHTML =
      `<span class="chip chip-default">${summary.total} bets</span>` +
      `<span class="chip chip-green">${wins}W</span>` +
      `<span class="chip chip-red">${losses}L</span>` +
      (summary.pushes  ? `<span class="chip chip-amber">${summary.pushes} Push</span>`     : '') +
      (summary.pending ? `<span class="chip chip-default">${summary.pending} Pending</span>` : '') +
      `<span class="chip chip-default">${wr} win rate</span>` +
      `<span class="chip ${pnlNum >= 0 ? 'chip-green' : 'chip-red'}">${pnlStr} P&L</span>`;
    renderFinished();

  } catch (err) {
    $('upcomingGrid').innerHTML = `<div class="live-empty">Error: ${E(err.message)}</div>`;
  }
}

/* ─── Analytics tab ─── */
async function loadAnalytics() {
  try {
    const res = await fetch('/api/analytics');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();

    // Sample progress
    const { settled, target } = d.sample_progress;
    const fillPct = Math.min(100, (settled / target * 100)).toFixed(1);
    $('sampleCount').textContent = settled;
    $('sampleFill').style.width  = fillPct + '%';
    const left = target - settled;
    $('sampleSub').textContent = settled >= target
      ? '✓ Sample target reached — evaluate for real money'
      : `${left} more settled bet${left===1?'':'s'} to reach ${target}-bet target`;

    // Automation status
    $('autoStatus').innerHTML = d.auto_status.map(j => {
      const dotCls = j.ok === null ? 'never' : j.ok ? 'ok' : 'err';
      const timeTxt = j.ok === false
        ? `<span style="color:var(--red)">${E(j.last_run)} — error</span>`
        : `<span class="auto-time">${E(j.last_run)}</span>`;
      return `<div class="auto-row">
        <span class="auto-name">${E(j.name)}</span>
        <div style="display:flex;align-items:center;gap:8px">
          ${timeTxt}<div class="auto-dot ${dotCls}"></div>
        </div>
      </div>`;
    }).join('');

    // Edge bucket table
    $('edgeBuckets').innerHTML = d.edge_buckets.length
      ? d.edge_buckets.map(b => {
          const roiCls = b.roi >= 0 ? 'pos' : 'neg';
          return `<tr>
            <td style="font-weight:700">${E(b.bucket)}</td>
            <td style="color:var(--muted)">${b.count}</td>
            <td>${b.wins}W–${b.losses}L</td>
            <td class="${roiCls}">${pct(b.roi)}</td>
          </tr>`;
        }).join('')
      : `<tr><td colspan="4" style="color:var(--subtle);padding:16px;text-align:center">No settled bets yet</td></tr>`;

    // CLV chart
    $('clvChartPill').textContent = `${d.clv_series.length} bets`;
    if (d.clv_series.length) {
      drawChart($('clvChart'), d.clv_series, 'clv', '#22c55e', false);
    } else {
      $('clvChart').getContext('2d').clearRect(0,0,$('clvChart').width,$('clvChart').height);
    }

    // Weekly cumulative P&L chart
    if (d.pnl_curve.length) {
      drawChart($('weeklyPnlChart'), d.pnl_curve, 'cum_pnl', '#3b82f6', true);
    }
  } catch (err) {
    console.error('Analytics error:', err);
  }
}

/* ─── Main load ─── */
const DEFAULT_BANKROLL = __DEFAULT_BANKROLL__;
let _loggedPropsToday = [];

async function load(forceOdds = false) {
  $('app').classList.add('loading');
  const bankroll = Number($('bankrollInput').value) || DEFAULT_BANKROLL;
  const edge     = Number($('edgeSelect').value) || 0.10;
  const url = `/api/dashboard?bankroll=${bankroll}&min_edge=${edge}&refresh=${forceOdds?1:0}`;
  let data;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    $('app').classList.remove('loading');
    $('subtitle').textContent = 'API error — ' + err.message;
    return;
  }
  $('app').classList.remove('loading');

  const {metrics:m, display:d, freshness:f, live, logged_today, suggestions} = data;

  $('subtitle').textContent =
    `${f.season_games.toLocaleString()} ${f.season} games · latest ${f.latest_game_date||'none'} · min edge ${d.min_edge}`;

  /* Status strip — "stale" only if data is 2+ days old */
  const _latestMs = f.latest_game_date ? new Date(f.latest_game_date + 'T12:00:00').getTime() : 0;
  const _todayMs  = new Date(f.today    + 'T12:00:00').getTime();
  const _daysOld  = (_todayMs - _latestMs) / 86400000;
  const fresh = _daysOld <= 1;
  $('freshDot').className = fresh ? 'dot' : 'dot stale';
  $('freshLabel').textContent = fresh
    ? `Data current · last game ${f.latest_game_date}`
    : `Data stale · last game ${f.latest_game_date||'none'} · run update_and_pick.py`;
  $('statusExtra').innerHTML =
    `<span class="chip chip-default">${f.total_games.toLocaleString()} games in DB</span>` +
    `&nbsp;<span class="chip ${f.close_2025?'chip-default':'chip-amber'}">${f.close_2025.toLocaleString()} 2025 closes</span>`;

  /* Metric cards */
  $('mBankroll').textContent = d.current_bankroll;
  $('mAvail').textContent    = `${d.available_bankroll} avail · ${d.pending_at_risk} at risk`;

  const roiNum = Number(m.roi || 0);
  $('mRoi').textContent = d.roi;
  $('mRoi').className   = 'm-val ' + (roiNum >= 0 ? 'pos' : 'neg');
  $('mcRoi').className  = 'm-card ' + (roiNum >= 0 ? 'pos-accent' : 'neg-accent');
  $('mProfit').textContent = d.total_profit + ' P&L';

  $('mRecord').textContent    = `${m.wins}–${m.losses}`;
  $('mRecordSub').textContent = m.settled
    ? `${pct(m.win_rate,false)} win rate · ${m.settled} settled`
    : 'No settled bets yet';
  const wrNum = m.win_rate - 0.5;
  $('mcRecord').className = 'm-card ' + (wrNum >= 0 ? 'pos-accent' : 'neg-accent');

  const clvNum = m.mean_clv;
  $('mClv').textContent  = d.mean_clv;
  $('mClv').className    = 'm-val ' + (clvNum == null ? '' : clvNum >= 0 ? 'pos' : 'neg');
  $('mcClv').className   = 'm-card ' + (clvNum == null ? '' : clvNum >= 0 ? 'pos-accent' : 'neg-accent');
  $('mClvSub').textContent = m.clv_count
    ? `${m.clv_count} bet${m.clv_count===1?'':'s'} with closing-line data`
    : 'No closing-line data yet';

  $('pendingPill').textContent  = `${m.pending} pending`;
  $('watchlistThresh').textContent = Math.round(edge * 100);

  /* Picks */
  _loggedKeys   = new Set(logged_today.filter(p=>p.game_pk).map(p=>`${p.game_pk}|${p.bet_side}`));
  _currentPicks = [...(live.strong||[]), ...(live.watchlist||[])];

  $('liveStatus').textContent = live.error ? 'Odds error' : `${live.games_with_odds} games${live.cached?' · cached':''}`;
  $('liveStatus').className   = `chip ${live.error ? 'chip-red' : 'chip-default'}`;

  $('strongPicks').innerHTML = live.error
    ? emptyState(live.error)
    : live.strong.length
      ? `<div class="pick-list">${live.strong.map((p,i)=>pickTile(p,i)).join('')}</div>`
      : emptyState('No picks clear the current edge threshold.');

  const off = (live.strong||[]).length;
  $('watchlist').innerHTML = (live.watchlist||[]).length
    ? `<div class="pick-list">${live.watchlist.map((p,i)=>pickTile(p,off+i)).join('')}</div>`
    : emptyState('No marginal picks above 3% edge right now.');

  /* Charts */
  drawChart($('bankrollChart'), m.curve, 'bankroll', '#c8a84b', true);
  drawChart($('roiChart'),      m.curve, 'roi',      '#27c47a', false);

  /* Recent bets */
  $('recentBets').innerHTML = m.latest_bets.length
    ? m.latest_bets.map(r => {
        const rCls  = r.outcome==='Won' ? 'chip chip-green' : r.outcome==='Lost' ? 'chip chip-red' : 'chip chip-default';
        const clvTd = r.clv != null
          ? `<span class="chip ${Number(r.clv)>=0?'chip-green':'chip-red'}" style="font-size:11px">${pct(r.clv)} CLV</span>`
          : `<span style="color:var(--subtle);font-size:11px">—</span>`;
        return `<tr>
          <td style="color:var(--muted);font-size:12px">${E(r.game_date)}</td>
          <td>${E(r.matchup)}</td>
          <td style="font-weight:800">${E(r.side)}</td>
          <td style="color:var(--gold);font-weight:800">${E(r.odds)}</td>
          <td class="${posNeg(r.edge)}">${pct(r.edge)}</td>
          <td>${money(r.stake)}</td>
          <td>${clvTd}</td>
          <td><span class="${rCls}">${E(r.outcome)}</span></td>
        </tr>`;
      }).join('')
    : `<tr><td colspan="8" style="text-align:center;color:var(--subtle);padding:24px">No paper bets logged yet.</td></tr>`;

  /* Freshness */
  $('freshPill').textContent = fresh ? 'Current' : 'Stale';
  $('freshPill').className   = `chip ${fresh ? 'chip-green' : 'chip-amber'}`;
  $('freshPills').innerHTML  =
    `<span class="chip chip-default">${f.total_games.toLocaleString()} total</span>` +
    `<span class="chip chip-default">${f.season_games.toLocaleString()} in ${f.season}</span>` +
    `<span class="chip ${f.close_2025?'chip-default':'chip-amber'}">${f.close_2025.toLocaleString()} 2025 closes</span>`;

  /* Logged today (ML) */
  $('loggedPill').textContent = logged_today.length;
  $('loggedToday').innerHTML  = logged_today.length
    ? `<div class="logged-list">${logged_today.map(loggedCard).join('')}</div>`
    : emptyState('No picks logged for today.');

  /* Logged props today (sidebar on Props tab) */
  if (data.logged_props_today) {
    _loggedPropsToday = data.logged_props_today;
    $('propLoggedPill').textContent = _loggedPropsToday.length;
    $('propLoggedToday').innerHTML = _loggedPropsToday.length
      ? `<div class="logged-list">${_loggedPropsToday.map(p => {
          const edgeCls = Number(p.edge||0)>=0?'pos':'neg';
          const stCls = p.status==='Pending'?'chip chip-amber':'chip chip-default';
          return `<div class="logged-row">
            <div>
              <div class="logged-team">${E(p.player_name)}</div>
              <div class="logged-meta">${E(p.bet_side)} ${E(String(p.line))} &middot;
                <span class="${edgeCls}">${pct(p.edge)} edge</span> &middot; ${money(p.stake)}
              </div>
            </div>
            <div class="logged-right">
              <span class="logged-odds">${E(p.odds)}</span>
              <span class="${stCls}">${E(p.status)}</span>
            </div>
          </div>`;
        }).join('')}</div>`
      : emptyState('No props logged today.');
  }

  /* Suggestions */
  $('suggestions').innerHTML = suggestions.map(s =>
    `<div class="suggest-row ${E(s.priority).toLowerCase()}">
      <div class="suggest-title">${E(s.title)}</div>
      <div class="suggest-body">${E(s.body)}</div>
    </div>`
  ).join('');

  $('refreshTs').textContent =
    'Updated ' + new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
  startCountdown();
}

function betSlip(b) {
  const isHome    = b.bet_side === 'home';
  const outLower  = b.outcome.toLowerCase();
  const pnlNum    = Number(b.profit || 0);
  const pnlStr    = b.outcome === 'Pending' ? '--'
    : (pnlNum >= 0 ? '+' : '') + '$' + Math.abs(pnlNum).toFixed(2);
  const pnlCls    = b.outcome === 'Pending' ? '' : pnlNum >= 0 ? 'pos' : 'neg';
  const clvChip   = b.clv != null
    ? `<span class="chip ${Number(b.clv)>=0?'chip-green':'chip-red'}">${pct(b.clv)} CLV</span>`
    : '';
  const bkChip   = b.bookmaker
    ? `<span class="chip chip-default" style="font-size:10px">${E(b.bookmaker)}</span>` : '';
  const typeChip = b.bet_type === 'F5'
    ? `<span class="chip chip-blue" style="font-size:10px">F5</span>` : '';

  // Prop bets get a different slip layout
  if (b.bet_type === 'Prop') {
    const isOver    = b.bet_side === 'over';
    const actualChip = b.actual_value != null
      ? `<span class="chip chip-default">Actual: ${b.actual_value} K</span>` : '';
    const mktLabel  = (b.market || '').replace('pitcher_', '').replace('_', ' ');
    return `<div class="slip ${outLower}">
      <div class="slip-hdr">
        <span class="slip-matchup">${E(b.matchup)}</span>
        <span class="slip-date">${E(b.game_date)}</span>
      </div>
      <div class="slip-odds-row">
        <div class="slip-team-btn ${!isOver ? 'selected' : ''}">
          <div class="slip-team-name">Under ${b.line}</div>
          <div class="slip-team-odds">${E(b.odds)}</div>
        </div>
        <div class="slip-team-btn ${isOver ? 'selected' : ''}">
          <div class="slip-team-name">Over ${b.line}</div>
          <div class="slip-team-odds">${E(b.odds)}</div>
        </div>
      </div>
      <div class="slip-body">
        <div class="slip-stats">
          <span class="chip chip-gold">${E(b.odds)}</span>
          <span class="chip ${Number(b.edge||0)>=0?'chip-green':'chip-red'}">${pct(b.edge)} edge</span>
          <span class="chip chip-default">${money(b.stake)}</span>
          <span class="chip chip-blue">Model ${pct(b.model_prob,false)}</span>
          <span class="chip chip-amber" style="font-size:10px">${E(mktLabel)}</span>
          ${actualChip}${bkChip}
        </div>
        <div class="slip-result-row">
          <span class="slip-outcome ${outLower}">${E(b.outcome)}</span>
          <span class="slip-pnl ${pnlCls}">${pnlStr}</span>
        </div>
      </div>
    </div>`;
  }

  return `<div class="slip ${outLower}">
    <div class="slip-hdr">
      <span class="slip-matchup">${E(b.matchup)}</span>
      <span class="slip-date">${E(b.game_date)}</span>
    </div>
    <div class="slip-odds-row">
      <div class="slip-team-btn ${!isHome ? 'selected' : ''}">
        <div class="slip-team-name">${E(b.away_team)}</div>
        <div class="slip-team-odds">${E(b.away_open || '--')}</div>
      </div>
      <div class="slip-team-btn ${isHome ? 'selected' : ''}">
        <div class="slip-team-name">${E(b.home_team)}</div>
        <div class="slip-team-odds">${E(b.home_open || '--')}</div>
      </div>
    </div>
    <div class="slip-body">
      <div class="slip-stats">
        <span class="chip chip-gold">${E(b.odds)}</span>
        <span class="chip ${Number(b.edge||0)>=0?'chip-green':'chip-red'}">${pct(b.edge)} edge</span>
        <span class="chip chip-default">${money(b.stake)}</span>
        <span class="chip chip-blue">Model ${pct(b.model_prob,false)}</span>
        ${clvChip}${bkChip}${typeChip}
      </div>
      <div class="slip-result-row">
        <span class="slip-outcome ${outLower}">${E(b.outcome)}</span>
        <span class="slip-pnl ${pnlCls}">${pnlStr}</span>
      </div>
    </div>
  </div>`;
}


$('refreshBtn').addEventListener('click',     () => load(false));
$('refreshLiveBtn').addEventListener('click', () => load(true));
$('bankrollInput').addEventListener('change', () => load(false));
$('edgeSelect').addEventListener('change',   () => load(false));

load(false);
</script>
</body>
</html>
"""


def _bet_history_payload() -> dict:
    """Return all paper bets (ML + F5 + Props) sorted newest first for the History tab."""
    with _connect() as conn:
        ml_rows = conn.execute(
            "SELECT *, 'ML' AS bet_type FROM paper_bets ORDER BY game_date DESC, id DESC"
        ).fetchall()
        f5_rows = conn.execute(
            "SELECT *, 'F5' AS bet_type FROM f5_paper_bets ORDER BY game_date DESC, id DESC"
        ).fetchall()
        prop_rows = conn.execute(
            "SELECT *, 'Prop' AS bet_type FROM prop_bets ORDER BY game_date DESC, id DESC"
        ).fetchall()

    def _outcome(raw) -> str:
        if raw is None:   return "Pending"
        if int(raw) == 1: return "Won"
        if int(raw) == 0: return "Lost"
        return "Push"

    def fmt(r: sqlite3.Row) -> dict:
        side_team = r["home_team"] if r["bet_side"] == "home" else r["away_team"]
        return {
            "id":          r["id"],
            "bet_type":    r["bet_type"],
            "game_date":   r["game_date"],
            "matchup":     f"{r['away_team']} @ {r['home_team']}",
            "home_team":   r["home_team"],
            "away_team":   r["away_team"],
            "side":        side_team,
            "bet_side":    r["bet_side"],
            "odds":        _american(r["bet_american_odds"]),
            "home_open":   _american(r["home_american_open"]),
            "away_open":   _american(r["away_american_open"]),
            "model_prob":  float(r["model_prob"] or 0),
            "fair_prob":   float(r["fair_prob"] or 0),
            "edge":        float(r["edge"] or 0),
            "stake":       float(r["stake_dollars"] or 0),
            "profit":      r["profit_dollars"],
            "outcome":     _outcome(r["outcome"]),
            "clv":         r["clv"],
            "bookmaker":   r["bookmaker"] or "",
            "created_at":  r["created_at"] or "",
        }

    def fmt_prop(r: sqlite3.Row) -> dict:
        return {
            "id":           r["id"],
            "bet_type":     "Prop",
            "game_date":    r["game_date"],
            "matchup":      r["player_name"],
            "home_team":    r["team"] or "",
            "away_team":    r["opponent"] or "",
            "side":         f"{r['bet_side'].title()} {r['line']}",
            "bet_side":     r["bet_side"],
            "odds":         _american(r["american_odds"]),
            "home_open":    None,
            "away_open":    None,
            "line":         float(r["line"] or 0),
            "market":       r["market"] or "",
            "actual_value": r["actual_value"],
            "model_prob":   float(r["model_prob"] or 0),
            "fair_prob":    float(r["fair_prob"] or 0),
            "edge":         float(r["edge"] or 0),
            "stake":        float(r["stake_dollars"] or 0),
            "profit":       r["profit_dollars"],
            "outcome":      _outcome(r["outcome"]),
            "clv":          None,
            "bookmaker":    r["bookmaker"] or "",
            "created_at":   r["created_at"] or "",
        }

    ml    = [fmt(r)      for r in ml_rows]
    f5    = [fmt(r)      for r in f5_rows]
    props = [fmt_prop(r) for r in prop_rows]

    combined = sorted(ml + f5 + props, key=lambda x: (x["game_date"], x["id"]), reverse=True)

    wins   = sum(1 for b in combined if b["outcome"] == "Won")
    losses = sum(1 for b in combined if b["outcome"] == "Lost")
    pushes = sum(1 for b in combined if b["outcome"] == "Push")
    pnl    = sum(float(b["profit"] or 0) for b in combined)

    return {
        "bets":    combined,
        "summary": {
            "total":    len(combined),
            "wins":     wins,
            "losses":   losses,
            "pushes":   pushes,
            "pending":  sum(1 for b in combined if b["outcome"] == "Pending"),
            "pnl":      round(pnl, 2),
            "win_rate": wins / (wins + losses) if (wins + losses) else 0.0,
        },
    }


def render_html() -> str:
    selected = {"0.10": "", "0.08": "", "0.05": "", "0.03": ""}
    selected[f"{DEFAULT_MIN_EDGE:.2f}"] = "selected"
    return (
        HTML_TEMPLATE
        .replace("__DEFAULT_BANKROLL__", f"{DEFAULT_BANKROLL:g}")
        .replace("__EDGE_010__", selected["0.10"])
        .replace("__EDGE_008__", selected["0.08"])
        .replace("__EDGE_005__", selected["0.05"])
        .replace("__EDGE_003__", selected["0.03"])
    )


def _parse_float(value: str | None, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if not _check_auth(self):
            _send_auth_challenge(self)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(render_html())
            return
        if parsed.path == "/api/dashboard":
            qs = parse_qs(parsed.query)
            bankroll = _parse_float(qs.get("bankroll", [None])[0], DEFAULT_BANKROLL)
            min_edge = _parse_float(qs.get("min_edge", [None])[0], DEFAULT_MIN_EDGE)
            bankroll = max(1.0, bankroll)
            min_edge = min(max(0.0, min_edge), 1.0)
            refresh = qs.get("refresh", ["0"])[0] == "1"
            self._send_json(dashboard_payload(bankroll, min_edge, refresh))
            return
        if parsed.path == "/api/live_bets":
            self._send_json(_live_bets_payload())
            return
        if parsed.path == "/api/bet_history":
            self._send_json(_bet_history_payload())
            return
        if parsed.path == "/api/analytics":
            self._send_json(_analytics_payload())
            return
        if parsed.path == "/api/props":
            qs = parse_qs(parsed.query)
            bankroll = _parse_float(qs.get("bankroll", [None])[0], DEFAULT_BANKROLL)
            min_edge = _parse_float(qs.get("min_edge", [None])[0], DEFAULT_MIN_EDGE)
            bankroll = max(1.0, bankroll)
            min_edge = min(max(0.0, min_edge), 1.0)
            refresh  = qs.get("refresh", ["0"])[0] == "1"
            self._send_json(_live_prop_recommendations(bankroll, min_edge, refresh))
            return
        if parsed.path == "/api/logged_props_today":
            today = datetime.now().strftime("%Y-%m-%d")
            with _connect() as conn:
                self._send_json({"props": _logged_today_props(conn, today)})
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if not _check_auth(self):
            _send_auth_challenge(self)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/log_bet":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
            except Exception:
                self._send_json({"ok": False, "error": "Invalid JSON"})
                return
            self._handle_log_bet(data)
            return
        if parsed.path == "/api/log_prop_bet":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
            except Exception:
                self._send_json({"ok": False, "error": "Invalid JSON"})
                return
            self._handle_log_prop_bet(data)
            return
        self.send_error(404, "Not found")

    def _handle_log_bet(self, data: dict) -> None:
        try:
            today     = datetime.now().strftime("%Y-%m-%d")
            game_pk   = int(data.get("game_pk") or 0) or None
            home      = str(data["home_team"])
            away      = str(data["away_team"])
            side      = str(data["bet_side"])
            odds_raw  = float(data["american_odds_raw"])
            m_prob    = float(data["model_prob"])
            f_prob    = float(data["fair_prob"])
            edge      = float(data["edge"])
            stake     = float(data["stake"])
            bookmaker = str(data.get("bookmaker", ""))
            game_date = str(data.get("game_date", today))
            bankroll  = float(data.get("bankroll") or DEFAULT_BANKROLL)
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json({"ok": False, "error": f"Bad data: {exc}"})
            return

        with _connect() as conn:
            if game_pk:
                dup = conn.execute(
                    "SELECT id FROM paper_bets WHERE game_pk=? AND bet_side=? AND game_date=?",
                    (game_pk, side, game_date),
                ).fetchone()
                if dup:
                    self._send_json({"ok": False, "error": "Already logged this game and side."})
                    return
            dec_odds = (100 / abs(odds_raw) + 1) if odds_raw < 0 else (odds_raw / 100 + 1)
            conn.execute(
                """INSERT INTO paper_bets
                   (game_pk, game_date, home_team, away_team, bet_side,
                    bet_american_odds, bet_decimal_odds, model_prob, fair_prob,
                    edge, stake_dollars, stake_fraction, bankroll_at_bet, bookmaker)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (game_pk, game_date, home, away, side,
                 odds_raw, dec_odds, m_prob, f_prob, edge, stake,
                 stake / bankroll if bankroll else None, bankroll, bookmaker),
            )
            conn.commit()

        self._send_json({"ok": True})

    def _handle_log_prop_bet(self, data: dict) -> None:
        try:
            today       = datetime.now().strftime("%Y-%m-%d")
            game_pk     = int(data.get("game_pk") or 0) or None
            home        = str(data["home_team"])
            away        = str(data["away_team"])
            player_name = str(data["player_name"])
            market      = str(data["market"])
            line        = float(data["line"])
            side        = str(data["bet_side"])
            odds_raw    = float(data["american_odds_raw"])
            m_prob      = float(data["model_prob"])
            f_prob      = float(data["fair_prob"])
            edge        = float(data["edge"])
            stake       = float(data["stake"])
            bookmaker   = str(data.get("bookmaker", ""))
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json({"ok": False, "error": f"Bad data: {exc}"})
            return

        bankroll = float(data.get("bankroll") or DEFAULT_BANKROLL)

        with _connect() as conn:
            dup = conn.execute(
                "SELECT id FROM prop_bets WHERE game_pk=? AND player_name=? AND market=? AND bet_side=?",
                (game_pk or 0, player_name, market, side),
            ).fetchone()
            if dup:
                self._send_json({"ok": False, "error": "Already logged this prop bet."})
                return
            dec_odds = (100 / abs(odds_raw) + 1) if odds_raw < 0 else (odds_raw / 100 + 1)
            conn.execute(
                """INSERT INTO prop_bets
                   (game_pk, game_date, player_name, team, opponent, market, line,
                    bet_side, american_odds, decimal_odds, fair_prob, model_prob,
                    edge, stake_dollars, stake_fraction, bankroll_at_bet, bookmaker,
                    is_paper, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    game_pk or 0, today, player_name, home, away, market, line,
                    side, odds_raw, dec_odds, f_prob, m_prob, edge, stake,
                    stake / bankroll if bankroll else None, bankroll, bookmaker, 1,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        self._send_json({"ok": True})

    def log_message(self, *_) -> None:
        return

    def _send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict) -> None:
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    global DEFAULT_BANKROLL, DEFAULT_MIN_EDGE, DASHBOARD_PASSWORD

    parser = argparse.ArgumentParser(description="Run the MLB betting dashboard.")
    parser.add_argument("--host",     default="127.0.0.1",
                        help="Bind address. Use 0.0.0.0 to allow external access.")
    parser.add_argument("--port",     type=int,   default=8765)
    parser.add_argument("--bankroll", type=float, default=CONFIG_DEFAULT_BANKROLL)
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE)
    parser.add_argument("--password", type=str,   default=os.getenv("DASHBOARD_PASSWORD", ""),
                        help="Protect the dashboard with HTTP Basic Auth.")
    args = parser.parse_args()

    DEFAULT_BANKROLL    = max(1.0, args.bankroll)
    DEFAULT_MIN_EDGE    = min(max(0.0, args.min_edge), 1.0)
    DASHBOARD_PASSWORD  = args.password
    LIVE_CACHE["bankroll"] = DEFAULT_BANKROLL

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url    = f"http://{args.host}:{args.port}"
    print(f"Dashboard running at {url}")
    if DASHBOARD_PASSWORD:
        print("Password protection: ON")
    else:
        print("Password protection: OFF (local only — use --password to enable)")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()

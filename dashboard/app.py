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
from datetime import datetime
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

    starting_bankroll = bankroll

    total_staked = sum(float(r["stake_dollars"] or 0) for r in settled)
    total_profit = sum(float(r["profit_dollars"] or 0) for r in settled)
    pending_at_risk = sum(float(r["stake_dollars"] or 0) for r in pending)
    wins = sum(int(r["outcome"] or 0) for r in settled)
    losses = len(settled) - wins
    roi = total_profit / total_staked if total_staked else 0.0
    current_bankroll = starting_bankroll + total_profit
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
        "total_logged": len(rows),
        "settled": len(settled),
        "pending": len(pending),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(settled) if settled else 0.0,
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
        and LIVE_CACHE["min_edge"] == min_edge
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
    """Pending bets enriched with live scores and current odds from cache."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM paper_bets WHERE outcome IS NULL ORDER BY game_date, game_pk"
        ).fetchall()

    bets = []
    for r in rows:
        side_team = r["home_team"] if r["bet_side"] == "home" else r["away_team"]
        bets.append({
            "id":        r["id"],
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

    scores = _fetch_live_scores([b["game_pk"] for b in bets])
    for bet in bets:
        bet["score"] = scores.get(bet["game_pk"], {})

    # Current odds from the live cache (populated when odds page loads)
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
    if freshness["latest_game_date"] != freshness["today"]:
        suggestions.append({
            "title": "Refresh 2026 results",
            "body": "Run the updater before picks so rolling team form uses the newest completed games.",
            "priority": "High",
        })
    n = metrics["pending"]
    if n > 0:
        suggestions.append({
            "title": f"Settle {n} pending bet{'s' if n != 1 else ''}",
            "body": f"{n} paper bet{'s are' if n != 1 else ' is'} still open. Settling keeps ROI and bankroll accurate.",
            "priority": "High",
        })
    if metrics["total_logged"] > 0 and metrics["clv_count"] < metrics["total_logged"]:
        suggestions.append({
            "title": "Record closing lines",
            "body": "CLV is the best signal that the model is beating the market, not just getting lucky.",
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
        recs       = find_prop_edges(scored, bankroll=bankroll, min_edge=0.05)
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
      --bg:        #0b0d11;
      --surface:   #12141a;
      --surface2:  #191c24;
      --surface3:  #21242f;
      --border:    #252833;
      --border2:   #313544;
      --text:      #e9ebf2;
      --muted:     #7a7f96;
      --subtle:    #42475a;
      --gold:      #c8a84b;
      --gold2:     #e8c96a;
      --gold-bg:   rgba(200,168,75,.13);
      --green:     #27c47a;
      --green-bg:  rgba(39,196,122,.13);
      --red:       #e8503a;
      --red-bg:    rgba(232,80,58,.13);
      --blue:      #4f8ef7;
      --blue-bg:   rgba(79,142,247,.13);
      --amber:     #f0a500;
      --amber-bg:  rgba(240,165,0,.13);
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", sans-serif;
      font-size: 14px;
      line-height: 1.5;
      min-height: 100vh;
    }

    /* ── Header ── */
    header {
      position: sticky; top: 0; z-index: 200;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(12px);
    }
    .hdr {
      max-width: 1440px; margin: 0 auto;
      padding: 0 20px; height: 58px;
      display: flex; align-items: center; justify-content: space-between; gap: 16px;
    }
    .hdr-left { display: flex; align-items: center; gap: 20px; }

    /* Brand */
    .brand { display: flex; align-items: center; gap: 11px; }
    .brand-icon {
      width: 34px; height: 34px; border-radius: 8px;
      background: var(--gold);
      display: flex; align-items: center; justify-content: center;
      font-size: 17px; font-weight: 900; color: #0b0d11; flex-shrink: 0;
      letter-spacing: -1px;
    }
    .brand-name { font-size: 16px; font-weight: 800; letter-spacing: -.3px; color: var(--text); }
    .brand-sub  { font-size: 11px; color: var(--muted); margin-top: 1px; }

    /* Nav tabs */
    .nav { display: flex; gap: 2px; background: var(--surface2); border: 1px solid var(--border); border-radius: 9px; padding: 3px; }
    .nav-btn {
      height: 30px; padding: 0 14px; border: none; border-radius: 6px;
      background: transparent; color: var(--muted);
      font-size: 13px; font-weight: 600; font-family: inherit;
      cursor: pointer; display: flex; align-items: center; gap: 6px;
      transition: all .15s; white-space: nowrap;
    }
    .nav-btn:hover { color: var(--text); background: var(--surface3); }
    .nav-btn.active { background: var(--gold); color: #0b0d11; }
    .nav-badge {
      background: var(--red); color: #fff;
      border-radius: 10px; font-size: 10px; font-weight: 800;
      padding: 1px 5px; min-width: 16px; text-align: center;
    }
    .nav-btn.active .nav-badge { background: rgba(0,0,0,.25); }

    /* Controls */
    .hdr-controls { display: flex; gap: 8px; align-items: center; }
    .ctl {
      height: 32px; border: 1px solid var(--border2); background: var(--surface2); color: var(--text);
      border-radius: 7px; padding: 0 10px; font-size: 13px; font-family: inherit; outline: none;
      transition: border-color .15s;
    }
    .ctl:focus { border-color: var(--gold); }
    select.ctl { cursor: pointer; }
    .btn { cursor: pointer; font-weight: 700; display: flex; align-items: center; gap: 5px; }
    .btn-ghost { background: transparent; color: var(--muted); }
    .btn-ghost:hover { color: var(--text); border-color: var(--border2); }
    .btn-gold { background: var(--gold); color: #0b0d11; border-color: var(--gold); }
    .btn-gold:hover { background: var(--gold2); border-color: var(--gold2); }
    .refresh-ts { font-size: 11px; color: var(--subtle); white-space: nowrap; }

    /* ── Shell ── */
    .shell { max-width: 1440px; margin: 0 auto; padding: 18px 20px; }

    /* ── Status strip ── */
    .status-strip {
      background: var(--surface); border: 1px solid var(--border); border-radius: 9px;
      padding: 9px 14px; font-size: 12px; color: var(--muted);
      display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 16px;
    }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); display: inline-block; flex-shrink: 0; }
    .dot.stale { background: var(--amber); }

    /* ── Two-col layout ── */
    .overview-grid { display: grid; grid-template-columns: minmax(0,1fr) 340px; gap: 16px; }
    .col { display: flex; flex-direction: column; gap: 14px; }

    /* ── Metrics row ── */
    .metrics { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; }
    .m-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 16px 14px; position: relative; overflow: hidden;
    }
    .m-card::after {
      content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
      background: var(--border2);
    }
    .m-card.pos-accent::after { background: var(--green); }
    .m-card.neg-accent::after { background: var(--red); }
    .m-card.gold-accent::after { background: var(--gold); }
    .m-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; color: var(--muted); }
    .m-val { font-size: 26px; font-weight: 900; line-height: 1.1; margin-top: 9px; letter-spacing: -.5px; }
    .m-val.pos  { color: var(--green); }
    .m-val.neg  { color: var(--red); }
    .m-val.gold { color: var(--gold); }
    .m-sub { font-size: 11px; color: var(--muted); margin-top: 5px; }

    /* ── Card ── */
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
    .card-hd {
      padding: 13px 16px 12px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
    }
    .card-title { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .7px; color: var(--muted); }
    .card-body  { padding: 14px 16px; }

    /* ── Chips / pills ── */
    .chip {
      display: inline-flex; align-items: center; height: 22px; padding: 0 8px;
      border-radius: 5px; font-size: 11px; font-weight: 700; white-space: nowrap; gap: 3px;
    }
    .chip-default { background: var(--surface2); color: var(--muted); border: 1px solid var(--border2); }
    .chip-gold    { background: var(--gold-bg);  color: var(--gold);  }
    .chip-green   { background: var(--green-bg); color: var(--green); }
    .chip-red     { background: var(--red-bg);   color: var(--red);   }
    .chip-amber   { background: var(--amber-bg); color: var(--amber); }
    .chip-blue    { background: var(--blue-bg);  color: var(--blue);  }

    /* ── Pick cards (sportsbook tile) ── */
    .pick-list { display: flex; flex-direction: column; }
    .pick-tile {
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      transition: background .12s;
    }
    .pick-tile:last-child { border-bottom: none; }
    .pick-tile:hover { background: var(--surface2); }

    .pick-header {
      display: flex; align-items: center; justify-content: space-between; gap: 8px;
      margin-bottom: 10px;
    }
    .pick-matchup { font-size: 12px; color: var(--muted); font-weight: 500; }
    .pick-market  {
      font-size: 9px; font-weight: 800; text-transform: uppercase; letter-spacing: .6px;
      color: var(--subtle); background: var(--surface3); padding: 3px 7px; border-radius: 4px;
    }

    /* Two odds buttons */
    .odds-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }
    .odds-btn {
      background: var(--surface2); border: 1px solid var(--border2);
      border-radius: 8px; padding: 10px 12px; text-align: center;
    }
    .odds-btn.selected {
      background: var(--gold-bg); border-color: var(--gold);
    }
    .odds-team  { font-size: 11px; color: var(--muted); margin-bottom: 4px; font-weight: 500; }
    .odds-price { font-size: 19px; font-weight: 900; color: var(--text); letter-spacing: -.3px; }
    .odds-btn.selected .odds-team  { color: var(--gold); }
    .odds-btn.selected .odds-price { color: var(--gold2); }

    .pick-footer { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
    .pick-chips  { display: flex; gap: 5px; flex-wrap: wrap; }

    /* ── Stake input row ── */
    .stake-row {
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
      background: var(--surface2); border: 1px solid var(--border2);
      border-radius: 8px; padding: 9px 12px; margin-bottom: 10px;
    }
    .stake-rec { display: flex; flex-direction: column; gap: 2px; }
    .stake-rec-lbl { font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .6px; color: var(--muted); }
    .stake-rec-amt { font-size: 14px; font-weight: 900; color: var(--gold); letter-spacing: -.2px; }
    .stake-input-wrap {
      display: flex; align-items: center; gap: 4px;
      background: var(--surface3); border: 1px solid var(--border2);
      border-radius: 6px; padding: 0 8px; height: 34px;
      transition: border-color .15s;
    }
    .stake-input-wrap:focus-within { border-color: var(--gold); }
    .stake-prefix { font-size: 13px; font-weight: 700; color: var(--muted); }
    .stake-input {
      width: 72px; background: transparent; border: none; outline: none;
      color: var(--text); font-size: 15px; font-weight: 800; font-family: inherit;
      text-align: right; letter-spacing: -.2px;
    }
    .stake-input::-webkit-inner-spin-button { opacity: .4; }

    .log-btn {
      height: 28px; padding: 0 14px; border: 1px solid var(--gold); border-radius: 6px;
      background: transparent; color: var(--gold);
      font-size: 12px; font-weight: 700; font-family: inherit; cursor: pointer; transition: all .15s;
      flex-shrink: 0;
    }
    .log-btn:hover:not(:disabled) { background: var(--gold); color: #0b0d11; }
    .log-btn:disabled { opacity: .35; cursor: default; }
    .logged-badge {
      display: inline-flex; align-items: center; height: 28px; padding: 0 12px;
      border-radius: 6px; font-size: 12px; font-weight: 700;
      background: var(--green-bg); color: var(--green); flex-shrink: 0;
    }

    /* ── Logged today (sidebar) ── */
    .logged-list { display: flex; flex-direction: column; }
    .logged-row {
      padding: 11px 16px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
    }
    .logged-row:last-child { border-bottom: none; }
    .logged-team { font-size: 14px; font-weight: 800; color: var(--text); }
    .logged-meta { font-size: 11px; color: var(--muted); margin-top: 2px; }
    .logged-right { display: flex; flex-direction: column; align-items: flex-end; gap: 5px; flex-shrink: 0; }
    .logged-odds  { font-size: 17px; font-weight: 900; color: var(--gold); }

    /* ── Charts ── */
    .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    canvas { width: 100% !important; display: block; height: 180px; }

    /* ── Table ── */
    .tbl-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; min-width: 460px; }
    th, td { padding: 10px 16px; border-bottom: 1px solid var(--border); vertical-align: middle; text-align: left; }
    th { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .5px; color: var(--subtle); background: var(--surface2); }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: var(--surface2); }
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
        <button class="nav-btn active" id="tabOverview" onclick="switchTab('overview')">Overview</button>
        <button class="nav-btn" id="tabLive" onclick="switchTab('live')">
          Live Bets&nbsp;<span class="nav-badge" id="pendingBadge">0</span>
        </button>
        <button class="nav-btn" id="tabProps" onclick="switchTab('props')">Props</button>
        <button class="nav-btn" id="tabHistory" onclick="switchTab('history')">Bet History</button>
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
            <span class="card-title">Recent Paper Bets</span>
            <span class="chip chip-default" id="pendingPill">--</span>
          </div>
          <div class="tbl-wrap">
            <table>
              <thead><tr>
                <th>Date</th><th>Matchup</th><th>Side</th>
                <th>Odds</th><th>Edge</th><th>Stake</th><th>Result</th>
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
          <div class="card-hd">
            <span class="card-title">Logged Today</span>
            <span class="chip chip-default" id="loggedPill">0</span>
          </div>
          <div id="loggedToday"></div>
        </div>

        <div class="card">
          <div class="card-hd"><span class="card-title">Actions</span></div>
          <div class="suggest-list" id="suggestions"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ── Live Bets Tab ── -->
<div id="tab-live" style="display:none">
  <div class="shell">
    <div class="live-section" style="padding-top:18px">
      <div class="live-hdr">
        <div style="display:flex;align-items:center;gap:10px">
          <span style="font-size:17px;font-weight:800">Pending Bets</span>
          <span class="chip chip-default" id="liveUpdated">--</span>
        </div>
        <button class="ctl btn btn-ghost" onclick="loadLive()">Refresh</button>
      </div>
      <div id="liveBetsGrid" class="live-grid"></div>
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

<!-- ── History Tab ── -->
<div id="tab-history" style="display:none">
  <div class="shell">
    <div class="history-section" style="padding-top:18px">
      <div class="history-toolbar">
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
          <span style="font-size:17px;font-weight:800">Bet History</span>
          <div class="history-summary" id="historySummary"></div>
        </div>
        <div class="filter-row">
          <button class="filter-btn active" onclick="setFilter('all',this)">All</button>
          <button class="filter-btn" onclick="setFilter('Won',this)">Won</button>
          <button class="filter-btn" onclick="setFilter('Lost',this)">Lost</button>
          <button class="filter-btn" onclick="setFilter('Pending',this)">Pending</button>
          <button class="filter-btn" onclick="setFilter('ML',this)">ML</button>
          <button class="filter-btn" onclick="setFilter('F5',this)">F5</button>
          <button class="filter-btn" onclick="setFilter('Prop',this)">Props</button>
        </div>
      </div>
      <div id="slipGrid" class="slip-grid"></div>
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
let _activeTab = 'overview', _liveTimer = null;

function switchTab(tab) {
  _activeTab = tab;
  $('tab-overview').style.display = tab === 'overview' ? '' : 'none';
  $('tab-live').style.display     = tab === 'live'     ? '' : 'none';
  $('tab-props').style.display    = tab === 'props'    ? '' : 'none';
  $('tab-history').style.display  = tab === 'history'  ? '' : 'none';
  $('tabOverview').classList.toggle('active', tab === 'overview');
  $('tabLive').classList.toggle('active',     tab === 'live');
  $('tabProps').classList.toggle('active',    tab === 'props');
  $('tabHistory').classList.toggle('active',  tab === 'history');
  if (tab === 'live') {
    loadLive();
    if (!_liveTimer) _liveTimer = setInterval(loadLive, 60000);
  } else {
    if (_liveTimer) { clearInterval(_liveTimer); _liveTimer = null; }
  }
  if (tab === 'history') loadHistory();
  if (tab === 'props')   loadProps(false);
}

/* ─── Prop tiles ─── */
let _currentProps = [], _loggedPropKeys = new Set();

function propTile(p, idx) {
  const key      = p.game_pk ? `${p.game_pk}|${p.market}|${p.line}|${p.bet_side}` : null;
  const isLogged = key ? _loggedPropKeys.has(key) : false;
  const isOver   = p.bet_side === 'over';
  const bkChip   = p.bookmaker ? `<span class="chip chip-default" style="font-size:10px">${E(p.bookmaker)}</span>` : '';
  const expChip  = p.expected_k != null
    ? `<span class="chip chip-blue">Exp ${Number(p.expected_k).toFixed(1)} K</span>`
    : p.expected_hits != null
      ? `<span class="chip chip-blue">Exp ${Number(p.expected_hits).toFixed(2)} H</span>`
      : '';

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
      <span class="pick-matchup">${E(p.player_name)} &middot; <span style="color:var(--subtle)">${E(p.matchup)}</span></span>
      <span class="pick-market">K Props</span>
    </div>
    <div class="odds-grid">
      <div class="odds-btn ${!isOver ? 'selected' : ''}">
        <div class="odds-team">Under ${E(String(p.line))}</div>
        <div class="odds-price">${p.under_american != null ? (p.under_american > 0 ? '+' : '') + p.under_american : '--'}</div>
      </div>
      <div class="odds-btn ${isOver ? 'selected' : ''}">
        <div class="odds-team">Over ${E(String(p.line))}</div>
        <div class="odds-price">${p.over_american != null ? (p.over_american > 0 ? '+' : '') + p.over_american : '--'}</div>
      </div>
    </div>
    ${stakeRow}
    <div class="pick-footer">
      <div class="pick-chips">
        <span class="${edgeChipCls(p.edge)}">${pct(p.edge)} edge</span>
        <span class="chip chip-blue">Model ${pct(p.model_prob,false)}</span>
        ${expChip}${bkChip}
      </div>
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
      }),
    });
    const result = await res.json();
    if (result.ok) {
      const propKey = p.game_pk ? `${p.game_pk}|${p.market}|${p.line}|${p.bet_side}` : null;
      if (propKey) _loggedPropKeys.add(propKey);
      if (btn) btn.outerHTML = `<span class="logged-badge">&#10003; Logged</span>`;
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
  let data;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    $('propStrongPicks').innerHTML = emptyState('Error loading props: ' + err.message);
    return;
  }

  const {strong, watchlist, total_props, scored, error, fetched_at} = data;
  $('propsStatus').textContent = error ? 'Error' : `${scored}/${total_props} scored${data.cached?' · cached':''}`;
  $('propsStatus').className   = `chip ${error ? 'chip-red' : 'chip-default'}`;
  $('propsWatchlistThresh').textContent = Math.round(edge * 100);

  _loggedPropKeys = new Set(
    (_loggedPropsToday || []).filter(p=>p.game_pk).map(p=>`${p.game_pk}|${p.market}|${p.line}|${p.bet_side}`)
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

function pickTile(p, idx) {
  const key      = p.game_pk ? `${p.game_pk}|${p.bet_side}` : null;
  const isLogged = key ? _loggedKeys.has(key) : false;
  const isHome   = p.bet_side === 'home';
  const bkChip   = p.bookmaker ? `<span class="chip chip-default" style="font-size:10px">${E(p.bookmaker)}</span>` : '';

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
      <span class="pick-market">Moneyline</span>
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
    ${stakeRow}
    <div class="pick-footer">
      <div class="pick-chips">
        <span class="${edgeChipCls(p.edge)}">${pct(p.edge)} edge</span>
        <span class="chip chip-blue">Model ${pct(p.model_prob,false)}</span>
        ${bkChip}
      </div>
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

/* ─── Live bets ─── */
async function loadLive() {
  const grid = $('liveBetsGrid');
  grid.style.opacity = '.4';
  try {
    const res = await fetch('/api/live_bets');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const {bets, fetched_at} = await res.json();
    $('liveUpdated').textContent =
      'Updated ' + new Date(fetched_at).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
    $('pendingBadge').textContent = bets.length;
    grid.innerHTML = bets.length
      ? bets.map(liveBetCard).join('')
      : `<div class="live-empty">No pending bets right now.</div>`;
  } catch (err) {
    grid.innerHTML = `<div class="live-empty">Error: ${E(err.message)}</div>`;
  }
  grid.style.opacity = '1';
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

  /* Status strip */
  const fresh = f.latest_game_date === f.today;
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
  $('pendingBadge').textContent = m.pending;
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
        const rCls = r.outcome==='Won' ? 'chip chip-green' : r.outcome==='Lost' ? 'chip chip-red' : 'chip chip-default';
        return `<tr>
          <td style="color:var(--muted);font-size:12px">${E(r.game_date)}</td>
          <td>${E(r.matchup)}</td>
          <td style="font-weight:800">${E(r.side)}</td>
          <td style="color:var(--gold);font-weight:800">${E(r.odds)}</td>
          <td class="${posNeg(r.edge)}">${pct(r.edge)}</td>
          <td>${money(r.stake)}</td>
          <td><span class="${rCls}">${E(r.outcome)}</span></td>
        </tr>`;
      }).join('')
    : `<tr><td colspan="7" style="text-align:center;color:var(--subtle);padding:24px">No paper bets logged yet.</td></tr>`;

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

/* ─── Bet History ─── */
let _allBets = [], _historyFilter = 'all';

function setFilter(f, btn) {
  _historyFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderSlips();
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

function renderSlips() {
  const f = _historyFilter;
  const filtered = _allBets.filter(b => {
    if (f === 'all')   return true;
    if (f === 'ML' || f === 'F5' || f === 'Prop') return b.bet_type === f;
    return b.outcome === f;
  });
  $('slipGrid').innerHTML = filtered.length
    ? filtered.map(betSlip).join('')
    : `<div class="live-empty" style="grid-column:1/-1">No bets match this filter.</div>`;
}

async function loadHistory() {
  $('slipGrid').style.opacity = '.4';
  try {
    const res = await fetch('/api/bet_history');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const {bets, summary} = await res.json();
    _allBets = bets;

    const wr = summary.wins + summary.losses > 0
      ? ((summary.wins / (summary.wins + summary.losses)) * 100).toFixed(1) + '%'
      : '--';
    const pnlStr = (summary.pnl >= 0 ? '+' : '') + '$' + Math.abs(summary.pnl).toFixed(2);
    const pnlCls = summary.pnl >= 0 ? 'chip-green' : 'chip-red';

    $('historySummary').innerHTML =
      `<span class="chip chip-default">${summary.total} bets</span>` +
      `<span class="chip chip-green">${summary.wins}W</span>` +
      `<span class="chip chip-red">${summary.losses}L</span>` +
      (summary.pushes ? `<span class="chip chip-amber">${summary.pushes} Push</span>` : '') +
      (summary.pending ? `<span class="chip chip-default">${summary.pending} Pending</span>` : '') +
      `<span class="chip chip-default">${wr} win rate</span>` +
      `<span class="chip ${pnlCls}">${pnlStr} P&L</span>`;

    renderSlips();
  } catch (err) {
    $('slipGrid').innerHTML = `<div class="live-empty" style="grid-column:1/-1">Error: ${E(err.message)}</div>`;
  }
  $('slipGrid').style.opacity = '1';
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
        if parsed.path == "/api/props":
            qs = parse_qs(parsed.query)
            bankroll = _parse_float(qs.get("bankroll", [None])[0], DEFAULT_BANKROLL)
            min_edge = _parse_float(qs.get("min_edge", [None])[0], DEFAULT_MIN_EDGE)
            bankroll = max(1.0, bankroll)
            min_edge = min(max(0.0, min_edge), 1.0)
            refresh  = qs.get("refresh", ["0"])[0] == "1"
            self._send_json(_live_prop_recommendations(bankroll, min_edge, refresh))
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
            conn.execute(
                """INSERT INTO paper_bets
                   (game_pk, game_date, home_team, away_team, bet_side,
                    bet_american_odds, model_prob, fair_prob, edge, stake_dollars, bookmaker)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (game_pk, game_date, home, away, side,
                 odds_raw, m_prob, f_prob, edge, stake, bookmaker),
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

        with _connect() as conn:
            dup = conn.execute(
                "SELECT id FROM prop_bets WHERE game_pk=? AND player_name=? AND market=? AND bet_side=?",
                (game_pk or 0, player_name, market, side),
            ).fetchone()
            if dup:
                self._send_json({"ok": False, "error": "Already logged this prop bet."})
                return
            conn.execute(
                """INSERT INTO prop_bets
                   (game_pk, game_date, player_name, team, opponent, market, line,
                    bet_side, american_odds, decimal_odds, fair_prob, model_prob,
                    edge, stake_dollars, bankroll_at_bet, bookmaker, is_paper, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    game_pk or 0, today, player_name, home, away, market, line,
                    side, odds_raw,
                    (100 / abs(odds_raw) + 1) if odds_raw < 0 else (odds_raw / 100 + 1),
                    f_prob, m_prob, edge, stake, DEFAULT_BANKROLL, bookmaker, 1,
                    datetime.now().isoformat(),
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

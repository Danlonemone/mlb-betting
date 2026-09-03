"""
Compatibility shim — all odds data now comes from odds_client.py.
Replaces The Odds API (the-odds-api.com) with the Action Network scoreboard
API, which is free and requires no key.

All callers (daily_picks, log_odds_snapshot, capture_clv, etc.) import from
this module and require zero changes.
"""
from paper_trade.odds_client import (  # noqa: F401
    fetch_mlb_odds,
    parse_game_odds,
    fetch_today_odds,
    fetch_today_f5_odds,
    OddsAPIError,
    ODDS_API_TEAM_MAP,
    PREFERRED_BOOKS,
    _extract_h2h,
    _decimal,
)

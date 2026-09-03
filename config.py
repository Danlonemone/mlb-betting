from pathlib import Path
from dotenv import load_dotenv
import os
import json

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "mlb_betting.db"

load_dotenv(BASE_DIR / ".env")

# Legacy — only needed by ingestion/historical_odds.py for past backfills.
# Daily picks now use odds_client.py (Action Network API, no key required).
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# Seasons to pull for historical backtesting
HISTORICAL_SEASONS = list(range(2019, 2025))  # 2019–2024, skip 2020 (COVID 60-game season)
SKIP_SEASONS = {2020}

# To add 2025 data for live 2026 paper trading, run:
#   python ingestion/build_game_table.py --refresh-2025
# That pulls 2025 pitcher/team stats and 2025 game results into the DB.

# Minimum innings pitched for a starter to be included in the pitcher stats lookup
MIN_SP_IP = 30

# Kelly fraction. Reverted to 0.125 (eighth-Kelly) on 2026-06-10.
# History: doubled to 0.25 on 2026-06-09 based on ~3 settled bets; bankroll
# subsequently fell ~43% from peak. That change violated the project's own
# "assume I am fooling myself" principle.
#
# PRE-COMMITMENT RULE — do not change KELLY_FRACTION or MAX_BET_PCT until BOTH:
#   1. >= 200 settled paper bets, AND
#   2. mean CLV > 0 over those bets (closing line value, not ROI — ROI on a
#      small sample is noise).
# No mid-drawdown changes in either direction.
KELLY_FRACTION = 0.125

# Cap on total stake logged across ALL markets (ML + F5 + props) per day,
# as a fraction of bankroll. ML and F5 bets on the same game, or a K-prop on
# a starter in a game we also bet, are correlated — this cap limits how much
# of the bankroll one bad slate can take.
MAX_DAILY_EXPOSURE_PCT = 0.15

# Default paper-trading bankroll. Bet sizing is proportional to this value.
DEFAULT_BANKROLL = 71.34

# Skip bets where the market's vig-free probability is below this — extreme
# underdogs where the market encodes information the model can't see (injuries,
# bullpen, sharp money). Raised from 0.10 to 0.20 after model proved unable to
# generate probabilities below ~35% for home teams, making large edges on big
# underdogs artifacts rather than genuine signal.
MIN_MARKET_PROB = 0.20

# Minimum edge to recommend a bet (model prob - vig-free book prob).
#
# 2026-06-10 real-odds sweep (3,786 bets vs real closing lines): ROI is
# NEGATIVE at every threshold (-3% to -6%). The previous "15% maximises ROI"
# finding came from synthetic log5-priced bets and was an artifact. The
# model has no demonstrated edge against the closing line.
#
# Raised from 0.05 → 0.10 on 2026-06-24: paper trading analysis (36 settled
# bets) showed 5% picks had 0/9 positive CLV on favorites and higher model
# edge correlated with *worse* outcomes — classic noise amplification.
MIN_EDGE = 0.10

# Stricter edge required when betting a favorite (American odds < 0).
# Paper trading: favorites had mean CLV -2.09% with 0/9 positive CLV at the
# 5% threshold. Underdogs showed +1.54% mean CLV. The model systematically
# overstates probability on favorites relative to the closing line.
MIN_EDGE_FAVORITE = 0.15

# Hard cap on underdog odds. At +400 the market implies ~20% win probability,
# which matches MIN_MARKET_PROB. Belt-and-suspenders guard: logistic regression
# cannot reliably price outcomes this extreme.
MAX_AMERICAN_ODDS = 400


def get_current_bankroll() -> float:
    """Return live bankroll from settings.json, falling back to DEFAULT_BANKROLL."""
    settings_path = DATA_DIR / "settings.json"
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text())
            val = float(data.get("bankroll", 0))
            if val > 0:
                return val
        except Exception:
            pass
    return DEFAULT_BANKROLL

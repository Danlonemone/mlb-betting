from pathlib import Path
from dotenv import load_dotenv
import os
import json

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "mlb_betting.db"

load_dotenv(BASE_DIR / ".env")

ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# Seasons to pull for historical backtesting
HISTORICAL_SEASONS = list(range(2019, 2025))  # 2019–2024, skip 2020 (COVID 60-game season)
SKIP_SEASONS = {2020}

# To add 2025 data for live 2026 paper trading, run:
#   python ingestion/build_game_table.py --refresh-2025
# That pulls 2025 pitcher/team stats and 2025 game results into the DB.

# Minimum innings pitched for a starter to be included in the pitcher stats lookup
MIN_SP_IP = 30

# Kelly fraction. Reduced from 0.25 to 0.20 to hedge against model overconfidence
# (model predicts ~63% avg on bets it takes; actual win rate is ~51%).
KELLY_FRACTION = 0.20

# Default paper-trading bankroll. Bet sizing is proportional to this value.
DEFAULT_BANKROLL = 71.34

# Skip bets where the market's vig-free probability is below this — extreme
# underdogs where the market encodes information the model can't see (injuries,
# bullpen, sharp money). Raised from 0.10 to 0.20 after model proved unable to
# generate probabilities below ~35% for home teams, making large edges on big
# underdogs artifacts rather than genuine signal.
MIN_MARKET_PROB = 0.20

# Minimum edge to recommend a bet (model prob - vig-free book prob).
# Threshold sweep on 2022-2025 walk-forward: 15% maximises ROI (+12.9% overall,
# +14.8% on 2025) at a sample size still large enough to trust (168 bets).
MIN_EDGE = 0.15

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

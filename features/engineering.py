"""
Build the model feature matrix from the games table.

Design choices:
  - All features are *differentials* (home minus away) where the stat
    is symmetric. This keeps the model interpretable and forces it to
    learn relative quality, not absolute scale.
  - Missing SP data (~27% of rows) is filled with 0 for differential
    features (i.e. "no edge either way") and flagged with a boolean
    indicator so the model can learn to discount those games.
  - Team ERA/FIP features represent the full rotation + bullpen from
    the prior season, providing bullpen signal on top of the individual
    SP features.
  - Rolling team-form features use only games from prior calendar dates
    in the same season, avoiding look-ahead leakage and same-day leakage.
"""

import sys
import pandas as pd
import numpy as np
from collections import deque
from datetime import timedelta
from pathlib import Path
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.schema import get_engine, Game

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

# Raw SP stats we pull from the DB
SP_COLS = ["era", "fip", "k_pct", "bb_pct", "ip"]

# Feature names produced by this module
FEATURE_COLS = [
    # Starting pitcher differentials (home - away, so higher = home advantage)
    "sp_fip_diff",       # lower FIP = better; home-away so negative = home pitcher better
    "sp_k_pct_diff",     # higher K% = better
    "sp_bb_pct_diff",    # lower BB% = better; home-away so negative = home better
    "sp_era_diff",       # lower ERA = better

    # SP data availability flag (1 = both SPs have prior-season stats)
    "sp_data_available",

    # Team offense differentials (wOBA: higher = better offence)
    "woba_diff",

    # Team pitching / bullpen differentials (ERA: lower = better)
    "team_era_diff",
    "team_fip_diff",

    # Contextual
    "park_factor",        # absolute (100 = neutral)
    "rest_diff",          # home_rest - away_rest (positive = home more rested)

    # Current-season form known before the game date
    "season_win_pct_diff",
    "season_run_diff_pg_diff",
    "recent_win_pct_diff",
    "home_away_win_pct_diff",

    # Recent scoring form (last 7 games) — separates offensive from pitching form
    "recent_scored_pg_diff",   # avg runs scored per game: home − away
    "recent_allowed_pg_diff",  # avg runs allowed per game: home − away

    # SP recent form (last 5 starts this season, 2025+ only)
    "sp_recent_era_diff",      # recent ERA: home SP − away SP
    "sp_recent_k9_diff",       # recent K/9: home SP − away SP
    "sp_recent_form_data",     # 1.0 if both SPs have recent start data

    # Bullpen freshness (last 3 calendar days, 2025+ only)
    "bullpen_freshness_diff",  # away bullpen IP l3d − home bullpen IP l3d (positive = away pen more tired)
    "bullpen_data_available",  # 1.0 if both teams have bullpen data for this window
]

TARGET    = "home_win"
TARGET_F5 = "home_win_f5"

# F5 drops bullpen-related features — the starter pitches all 5 innings
_F5_EXCLUDE = {"team_era_diff", "team_fip_diff", "bullpen_freshness_diff", "bullpen_data_available"}
F5_FEATURE_COLS = [c for c in FEATURE_COLS if c not in _F5_EXCLUDE]


# ---------------------------------------------------------------------------
# SP recent form helpers
# ---------------------------------------------------------------------------

def load_game_log_cache(engine) -> dict:
    """
    Returns {mlbam_id: [(game_date, earned_runs, strikeouts, ip), ...]}
    sorted ascending by game_date for each pitcher.
    """
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT mlbam_id, game_date, earned_runs, strikeouts, ip "
            "FROM pitcher_game_logs ORDER BY mlbam_id, game_date"
        )).fetchall()
    cache: dict = {}
    for r in rows:
        cache.setdefault(int(r.mlbam_id), []).append((
            r.game_date,
            r.earned_runs or 0,
            r.strikeouts or 0,
            r.ip or 0.0,
        ))
    return cache


def recent_sp_stats(
    pid,
    before_date: str,
    cache: dict,
    n: int = 5,
    min_starts: int = 3,
) -> dict | None:
    """
    Returns {"era": float, "k9": float} from the last n starts strictly before
    before_date, or None if fewer than min_starts are available.
    """
    if not pid:
        return None
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return None
    starts = cache.get(pid_int)
    if not starts:
        return None
    prior = [(d, er, k, ip) for d, er, k, ip in starts if d < before_date]
    if len(prior) < min_starts:
        return None
    recent = prior[-n:]
    total_ip = sum(ip for _, _, _, ip in recent)
    if total_ip <= 0:
        return None
    return {
        "era": sum(er for _, er, _, _ in recent) * 9 / total_ip,
        "k9":  sum(k  for _, _, k, _ in recent) * 9 / total_ip,
    }


def build_bullpen_cache(engine, game_log_cache: dict) -> dict:
    """
    Returns {(team_abbr, game_date): bullpen_ip_used} for every game where
    the SP's game log entry exists. Bullpen IP ≈ 9.0 − starter_IP.
    Only populated for 2025+ games (where pitcher_game_logs is available).
    """
    # Flat lookup: {(mlbam_id, date): ip}
    start_ip_index: dict[tuple[int, str], float] = {
        (int(pid), d): ip
        for pid, starts in game_log_cache.items()
        for d, _er, _k, ip in starts
    }

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT game_date, home_team, away_team, "
            "home_sp_mlbam_id, away_sp_mlbam_id "
            "FROM games WHERE home_win IS NOT NULL"
        )).fetchall()

    cache: dict[tuple[str, str], float] = {}
    for g in rows:
        date = g.game_date
        if g.home_sp_mlbam_id:
            ip = start_ip_index.get((int(g.home_sp_mlbam_id), date))
            if ip is not None:
                cache[(g.home_team, date)] = max(0.0, 9.0 - ip)
        if g.away_sp_mlbam_id:
            ip = start_ip_index.get((int(g.away_sp_mlbam_id), date))
            if ip is not None:
                cache[(g.away_team, date)] = max(0.0, 9.0 - ip)
    return cache


def recent_bullpen_ip(
    team: str,
    before_date: str,
    bullpen_cache: dict,
    days: int = 3,
) -> float | None:
    """
    Sum bullpen IP for a team over the `days` calendar days strictly before
    before_date. Returns None if no data exists for any of those days.
    """
    date_obj = pd.Timestamp(before_date)
    total_ip = 0.0
    found = 0
    for d in range(1, days + 1):
        check_date = (date_obj - timedelta(days=d)).strftime("%Y-%m-%d")
        ip = bullpen_cache.get((team, check_date))
        if ip is not None:
            total_ip += ip
            found += 1
    return total_ip if found > 0 else None


def _add_bullpen_freshness(df: pd.DataFrame, feats: pd.DataFrame, bullpen_cache: dict) -> None:
    """Fill bullpen_freshness_diff and bullpen_data_available features."""
    feats["bullpen_freshness_diff"] = 0.0
    feats["bullpen_data_available"] = 0.0

    if not bullpen_cache:
        return

    for idx, row in df.iterrows():
        home_ip = recent_bullpen_ip(row["home_team"], row["game_date"], bullpen_cache)
        away_ip = recent_bullpen_ip(row["away_team"], row["game_date"], bullpen_cache)
        if home_ip is not None and away_ip is not None:
            feats.at[idx, "bullpen_freshness_diff"] = away_ip - home_ip
            feats.at[idx, "bullpen_data_available"] = 1.0


def _add_recent_sp_form(df: pd.DataFrame, feats: pd.DataFrame, cache: dict) -> None:
    """Fill sp_recent_* features using the pitcher game log cache."""
    feats["sp_recent_era_diff"]  = 0.0
    feats["sp_recent_k9_diff"]   = 0.0
    feats["sp_recent_form_data"] = 0.0

    if not cache:
        return

    for idx, row in df.iterrows():
        home_pid = row.get("home_sp_mlbam_id")
        away_pid = row.get("away_sp_mlbam_id")
        date     = row.get("game_date", "")
        h = recent_sp_stats(home_pid, date, cache)
        a = recent_sp_stats(away_pid, date, cache)
        if h and a:
            feats.at[idx, "sp_recent_era_diff"]  = h["era"] - a["era"]
            feats.at[idx, "sp_recent_k9_diff"]   = h["k9"]  - a["k9"]
            feats.at[idx, "sp_recent_form_data"]  = 1.0


# ---------------------------------------------------------------------------
# Load raw game rows from SQLite
# ---------------------------------------------------------------------------

def load_games(
    seasons: list[int] | None = None,
    settled_only: bool = True,
    before_date: str | None = None,
) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        q = "SELECT * FROM games"
        filters = []
        if settled_only:
            filters.append("home_win IS NOT NULL")
        if seasons:
            placeholders = ",".join(str(s) for s in seasons)
            filters.append(f"season IN ({placeholders})")
        if before_date:
            filters.append("game_date < :before_date")
        if filters:
            q += " WHERE " + " AND ".join(filters)
        params = {"before_date": before_date} if before_date else None
        df = pd.read_sql(text(q), conn, params=params)
    return df


# ---------------------------------------------------------------------------
# Rolling team form
# ---------------------------------------------------------------------------

def _blank_team_state() -> dict:
    return {
        "games": 0,
        "wins": 0,
        "runs_for": 0,
        "runs_against": 0,
        "home_games": 0,
        "home_wins": 0,
        "away_games": 0,
        "away_wins": 0,
        "recent": deque(maxlen=10),
        "recent_scored": deque(maxlen=7),
        "recent_allowed": deque(maxlen=7),
    }


def _shrunken_pct(wins: int, games: int, prior_games: int = 10) -> float:
    """
    Beta-style shrinkage toward .500 so April samples do not dominate.
    prior_games=10 means 0-0 starts at .500 and early records move gradually.
    """
    return (wins + 0.5 * prior_games) / (games + prior_games)


def _team_values(state: dict, side: str) -> dict:
    games = state["games"]
    run_diff_pg = (
        (state["runs_for"] - state["runs_against"]) / games
        if games
        else 0.0
    )
    recent_games = len(state["recent"])
    recent_wins = sum(state["recent"])

    if side == "home":
        split_wins = state["home_wins"]
        split_games = state["home_games"]
    else:
        split_wins = state["away_wins"]
        split_games = state["away_games"]

    n_scored  = len(state["recent_scored"])
    n_allowed = len(state["recent_allowed"])
    recent_scored_pg  = sum(state["recent_scored"])  / n_scored  if n_scored  else 4.5
    recent_allowed_pg = sum(state["recent_allowed"]) / n_allowed if n_allowed else 4.5

    return {
        "season_win_pct": _shrunken_pct(state["wins"], games),
        "season_run_diff_pg": run_diff_pg,
        "recent_win_pct": _shrunken_pct(recent_wins, recent_games, prior_games=6),
        "split_win_pct": _shrunken_pct(split_wins, split_games, prior_games=8),
        "recent_scored_pg": recent_scored_pg,
        "recent_allowed_pg": recent_allowed_pg,
    }


def _add_rolling_team_form(df: pd.DataFrame, feats: pd.DataFrame) -> None:
    """
    Add current-season team form using only games from prior calendar dates.

    We process full date groups before updating state so doubleheaders and
    other same-day games cannot leak into each other.
    """
    defaults = {
        "season_win_pct_diff": 0.0,
        "season_run_diff_pg_diff": 0.0,
        "recent_win_pct_diff": 0.0,
        "home_away_win_pct_diff": 0.0,
        "recent_scored_pg_diff": 0.0,
        "recent_allowed_pg_diff": 0.0,
    }
    for col, val in defaults.items():
        feats[col] = val

    required = {
        "season", "game_date", "home_team", "away_team",
        "home_score", "away_score", "home_win",
    }
    if not required.issubset(df.columns):
        return

    work = df[
        [
            "season", "game_date", "home_team", "away_team",
            "home_score", "away_score", "home_win",
        ]
    ].copy()
    work["_idx"] = df.index
    work = work.sort_values(["season", "game_date", "_idx"])

    for _, season_games in work.groupby("season", sort=True):
        states: dict[str, dict] = {}

        for _, date_games in season_games.groupby("game_date", sort=True):
            for _, row in date_games.iterrows():
                home_state = states.setdefault(row["home_team"], _blank_team_state())
                away_state = states.setdefault(row["away_team"], _blank_team_state())

                home_vals = _team_values(home_state, side="home")
                away_vals = _team_values(away_state, side="away")

                idx = row["_idx"]
                feats.at[idx, "season_win_pct_diff"] = (
                    home_vals["season_win_pct"] - away_vals["season_win_pct"]
                )
                feats.at[idx, "season_run_diff_pg_diff"] = (
                    home_vals["season_run_diff_pg"] - away_vals["season_run_diff_pg"]
                )
                feats.at[idx, "recent_win_pct_diff"] = (
                    home_vals["recent_win_pct"] - away_vals["recent_win_pct"]
                )
                feats.at[idx, "home_away_win_pct_diff"] = (
                    home_vals["split_win_pct"] - away_vals["split_win_pct"]
                )
                feats.at[idx, "recent_scored_pg_diff"] = (
                    home_vals["recent_scored_pg"] - away_vals["recent_scored_pg"]
                )
                feats.at[idx, "recent_allowed_pg_diff"] = (
                    home_vals["recent_allowed_pg"] - away_vals["recent_allowed_pg"]
                )

            # Update state only after every game for the date has received
            # features. This preserves the morning-line assumption.
            for _, row in date_games.iterrows():
                if pd.isna(row["home_score"]) or pd.isna(row["away_score"]) or pd.isna(row["home_win"]):
                    continue

                home_state = states.setdefault(row["home_team"], _blank_team_state())
                away_state = states.setdefault(row["away_team"], _blank_team_state())
                home_win = int(row["home_win"])
                away_win = 1 - home_win

                home_state["games"] += 1
                home_state["wins"] += home_win
                home_state["runs_for"] += int(row["home_score"])
                home_state["runs_against"] += int(row["away_score"])
                home_state["home_games"] += 1
                home_state["home_wins"] += home_win
                home_state["recent"].append(home_win)
                home_state["recent_scored"].append(int(row["home_score"]))
                home_state["recent_allowed"].append(int(row["away_score"]))

                away_state["games"] += 1
                away_state["wins"] += away_win
                away_state["runs_for"] += int(row["away_score"])
                away_state["runs_against"] += int(row["home_score"])
                away_state["away_games"] += 1
                away_state["away_wins"] += away_win
                away_state["recent"].append(away_win)
                away_state["recent_scored"].append(int(row["away_score"]))
                away_state["recent_allowed"].append(int(row["home_score"]))


# ---------------------------------------------------------------------------
# Build the feature matrix
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    game_log_cache: dict | None = None,
    bullpen_cache: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (X, y) where X is a DataFrame of FEATURE_COLS and y is home_win.
    """
    feats = pd.DataFrame(index=df.index)

    # --- SP differentials ---
    # FIP: lower is better for the pitcher.
    # home_sp_fip - away_sp_fip: negative means home SP has lower (better) FIP.
    feats["sp_fip_diff"]   = df["home_sp_fip"]   - df["away_sp_fip"]
    feats["sp_k_pct_diff"] = df["home_sp_k_pct"] - df["away_sp_k_pct"]
    feats["sp_bb_pct_diff"]= df["home_sp_bb_pct"]- df["away_sp_bb_pct"]
    feats["sp_era_diff"]   = df["home_sp_era"]    - df["away_sp_era"]

    # Flag rows where both SPs have prior-season data
    feats["sp_data_available"] = (
        df["home_sp_fip"].notna() & df["away_sp_fip"].notna()
    ).astype(float)

    # Fill missing SP differentials with 0 (no modelled edge either way)
    for col in ["sp_fip_diff", "sp_k_pct_diff", "sp_bb_pct_diff", "sp_era_diff"]:
        feats[col] = feats[col].fillna(0.0)

    # --- Team offense ---
    feats["woba_diff"] = df["home_woba"] - df["away_woba"]
    feats["woba_diff"] = feats["woba_diff"].fillna(0.0)

    # --- Team pitching / bullpen ---
    feats["team_era_diff"] = df["home_team_era"] - df["away_team_era"]
    feats["team_fip_diff"] = df["home_team_fip"] - df["away_team_fip"]
    feats["team_era_diff"] = feats["team_era_diff"].fillna(0.0)
    feats["team_fip_diff"] = feats["team_fip_diff"].fillna(0.0)

    # --- Contextual ---
    feats["park_factor"] = df["park_factor"].fillna(100.0)
    feats["rest_diff"]   = (
        df["home_rest_days"].fillna(1) - df["away_rest_days"].fillna(1)
    )

    # --- Current-season team form ---
    _add_rolling_team_form(df, feats)

    # --- SP recent form (2025+ only; 0 for historical games without logs) ---
    _add_recent_sp_form(df, feats, game_log_cache or {})

    # --- Bullpen freshness (2025+ only) ---
    _add_bullpen_freshness(df, feats, bullpen_cache or {})

    # Sanity check column order
    feats = feats[FEATURE_COLS]
    y = df[TARGET].astype(int)

    return feats, y


def load_f5_feature_matrix(
    seasons: list[int] | None = None,
    before_date: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Returns (X, y, meta) for the F5 model.
    Only includes games with a non-tie F5 outcome (home_win_f5 IS NOT NULL).
    """
    engine = get_engine()
    game_log_cache = load_game_log_cache(engine)
    bullpen_cache  = build_bullpen_cache(engine, game_log_cache)

    with engine.connect() as conn:
        q = "SELECT * FROM games WHERE home_win_f5 IS NOT NULL"
        filters = []
        if seasons:
            placeholders = ",".join(str(s) for s in seasons)
            filters.append(f"season IN ({placeholders})")
        if before_date:
            filters.append("game_date < :before_date")
        if filters:
            q += " AND " + " AND ".join(filters)
        params = {"before_date": before_date} if before_date else None
        df = pd.read_sql(text(q), conn, params=params)

    feats, _ = build_features(df, game_log_cache=game_log_cache, bullpen_cache=bullpen_cache)
    feats = feats[F5_FEATURE_COLS]
    y    = df[TARGET_F5].astype(int)
    meta = df[["game_pk", "game_date", "season", "home_team", "away_team"]].copy()
    return feats, y, meta


def load_feature_matrix(
    seasons: list[int] | None = None,
    before_date: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Returns (X, y, meta) where meta has game_pk, game_date, season for
    joining predictions back to games.
    """
    engine = get_engine()
    game_log_cache = load_game_log_cache(engine)
    bullpen_cache  = build_bullpen_cache(engine, game_log_cache)
    df = load_games(seasons=seasons, settled_only=True, before_date=before_date)
    X, y = build_features(df, game_log_cache=game_log_cache, bullpen_cache=bullpen_cache)
    meta = df[["game_pk", "game_date", "season", "home_team", "away_team"]].copy()
    return X, y, meta

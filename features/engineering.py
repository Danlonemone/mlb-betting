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
# Elo rating parameters
# ---------------------------------------------------------------------------

_ELO_BASE    = 1500.0   # starting rating for any team with no history
_ELO_K       = 20.0     # update sensitivity per game (standard for baseball)
_ELO_HOME    = 35.0     # home-field bonus in Elo points (≈ +2.4% win prob)
_ELO_REGRESS = 0.33     # fraction regressed toward mean at each season start


def _elo_expected(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(home_elo - away_elo + _ELO_HOME) / 400.0))


def _elo_sequence(rows) -> tuple[dict[int, float], dict[str, float]]:
    """
    Core Elo engine. Processes namedtuple-like rows sorted by (season, game_date).

    Returns:
      game_elo   : {game_pk: home_elo_pre - away_elo_pre}  (pre-game diff, no leakage)
      ratings    : {team: final_elo}  (end state after all rows)
    """
    ratings: dict[str, float] = {}
    game_elo: dict[int, float] = {}
    prev_season: int | None = None

    i, n = 0, len(rows)
    while i < n:
        season = int(rows[i].season)

        # Season-start regression to the mean
        if prev_season is not None and season != prev_season:
            for team in list(ratings):
                ratings[team] = _ELO_BASE + (1 - _ELO_REGRESS) * (ratings[team] - _ELO_BASE)
        prev_season = season

        # Collect all games for this (season, game_date) — flush together
        # so doubleheaders can't leak into each other.
        j = i
        while j < n and int(rows[j].season) == season and rows[j].game_date == rows[i].game_date:
            j += 1
        day = rows[i:j]

        # Assign pre-game Elo to each game in the group BEFORE updating any rating
        for row in day:
            h = ratings.get(row.home_team, _ELO_BASE)
            a = ratings.get(row.away_team, _ELO_BASE)
            if row.game_pk:
                game_elo[row.game_pk] = h - a

        # Update ratings based on results
        for row in day:
            home, away = row.home_team, row.away_team
            h = ratings.get(home, _ELO_BASE)
            a = ratings.get(away, _ELO_BASE)
            expected = _elo_expected(h, a)
            delta = _ELO_K * (float(row.home_win) - expected)
            ratings[home] = h + delta
            ratings[away] = a - delta

        i = j

    return game_elo, ratings


def build_elo_cache(engine) -> dict[int, float]:
    """
    Returns {game_pk: home_elo_pre - away_elo_pre} for every settled game.
    Used for historical feature matrix construction.
    """
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT game_pk, season, game_date, home_team, away_team, home_win "
            "FROM games WHERE home_win IS NOT NULL "
            "ORDER BY season, game_date, game_pk"
        )).fetchall()
    game_elo, _ = _elo_sequence(rows)
    return game_elo


def build_ump_run_cache(engine, min_games: int = 10) -> dict[int, float]:
    """
    Returns {game_pk: prior_runs_vs_avg} for historical feature rows.

    The value is computed from games before the target game date only. That
    keeps walk-forward training honest; a career aggregate from the full DB
    would leak future run environments into older games.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT u.game_pk,
                   u.ump_name,
                   g.season,
                   g.game_date,
                   g.home_score,
                   g.away_score
            FROM   game_umpires u
            JOIN   games g ON u.game_pk = g.game_pk
            WHERE  g.home_score IS NOT NULL
               AND g.away_score IS NOT NULL
               AND u.ump_name != ''
            ORDER  BY g.season, g.game_date, u.game_pk
        """)).fetchall()

    cache: dict[int, float] = {}
    ump_state: dict[str, dict[str, float]] = {}
    league_games = 0
    league_runs = 0.0

    i, n = 0, len(rows)
    while i < n:
        date = rows[i].game_date
        j = i
        while j < n and rows[j].game_date == date:
            j += 1
        day = rows[i:j]

        league_avg = league_runs / league_games if league_games else 9.0
        for row in day:
            state = ump_state.get(row.ump_name)
            if state and state["games"] >= min_games:
                ump_avg = state["runs"] / state["games"]
                cache[int(row.game_pk)] = float(ump_avg - league_avg)

        for row in day:
            total_runs = float(row.home_score or 0) + float(row.away_score or 0)
            state = ump_state.setdefault(row.ump_name, {"games": 0, "runs": 0.0})
            state["games"] += 1
            state["runs"] += total_runs
            league_games += 1
            league_runs += total_runs

        i = j

    return cache


def build_live_elo_ratings(engine, before_date: str) -> dict[str, float]:
    """
    Returns {team: current_elo} as of before_date.
    Used for today's live pick features.
    """
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT game_pk, season, game_date, home_team, away_team, home_win "
            "FROM games WHERE home_win IS NOT NULL AND game_date < :d "
            "ORDER BY season, game_date, game_pk"
        ), {"d": before_date}).fetchall()
    _, ratings = _elo_sequence(rows)
    return ratings

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

    # Bullpen quality (prior-season reliever ERA/FIP, separate from full team pitching)
    "bullpen_era_diff",        # home bullpen ERA − away bullpen ERA (lower ERA = better)
    "bullpen_fip_diff",        # home bullpen FIP − away bullpen FIP

    # Elo rating differential (home - away, pre-game)
    # Dynamic team strength updated game-by-game from 2019 onward.
    # Regressed 33% toward the mean at each season start.
    "elo_diff",

    # Home plate umpire run-scoring tendency (career avg runs/game vs league avg)
    # 0 when ump assignment is unavailable for the game.
    "ump_runs_vs_avg",
    "ump_data_available",
]

TARGET    = "home_win"
TARGET_F5 = "home_win_f5"

# F5 drops bullpen-related features — it should mostly isolate starters and
# early-game team context.
_F5_EXCLUDE = {
    "team_era_diff",
    "team_fip_diff",
    "bullpen_era_diff",
    "bullpen_fip_diff",
    "bullpen_freshness_diff",
    "bullpen_data_available",
}
F5_FEATURE_COLS = [c for c in FEATURE_COLS if c not in _F5_EXCLUDE]


# ---------------------------------------------------------------------------
# wOBA helpers — current-season batter logs
# ---------------------------------------------------------------------------

# 2024 FanGraphs linear weights (denominator ≈ PA)
_WOBA_BB  = 0.690
_WOBA_1B  = 0.888
_WOBA_2B  = 1.271
_WOBA_3B  = 1.616
_WOBA_HR  = 2.101


def _woba_from_totals(bb: float, singles: float, doubles: float,
                      triples: float, hr: float, pa: float) -> float | None:
    if pa <= 0:
        return None
    return (
        _WOBA_BB * bb +
        _WOBA_1B * singles +
        _WOBA_2B * doubles +
        _WOBA_3B * triples +
        _WOBA_HR * hr
    ) / pa


def build_player_woba_cache(
    engine,
    before_date: str,
    min_pa: int = 30,
) -> dict[int, float]:
    """
    Returns {mlbam_id: woba} for batters with at least min_pa PA in the current
    season strictly before before_date.

    Used for live lineup-weighted wOBA when today's batting order is confirmed.
    """
    season = int(before_date[:4])
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT mlbam_id, "
            "SUM(pa) AS pa, SUM(hits) AS h, SUM(doubles) AS d2, "
            "SUM(triples) AS d3, SUM(home_runs) AS hr, SUM(walks) AS bb "
            "FROM batter_game_logs "
            "WHERE season = :s AND game_date < :d AND pa > 0 "
            "GROUP BY mlbam_id "
            "HAVING SUM(pa) >= :min_pa"
        ), {"s": season, "d": before_date, "min_pa": min_pa}).fetchall()

    cache: dict[int, float] = {}
    for r in rows:
        singles = (r.h or 0) - (r.d2 or 0) - (r.d3 or 0) - (r.hr or 0)
        w = _woba_from_totals(r.bb or 0, max(0, singles),
                              r.d2 or 0, r.d3 or 0, r.hr or 0, r.pa or 0)
        if w is not None:
            cache[int(r.mlbam_id)] = round(w, 4)
    return cache


def build_current_team_woba_cache(
    engine,
    before_date: str,
    min_pa: int = 100,
) -> dict[str, float]:
    """
    Returns {team_abbr: woba} for all teams, aggregated from qualified batters
    (min_pa cumulative PA) strictly before before_date.

    Derives team from each player's most recent game_pk + home_away via
    a join to the games table, since batter_game_logs.team is not reliably set.

    Used for live picks when no confirmed lineup is available.
    """
    season = int(before_date[:4])
    with engine.connect() as conn:
        # Step 1: each player's season-to-date batting totals
        totals = conn.execute(text("""
            SELECT mlbam_id,
                   SUM(pa) AS pa, SUM(hits) AS h, SUM(doubles) AS d2,
                   SUM(triples) AS d3, SUM(home_runs) AS hr, SUM(walks) AS bb
            FROM batter_game_logs
            WHERE season = :s AND game_date < :d AND pa > 0
            GROUP BY mlbam_id
            HAVING SUM(pa) >= :min_pa
        """), {"s": season, "d": before_date, "min_pa": min_pa}).fetchall()

        if not totals:
            return {}

        qualified_ids = [int(r.mlbam_id) for r in totals]
        placeholder   = ",".join(str(i) for i in qualified_ids)

        # Step 2: most recent (game_pk, home_away) per player to identify current team
        recent = conn.execute(text(f"""
            SELECT b.mlbam_id, b.game_pk, b.home_away
            FROM batter_game_logs b
            JOIN (
                SELECT mlbam_id, MAX(game_date) AS last_date
                FROM batter_game_logs
                WHERE season = :s AND game_date < :d AND pa > 0
                  AND mlbam_id IN ({placeholder})
                GROUP BY mlbam_id
            ) lat ON b.mlbam_id = lat.mlbam_id AND b.game_date = lat.last_date
            WHERE b.season = :s
            GROUP BY b.mlbam_id
        """), {"s": season, "d": before_date}).fetchall()

        # Step 3: resolve game_pk → home/away team
        pks = list({int(r.game_pk) for r in recent if r.game_pk})
        if not pks:
            return {}
        pk_placeholder = ",".join(str(p) for p in pks)
        game_rows = conn.execute(text(
            f"SELECT game_pk, home_team, away_team FROM games WHERE game_pk IN ({pk_placeholder})"
        )).fetchall()

    pk_to_teams = {int(r.game_pk): (r.home_team, r.away_team) for r in game_rows}
    pid_to_team = {}
    for r in recent:
        teams = pk_to_teams.get(int(r.game_pk) if r.game_pk else 0)
        if teams:
            pid_to_team[int(r.mlbam_id)] = teams[0] if r.home_away == "home" else teams[1]

    # Step 4: aggregate qualified batters by team
    team_stats: dict[str, dict] = {}
    for r in totals:
        team = pid_to_team.get(int(r.mlbam_id))
        if not team:
            continue
        if team not in team_stats:
            team_stats[team] = {"pa": 0, "bb": 0, "1b": 0, "2b": 0, "3b": 0, "hr": 0}
        singles = (r.h or 0) - (r.d2 or 0) - (r.d3 or 0) - (r.hr or 0)
        st = team_stats[team]
        st["pa"] += r.pa or 0
        st["bb"] += r.bb or 0
        st["1b"] += max(0, singles)
        st["2b"] += r.d2 or 0
        st["3b"] += r.d3 or 0
        st["hr"] += r.hr or 0

    cache: dict[str, float] = {}
    for team, st in team_stats.items():
        w = _woba_from_totals(st["bb"], st["1b"], st["2b"], st["3b"], st["hr"], st["pa"])
        if w is not None:
            cache[team] = round(w, 4)
    return cache


def build_rolling_team_woba_cache(engine, min_pa: int = 100) -> dict[tuple[str, str], float]:
    """
    Returns {(team_abbr, game_date): woba} for historical training.

    Uses batter_game_logs joined to games (for reliable team assignment) to
    compute each team's offensive wOBA from qualified batters (min_pa cumulative
    PA) strictly before each game date. Only populated for seasons where
    batter_game_logs has data (2023+).

    Look-ahead-safe: stats from date D are only used for games on dates > D.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                b.game_date,
                CASE WHEN b.home_away = 'home' THEN g.home_team ELSE g.away_team END AS team,
                b.mlbam_id,
                b.season,
                b.pa,
                b.hits,
                b.doubles,
                b.triples,
                b.home_runs,
                b.walks
            FROM batter_game_logs b
            JOIN games g ON b.game_pk = g.game_pk
            WHERE b.pa > 0
            ORDER BY b.season, b.game_date
        """)).fetchall()

    if not rows:
        return {}

    cum: dict[tuple, dict] = {}   # {(season, team, mlbam_id): stats}
    cache: dict[tuple[str, str], float] = {}
    prev_season: int | None = None

    i = 0
    n = len(rows)
    while i < n:
        season = int(rows[i].season)
        date   = rows[i].game_date

        if season != prev_season:
            cum = {}
            prev_season = season

        j = i
        while j < n and rows[j].game_date == date and int(rows[j].season) == season:
            j += 1
        day = rows[i:j]

        # Compute team wOBA from stats accumulated BEFORE today
        teams_today = {r.team for r in day if r.team}
        for team in teams_today:
            total_pa = total_bb = total_1b = total_2b = total_3b = total_hr = 0
            for (s, t, _), st in cum.items():
                if s == season and t == team and st["pa"] >= min_pa:
                    total_pa += st["pa"]
                    total_bb += st["bb"]
                    total_1b += st["1b"]
                    total_2b += st["2b"]
                    total_3b += st["3b"]
                    total_hr += st["hr"]
            w = _woba_from_totals(total_bb, total_1b, total_2b, total_3b, total_hr, total_pa)
            if w is not None:
                cache[(team, date)] = round(w, 4)

        # Update cumulative totals with today's games
        for r in day:
            if not r.team:
                continue
            key = (season, r.team, r.mlbam_id)
            if key not in cum:
                cum[key] = {"pa": 0, "bb": 0, "1b": 0, "2b": 0, "3b": 0, "hr": 0}
            singles = (r.hits or 0) - (r.doubles or 0) - (r.triples or 0) - (r.home_runs or 0)
            st = cum[key]
            st["pa"] += r.pa or 0
            st["bb"] += r.walks or 0
            st["1b"] += max(0, singles)
            st["2b"] += r.doubles or 0
            st["3b"] += r.triples or 0
            st["hr"] += r.home_runs or 0

        i = j

    return cache


def lineup_woba(
    player_ids: list[int],
    player_woba_cache: dict[int, float],
    min_players: int = 5,
) -> float | None:
    """
    Compute average wOBA across a confirmed batting order.
    Returns None if fewer than min_players have wOBA data (insufficient season history).
    """
    wobas = [player_woba_cache[pid] for pid in player_ids if pid in player_woba_cache]
    if len(wobas) < min_players:
        return None
    return round(sum(wobas) / len(wobas), 4)


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

    home_ips = [recent_bullpen_ip(t, d, bullpen_cache) for t, d in zip(df["home_team"], df["game_date"])]
    away_ips = [recent_bullpen_ip(t, d, bullpen_cache) for t, d in zip(df["away_team"], df["game_date"])]
    for i, (h, a) in enumerate(zip(home_ips, away_ips)):
        if h is not None and a is not None:
            feats.iloc[i, feats.columns.get_loc("bullpen_freshness_diff")] = a - h
            feats.iloc[i, feats.columns.get_loc("bullpen_data_available")] = 1.0


def load_team_bullpen_cache(engine) -> dict:
    """
    Returns {(team_abbr, season): {"era": float, "fip": float}}
    for all rows in team_seasons with stat_type='bullpen'.
    """
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT team_abbr, season, era, fip FROM team_seasons WHERE stat_type = 'bullpen'"
        )).fetchall()
    return {(r[0], int(r[1])): {"era": r[2], "fip": r[3]} for r in rows}


def _add_bullpen_quality(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    team_bullpen_cache: dict,
) -> None:
    """Fill bullpen_era_diff and bullpen_fip_diff from prior-season reliever stats."""
    feats["bullpen_era_diff"] = 0.0
    feats["bullpen_fip_diff"] = 0.0

    if not team_bullpen_cache:
        return

    prior = df["season"].astype(int) - 1
    home_keys = list(zip(df["home_team"], prior))
    away_keys = list(zip(df["away_team"], prior))
    h_era = pd.array([(team_bullpen_cache.get(k) or {}).get("era") or 0.0 for k in home_keys])
    a_era = pd.array([(team_bullpen_cache.get(k) or {}).get("era") or 0.0 for k in away_keys])
    h_fip = pd.array([(team_bullpen_cache.get(k) or {}).get("fip") or 0.0 for k in home_keys])
    a_fip = pd.array([(team_bullpen_cache.get(k) or {}).get("fip") or 0.0 for k in away_keys])
    feats["bullpen_era_diff"] = h_era - a_era
    feats["bullpen_fip_diff"] = h_fip - a_fip


def _add_recent_sp_form(df: pd.DataFrame, feats: pd.DataFrame, cache: dict) -> None:
    """Fill sp_recent_* features using the pitcher game log cache."""
    feats["sp_recent_era_diff"]  = 0.0
    feats["sp_recent_k9_diff"]   = 0.0
    feats["sp_recent_form_data"] = 0.0

    if not cache:
        return

    for i, (hpid, apid, date) in enumerate(zip(
        df.get("home_sp_mlbam_id", [None]*len(df)),
        df.get("away_sp_mlbam_id", [None]*len(df)),
        df.get("game_date", [""]*len(df)),
    )):
        h = recent_sp_stats(hpid, date, cache)
        a = recent_sp_stats(apid, date, cache)
        if h and a:
            feats.iloc[i, feats.columns.get_loc("sp_recent_era_diff")]  = h["era"] - a["era"]
            feats.iloc[i, feats.columns.get_loc("sp_recent_k9_diff")]   = h["k9"]  - a["k9"]
            feats.iloc[i, feats.columns.get_loc("sp_recent_form_data")] = 1.0


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


def _add_ump_features(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    ump_cache: dict,
) -> None:
    """Fill ump_runs_vs_avg and ump_data_available from the game_pk lookup."""
    if ump_cache and "game_pk" in df.columns:
        feats["ump_runs_vs_avg"]   = df["game_pk"].map(ump_cache).fillna(0.0)
        feats["ump_data_available"] = df["game_pk"].isin(ump_cache).astype(float)
    else:
        feats["ump_runs_vs_avg"]   = 0.0
        feats["ump_data_available"] = 0.0


# ---------------------------------------------------------------------------
# Build the feature matrix
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    game_log_cache: dict | None = None,
    bullpen_cache: dict | None = None,
    team_bullpen_cache: dict | None = None,
    elo_cache: dict | None = None,
    ump_cache: dict | None = None,
    team_woba_cache: dict | None = None,
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
    # Use rolling current-season wOBA from batter_game_logs when available
    # (2023+). Falls back to prior-season team average for earlier seasons.
    if team_woba_cache and "home_team" in df.columns and "game_date" in df.columns:
        home_cur = pd.Series(
            [team_woba_cache.get((t, d)) for t, d in zip(df["home_team"], df["game_date"])],
            index=df.index, dtype=float,
        )
        away_cur = pd.Series(
            [team_woba_cache.get((t, d)) for t, d in zip(df["away_team"], df["game_date"])],
            index=df.index, dtype=float,
        )
        home_woba = home_cur.combine_first(df["home_woba"])
        away_woba = away_cur.combine_first(df["away_woba"])
    else:
        home_woba = df["home_woba"]
        away_woba = df["away_woba"]
    feats["woba_diff"] = (home_woba - away_woba).fillna(0.0)

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

    # --- Bullpen quality (prior-season reliever ERA/FIP) ---
    _add_bullpen_quality(df, feats, team_bullpen_cache or {})

    # --- Elo rating differential ---
    if elo_cache and "game_pk" in df.columns:
        feats["elo_diff"] = df["game_pk"].map(elo_cache).fillna(0.0)
    else:
        feats["elo_diff"] = 0.0

    # --- Umpire run-scoring tendency ---
    _add_ump_features(df, feats, ump_cache or {})

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
    game_log_cache     = load_game_log_cache(engine)
    bullpen_cache      = build_bullpen_cache(engine, game_log_cache)
    team_bullpen_cache = load_team_bullpen_cache(engine)
    elo_cache          = build_elo_cache(engine)
    ump_cache          = build_ump_run_cache(engine)
    team_woba_cache    = build_rolling_team_woba_cache(engine)

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

    feats, _ = build_features(df, game_log_cache=game_log_cache, bullpen_cache=bullpen_cache,
                               team_bullpen_cache=team_bullpen_cache, elo_cache=elo_cache,
                               ump_cache=ump_cache, team_woba_cache=team_woba_cache)
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
    game_log_cache     = load_game_log_cache(engine)
    bullpen_cache      = build_bullpen_cache(engine, game_log_cache)
    team_bullpen_cache = load_team_bullpen_cache(engine)
    elo_cache          = build_elo_cache(engine)
    ump_cache          = build_ump_run_cache(engine)
    team_woba_cache    = build_rolling_team_woba_cache(engine)
    df = load_games(seasons=seasons, settled_only=True, before_date=before_date)
    X, y = build_features(df, game_log_cache=game_log_cache, bullpen_cache=bullpen_cache,
                           team_bullpen_cache=team_bullpen_cache, elo_cache=elo_cache,
                           ump_cache=ump_cache, team_woba_cache=team_woba_cache)
    meta = df[["game_pk", "game_date", "season", "home_team", "away_team"]].copy()
    return X, y, meta

"""
Database schema and connection helpers.

One row per completed regular-season game. Every column reflects information
that was knowable *before* first pitch (starters, season-to-date team stats,
park factors, rest days). The home_win outcome label is filled in after the game.
"""

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, Session
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

Base = declarative_base()


class Game(Base):
    __tablename__ = "games"
    __table_args__ = (
        UniqueConstraint("game_pk", name="uq_game_pk"),
        Index("ix_games_date", "game_date"),
        Index("ix_games_season", "season"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_pk = Column(Integer, nullable=False)
    game_date = Column(String, nullable=False)   # ISO date "YYYY-MM-DD"
    season = Column(Integer, nullable=False)

    home_team = Column(String)
    away_team = Column(String)

    # Probable starting pitchers (pre-game announcement)
    home_sp_name = Column(String)
    away_sp_name = Column(String)
    home_sp_mlbam_id = Column(Integer)
    away_sp_mlbam_id = Column(Integer)

    # Starting pitcher quality (prior-season full stats as proxy)
    home_sp_era = Column(Float)
    home_sp_fip = Column(Float)
    home_sp_xfip = Column(Float)
    home_sp_k_pct = Column(Float)
    home_sp_bb_pct = Column(Float)
    home_sp_ip = Column(Float)

    away_sp_era = Column(Float)
    away_sp_fip = Column(Float)
    away_sp_xfip = Column(Float)
    away_sp_k_pct = Column(Float)
    away_sp_bb_pct = Column(Float)
    away_sp_ip = Column(Float)

    # Team offense (prior-season full stats)
    home_woba = Column(Float)
    home_wrc_plus = Column(Float)
    away_woba = Column(Float)
    away_wrc_plus = Column(Float)

    # Team pitching / bullpen quality (prior-season full stats)
    home_team_era = Column(Float)
    home_team_fip = Column(Float)
    away_team_era = Column(Float)
    away_team_fip = Column(Float)

    # Park factor (run-scoring environment, 100 = neutral)
    park_factor = Column(Float)

    # Rest days between games for each team
    home_rest_days = Column(Integer)
    away_rest_days = Column(Integer)

    # Outcome (filled after the game completes)
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_win = Column(Integer)   # 1 = home won, 0 = away won, NULL = not yet settled

    # First-5-innings outcome (NULL for ties or games with < 5 completed innings)
    home_score_f5 = Column(Integer)
    away_score_f5 = Column(Integer)
    home_win_f5   = Column(Integer)   # 1 = home ahead, 0 = away ahead, NULL = tie/push

    # Real bookmaker closing odds (filled by historical_odds ingestion)
    # NULL until historical odds are pulled for this game.
    home_close_american = Column(Float)
    away_close_american = Column(Float)
    close_overround     = Column(Float)
    close_home_fair     = Column(Float)   # vig-removed implied prob for home
    close_away_fair     = Column(Float)
    closing_bookmaker   = Column(String)

    data_source = Column(String, default="historical")
    created_at = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())


class PitcherSeason(Base):
    """FanGraphs season-level pitcher stats, keyed by (mlbam_id, season)."""
    __tablename__ = "pitcher_seasons"
    __table_args__ = (
        UniqueConstraint("mlbam_id", "season", name="uq_pitcher_season"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    mlbam_id = Column(Integer)
    fangraphs_id = Column(String)
    name = Column(String)
    season = Column(Integer)
    team = Column(String)
    ip = Column(Float)
    era = Column(Float)
    fip = Column(Float)
    xfip = Column(Float)
    k_pct = Column(Float)
    bb_pct = Column(Float)
    whip = Column(Float)
    hr9 = Column(Float)


class TeamSeason(Base):
    """FanGraphs season-level team batting and pitching stats."""
    __tablename__ = "team_seasons"
    __table_args__ = (
        UniqueConstraint("team_abbr", "season", "stat_type", name="uq_team_season_type"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_abbr = Column(String)
    team_name = Column(String)
    season = Column(Integer)
    stat_type = Column(String)   # 'batting' or 'pitching'

    # Batting
    woba = Column(Float)
    wrc_plus = Column(Float)
    ops = Column(Float)
    avg = Column(Float)

    # Pitching
    era = Column(Float)
    fip = Column(Float)
    whip = Column(Float)


class ParkFactor(Base):
    __tablename__ = "park_factors"
    __table_args__ = (
        UniqueConstraint("team_abbr", "season", name="uq_park_factor"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_abbr = Column(String)
    team_name = Column(String)
    season = Column(Integer)
    basic_pf = Column(Float)    # runs park factor (100 = neutral)


class PitcherGameLog(Base):
    """
    Per-start Statcast stats for each pitcher, used as rolling features
    for the strikeout prop model. One row per (pitcher, game_date).
    """
    __tablename__ = "pitcher_game_logs"
    __table_args__ = (
        UniqueConstraint("mlbam_id", "game_date", name="uq_pitcher_game"),
        Index("ix_pitcher_game_mlbam", "mlbam_id"),
    )

    id          = Column(Integer, primary_key=True, autoincrement=True)
    mlbam_id    = Column(Integer, nullable=False)
    player_name = Column(String)
    game_date   = Column(String, nullable=False)
    game_pk     = Column(Integer)
    season      = Column(Integer)
    opponent    = Column(String)
    home_away   = Column(String)    # "home" or "away"

    # Per-start outcomes
    ip          = Column(Float)
    strikeouts  = Column(Integer)
    walks       = Column(Integer)
    hits        = Column(Integer)
    earned_runs = Column(Integer)
    pitches     = Column(Integer)
    strikes     = Column(Integer)

    # Statcast pitch-mix / quality (season-to-date up to this start)
    k_pct_std   = Column(Float)   # rolling K% (season-to-date)
    swstr_pct   = Column(Float)   # swinging strike rate
    csw_pct     = Column(Float)   # called strike + whiff rate


class BatterGameLog(Base):
    """
    Per-game Statcast stats for batters, used for hits/TB prop model.
    One row per (batter, game_date).
    """
    __tablename__ = "batter_game_logs"
    __table_args__ = (
        UniqueConstraint("mlbam_id", "game_date", name="uq_batter_game"),
        Index("ix_batter_game_mlbam", "mlbam_id"),
    )

    id          = Column(Integer, primary_key=True, autoincrement=True)
    mlbam_id    = Column(Integer, nullable=False)
    player_name = Column(String)
    game_date   = Column(String, nullable=False)
    game_pk     = Column(Integer)
    season      = Column(Integer)
    team        = Column(String)
    opponent    = Column(String)
    home_away   = Column(String)

    # Game results
    ab          = Column(Integer)
    pa          = Column(Integer)
    hits        = Column(Integer)
    doubles     = Column(Integer)
    triples     = Column(Integer)
    home_runs   = Column(Integer)
    total_bases = Column(Integer)
    strikeouts  = Column(Integer)
    walks       = Column(Integer)

    # Statcast quality (for the game)
    avg_exit_velo  = Column(Float)
    barrel_pct     = Column(Float)
    hard_hit_pct   = Column(Float)


class PropBet(Base):
    """
    One row per player prop bet (paper or real money).
    Covers pitcher strikeouts and batter hits/total bases.
    """
    __tablename__ = "prop_bets"
    __table_args__ = (
        UniqueConstraint("game_pk", "player_name", "market", name="uq_prop_bet"),
    )

    id              = Column(Integer, primary_key=True, autoincrement=True)
    game_pk         = Column(Integer, nullable=False)
    game_date       = Column(String, nullable=False)
    player_id       = Column(Integer)
    player_name     = Column(String)
    team            = Column(String)
    opponent        = Column(String)

    # Market details
    market          = Column(String)    # "pitcher_strikeouts", "batter_hits", etc.
    line            = Column(Float)     # e.g. 6.5 Ks
    bet_side        = Column(String)    # "over" or "under"
    american_odds   = Column(Float)
    decimal_odds    = Column(Float)
    fair_prob       = Column(Float)
    model_prob      = Column(Float)
    edge            = Column(Float)

    # Closing line (for CLV)
    close_american  = Column(Float)
    clv             = Column(Float)

    # Stake
    stake_fraction  = Column(Float)
    stake_dollars   = Column(Float)
    bankroll_at_bet = Column(Float)
    bookmaker       = Column(String)

    # Settlement
    actual_value    = Column(Float)     # e.g. actual Ks thrown
    outcome         = Column(Integer)   # 1=won, 0=lost, NULL=pending
    profit_dollars  = Column(Float)

    is_paper        = Column(Integer, default=1)   # 1=paper, 0=real money
    created_at      = Column(String)
    settled_at      = Column(String)


class PaperBet(Base):
    """
    One row per bet logged during paper trading.
    Odds are recorded at the time the bet is logged (morning of game day).
    Closing odds are recorded separately (close to first pitch) for CLV.
    """
    __tablename__ = "paper_bets"
    __table_args__ = (
        UniqueConstraint("game_pk", name="uq_paper_bet_game"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_pk       = Column(Integer, nullable=False)
    game_date     = Column(String, nullable=False)
    home_team     = Column(String)
    away_team     = Column(String)
    bet_side      = Column(String)   # "home" or "away"

    # Model output
    model_prob    = Column(Float)
    fair_prob     = Column(Float)    # book's vig-free implied prob for bet side
    edge          = Column(Float)    # model_prob - fair_prob at time of bet

    # Odds at time of bet (morning line)
    home_american_open  = Column(Float)
    away_american_open  = Column(Float)
    bet_american_odds   = Column(Float)   # odds for the recommended side
    bet_decimal_odds    = Column(Float)
    overround_open      = Column(Float)

    # Closing odds (filled in ~30 min before first pitch for CLV)
    home_american_close = Column(Float)
    away_american_close = Column(Float)
    bet_american_close  = Column(Float)
    clv                 = Column(Float)   # fair_close_prob - bet_implied_prob

    # Stake
    stake_fraction  = Column(Float)   # fraction of bankroll
    stake_dollars   = Column(Float)   # actual dollars (based on current bankroll)
    bankroll_at_bet = Column(Float)

    # Settlement
    home_score    = Column(Integer)
    away_score    = Column(Integer)
    outcome       = Column(Integer)   # 1=won, 0=lost, NULL=unsettled
    profit_dollars = Column(Float)    # NULL until settled

    # Bookmaker used (for reference)
    bookmaker     = Column(String)
    created_at    = Column(String)
    settled_at    = Column(String)


class F5Bet(Base):
    """
    Paper bets on the first-5-innings (F5) moneyline.
    Settlement uses per-inning linescore; ties are pushes (outcome = -1).
    """
    __tablename__ = "f5_paper_bets"
    __table_args__ = (
        UniqueConstraint("game_pk", name="uq_f5_paper_bet_game"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_pk       = Column(Integer, nullable=False)
    game_date     = Column(String, nullable=False)
    home_team     = Column(String)
    away_team     = Column(String)
    bet_side      = Column(String)   # "home" or "away"

    model_prob    = Column(Float)
    fair_prob     = Column(Float)
    edge          = Column(Float)

    home_american_open  = Column(Float)
    away_american_open  = Column(Float)
    bet_american_odds   = Column(Float)
    bet_decimal_odds    = Column(Float)
    overround_open      = Column(Float)

    home_american_close = Column(Float)
    away_american_close = Column(Float)
    bet_american_close  = Column(Float)
    clv                 = Column(Float)

    stake_fraction  = Column(Float)
    stake_dollars   = Column(Float)
    bankroll_at_bet = Column(Float)

    # F5-specific scores (home/away runs through 5 innings)
    home_score_f5   = Column(Integer)
    away_score_f5   = Column(Integer)
    outcome         = Column(Integer)   # 1=won, 0=lost, -1=push, NULL=pending
    profit_dollars  = Column(Float)

    bookmaker     = Column(String)
    created_at    = Column(String)
    settled_at    = Column(String)


class GameUmpire(Base):
    """Home plate umpire assignment per game (populated by umpire_data.py)."""
    __tablename__ = "game_umpires"
    game_pk  = Column(Integer, primary_key=True)
    ump_name = Column(String)
    ump_id   = Column(Integer)


class UmpireStat(Base):
    """Career umpire tendency stats computed from game data."""
    __tablename__ = "umpire_stats"
    __table_args__ = (UniqueConstraint("ump_name", name="uq_ump_name"),)

    id          = Column(Integer, primary_key=True, autoincrement=True)
    ump_name    = Column(String, nullable=False)
    games       = Column(Integer)
    runs_pg     = Column(Float)     # career avg total runs per game
    runs_vs_avg = Column(Float)     # runs_pg - overall league avg
    k9          = Column(Float)     # career SP K9 from pitcher_game_logs
    k9_vs_avg   = Column(Float)     # k9 - overall league avg K9


class LineSnapshot(Base):
    """
    Timestamped odds snapshot for a single game.

    Three snapshots per game per day:
      "open"    — 7:00am, first look at the market
      "morning" — ~10:00am, captured when update_and_pick.py runs
      "close"   — ~4:00pm, captured when capture_clv.py runs

    Comparing open→morning tells us if lines already moved before we bet.
    Comparing morning→close tells us if lines moved with or against us after.
    """
    __tablename__ = "line_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "game_date", "home_team", "away_team", "snapshot_label",
            name="uq_line_snapshot",
        ),
    )

    id             = Column(Integer, primary_key=True, autoincrement=True)
    game_pk        = Column(Integer)
    game_date      = Column(String, nullable=False)
    home_team      = Column(String, nullable=False)
    away_team      = Column(String, nullable=False)
    snapshot_label = Column(String, nullable=False)   # "open", "morning", "close"
    snapshot_time  = Column(String, nullable=False)   # ISO timestamp

    # Consensus pair (used for vig removal and edge calculation)
    home_american  = Column(Float)
    away_american  = Column(Float)
    bookmaker      = Column(String)

    # Best per-side odds across all books
    best_home_american = Column(Float)
    best_away_american = Column(Float)
    best_home_book     = Column(String)
    best_away_book     = Column(String)


def get_engine(db_path=None):
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False)


def init_db(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session(engine=None) -> Session:
    if engine is None:
        engine = get_engine()
    return Session(engine)

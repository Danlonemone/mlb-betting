# MLB Value Betting Model

A full pipeline for finding value in MLB moneyline and player-prop markets: data
ingestion, walk-forward-validated modeling, vig removal, fractional-Kelly bet
sizing, and daily paper-trading automation with CLV tracking.

**Status: paper trading only.** The model's calibration is solid, but a
backtest against real bookmaker closing odds shows no positive ROI at any
edge threshold tested — see [Results](#results) for why that's reported here
rather than hidden.

---

## What this project does

1. **Ingests** MLB Stats API data daily — schedules, box scores, pitcher and
   team stats, park factors, home-plate umpire assignments — into SQLite.
2. **Builds features** from that data: starting-pitcher form, team offense/defense,
   bullpen freshness, ballpark factors, Elo ratings, umpire run-scoring tendency,
   and the market's own vig-free implied probability.
3. **Trains** a walk-forward-validated logistic regression (primary) and XGBoost
   model on 2019–2026 (2020 excluded — 60-game COVID season), predicting
   home-team win probability.
4. **Prices bets**: removes the vig from bookmaker odds, compares to the model's
   probability, and sizes any edge found with fractional Kelly, subject to
   hard per-bet and per-day exposure caps.
5. **Paper-trades it daily**: an unattended pipeline fetches odds, generates
   picks, and — since all bets are logged dry-run for manual confirmation —
   a local dashboard lets you review and place them, then settles and captures
   closing-line value (CLV) automatically.

A parallel pipeline does the same for player props (pitcher strikeouts, batter
hits, batter total bases) using dedicated per-market models.

---

## Architecture

```
ingestion/       MLB Stats API pulls (schedules, box scores, pitcher/team
                  stats, park factors, umpire assignments, Statcast)
features/        Feature engineering — training and live share one
                  convention for missing data (see Engineering notes)
model/           Walk-forward training (logistic + XGBoost), calibration
betting/         Vig removal, edge calculation, fractional-Kelly sizing
props/           Per-market prop models (K's, hits, total bases) + pricing
paper_trade/     Daily pick generation, odds fetching, settlement, CLV
db/              SQLAlchemy schema (SQLite)
backtest/        Walk-forward harness + edge-threshold sweep
dashboard/       Local HTTP dashboard (basic-auth protected) for reviewing
                  and confirming picks, bankroll, and performance
automation/      launchd job templates for the daily unattended pipeline
tests/           Odds math, Kelly, settlement, and feature-parity tests
docs/            Engineering notes and roadmap (see below)
```

## Data sources

- **Moneyline / F5 odds**: a free, no-key scraper against a public sportsbook
  scoreboard API (`paper_trade/odds_client.py`). This has run unattended for
  months with no cost.
- **Player props odds**: [The Odds API](https://the-odds-api.com) — props
  require a paid tier (their free tier is moneyline-only). Without an active
  subscription, the props step fails a single API call per run and the rest
  of the pipeline continues unaffected; with one, it works end-to-end.
- **Everything else** (stats, schedules, park factors, umpires): the free
  MLB Stats API.

## Modeling notes

- **28 features**, walk-forward validated (train on seasons ≤ N, test on N+1
  — never trained on future data). Full list in `features/engineering.py`.
- The single most important feature is `market_fair_prob` — the book's own
  vig-free probability, fed back in as a model input. Adding this fixed a
  serious calibration failure (predicted 92% win probability bins were only
  winning 50% of the time before it was added).
- Isotonic calibration on top of the base classifier; walk-forward calibration
  gaps are now within a few points across all probability bins.
- Both a logistic regression and an XGBoost model are trained; logistic has
  consistently generalized better out-of-sample.

## Betting logic

- Vig is removed from two-sided markets before comparing to the model's
  probability (`betting/odds.py`).
- Bet sizing is fractional Kelly (`KELLY_FRACTION = 0.125`, i.e. eighth-Kelly)
  with a hard 3%-of-bankroll cap per bet and a 15%-of-bankroll cap on total
  same-day exposure across moneyline, F5, and props combined — because a
  moneyline bet, an F5 bet, and a strikeout prop on the same game are all
  correlated (same starters decide the outcome).
- `config.py` enforces a **pre-commitment rule**: sizing parameters may not
  change until there are ≥200 settled paper bets *and* positive mean CLV.
  This exists because an earlier version of this project doubled the Kelly
  fraction off three settled bets of evidence, immediately before a 43%
  bankroll drawdown — see `docs/REVIEW-2026-06-10.md` for the full account.
- Favorites require a stricter edge threshold than underdogs
  (`MIN_EDGE_FAVORITE = 0.15` vs `MIN_EDGE = 0.10`), because paper trading
  showed favorites landing 0/9 positive CLV at the base threshold while
  underdogs were positive.

## Results

Two separate, honest findings, not one polished number:

- **Calibration is good.** After adding the market-probability feature, the
  model's predicted probabilities track actual outcomes closely across every
  bin in walk-forward validation (2022→2026 test seasons).
- **No demonstrated edge against real closing odds.** A threshold sweep run
  against actual bookmaker closing lines (not the synthetic log5-priced
  market some earlier iterations of this project used for backtesting) shows
  *negative* ROI at every edge threshold from 2% to 15%. An earlier version
  of the backtest reported "+12.9% ROI at a 15% threshold" — that number came
  from pricing bets against a synthetic market that's a much softer opponent
  than a real book, and didn't survive contact with real prices. Live paper
  trading (139 settled bets, June 2026) also ran net-negative.

MLB moneylines are an efficient market; this is the expected outcome for a
model using publicly available box-score-level stats, not a bug. The
value of the project, as it stands, is the pipeline and validation discipline
around it — not a proven trading edge.

## Engineering practices worth noting

- **No look-ahead, enforced structurally.** Every rolling stat and cutoff
  (`before_date` everywhere) is date-gated; walk-forward validation trains
  only on seasons strictly before the test season.
- **Train/serve parity is tested**, not assumed — `tests/test_core.py`
  asserts that live feature construction produces the same values as
  training for the same inputs, after a real bug was found where they
  silently diverged on missing data.
- **Real vs. synthetic odds are tracked separately** in every backtest so a
  threshold or feature can't look good only because it was evaluated against
  a weaker synthetic market.
- **CLV (closing-line value) is captured end-to-end**, not just win/loss —
  the more reliable signal for whether a betting process has edge, since it's
  available immediately rather than needing hundreds of settled bets.
- 24 automated tests cover odds math, Kelly edge cases, push/void settlement
  logic, and feature parity.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # add ODDS_API_KEY if you have a props subscription

python model/train.py                    # walk-forward validation + retrain
python props/strikeout_model.py          # K-props walk-forward + retrain
python backtest/threshold_sweep.py       # edge-threshold sweep vs real odds

python paper_trade/update_and_pick.py    # one full daily cycle (dry-run)
python dashboard/app.py --port 8765      # review/confirm picks, see performance

pytest tests/ -v
```

`automation/` has launchd job templates for running the daily cycle, opening-line
snapshot, and settlement unattended on macOS.

## Further reading

- [`docs/REVIEW-2026-06-10.md`](docs/REVIEW-2026-06-10.md) — a self-audit that
  caught and fixed a risky mid-drawdown sizing change, a train/live feature
  bug, and several settlement bugs.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — feature ideas and prioritization notes.

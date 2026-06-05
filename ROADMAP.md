# MLB Betting Model — Roadmap

*Current state as of 2026-06-04:*
- Moneyline model live, paper trading, walk-forward validated (2019–2026)
- 19 features: prior-season SP/team stats, rolling team form, SP recent form (last 5 starts)
- Kelly sizing at 20% fraction, MIN_EDGE 15%, MIN_MARKET_PROB 20%
- Bankroll: ~$106.96 (started $75.04) — 5 bets logged, 3 settled (2W/1L)

---

## Tier 1 — Improve the Moneyline Model

These fill concrete gaps in what the model currently knows before first pitch.

### 1. Today's Starting Lineup
The biggest blind spot. `woba_diff` uses the prior-season team average, but today's lineup might be missing three regulars. The MLB Stats API provides confirmed lineups ~2 hours before first pitch.

- Pull lineup via `/api/v1/schedule?hydrate=lineups`
- Compute lineup wOBA from Statcast (average over players in the card)
- Replace `woba_diff` with a live-lineup version when available; fall back to season avg when not

### 2. Bullpen Freshness
`team_era_diff` is a prior-season bullpen proxy. It misses whether the pen threw 4 innings yesterday. The MLB Stats API has per-pitcher game logs — track IP thrown in the last 3 days.

- New features: `home_bullpen_ip_l3d`, `away_bullpen_ip_l3d`
- Starters who go deep reduce bullpen load; can combine with SP recent form

### 3. Umpire Home Plate Tendencies
Home plate umpires have measurable tendencies on ball/strike calls that affect K% and walk rate. Historical ump assignments are public.

- Pull ump assignment from MLB Stats API schedule
- Join to historical ump K% and walk rate differential vs league average
- New features: `ump_k_rate_adj`, `ump_bb_rate_adj`

### 4. Weather / Wind
Wind direction and speed at Wrigley, Coors, and similar parks meaningfully shifts run expectation. APIs: OpenWeather or WeatherAPI (free tier).

- Fetch at pick time, not ingestion time
- Features: `wind_speed`, `wind_out` (boolean — blowing out to CF), `temperature`
- Most impactful at Coors, Wrigley, Fenway

### 5. Better Calibration at Extremes
The model can't output home-win probabilities below ~35%. The MIN_MARKET_PROB floor (raised to 20% today) is a band-aid. A longer-term fix:

- Train a separate "range of outcomes" model on implied market probability as a feature
- Or: add market implied probability as a soft prior in the feature set
- Evaluate: does including `market_implied_prob` as a feature improve Brier score?

---

## Tier 2 — Expand to New Markets

Each is a separate model trained on a separate target. The DB schema already has tables scaffolded for props.

### 6. First 5 Innings (F5) Moneyline
Removes bullpen variance entirely — outcome determined only by the two starters. Cleaner signal for the SP features that already dominate the model. The F5 market is liquid at DraftKings/FanDuel.

- Same features as the main model, minus team bullpen ERA
- Target: `home_win_f5` (requires per-inning score from MLB API)
- Very low lift to implement — reuse the entire pipeline

### 7. Game Totals (Over/Under)
Different target, different feature emphasis. Park factor and team offenses matter more; SP ERA/FIP matter differently.

- Features: `park_factor`, `home_sp_era`, `away_sp_era`, `home_woba`, `away_woba`, `wind_out` (critical)
- Target: `total_runs > line`
- Requires pulling the O/U line from The Odds API (already available via `spreads` market)

### 8. Pitcher Strikeout Props
The `PitcherGameLog` table is already populated (6,184 starts for 2025–2026). The schema for `PropBet` is already in the DB.

- Features: SP recent K/9 (already computed), opposing team K%, ump K-rate tendency, pitch count environment
- Target: `strikeouts > line` (e.g., 5.5 Ks)
- Odds source: The Odds API `pitcher_strikeouts` market
- This is the highest-edge prop market — books are slower to adjust than moneyline

### 9. Batter Hits / Total Bases
`BatterGameLog` table already in schema (empty — needs ingestion). Requires per-game Statcast from Baseball Savant.

- Features: batter rolling BA/wOBA last 10 games, opposing SP K%, matchup handedness
- Lower priority than strikeout props — noisier market

---

## Tier 3 — Infrastructure & Process

### 10. Multi-Book Odds Comparison
Currently takes the best available line from The Odds API response. Should explicitly track which book offers the best price per game and flag line shopping opportunities.

- Store all bookmaker odds per game (already in the API response, currently aggregated)
- Recommend: "bet this at FanDuel (+215) not DraftKings (+205)"

### 11. Automated Daily Loop
The `update_and_pick.py` script is ready. Wire it to macOS launchd to run at 10am daily (after lineups and morning lines are posted).

- launchd plist at `~/Library/LaunchAgents/mlb.betting.plist`
- Send pick output to a Slack webhook or iMessage for review before logging

### 12. Automated Settlement
`settle.py` runs manually right now. Run it again at midnight once games are final.

- Add a second launchd job at 12:15am
- Flag any bet that aged past 24 hours without settling (postponed game)

### 13. Walk-Forward Backtest Dashboard
Currently MIN_EDGE, KELLY_FRACTION, and MIN_MARKET_PROB are tuned manually. Build a grid search over the 2022–2025 walk-forward window.

- Sweep MIN_EDGE: 0.10–0.25 in 0.01 steps
- Sweep MIN_MARKET_PROB: 0.15–0.35
- Report: ROI, N bets, Sharpe, max drawdown per combination
- Run once per off-season on fresh historical data

---

## Order of Attack

| Priority | Item | Effort | Expected lift |
|----------|------|--------|---------------|
| 1 | F5 moneyline model | Low | Medium — cleaner SP signal |
| 2 | Bullpen freshness | Low | Medium — fixes the biggest missing signal |
| 3 | Pitcher strikeout props | Medium | High — least efficient market |
| 4 | Automated daily loop (launchd) | Low | Process only |
| 5 | Today's starting lineup | Medium | High — fills the biggest feature gap |
| 6 | Umpire tendencies | Medium | Low-medium |
| 7 | Weather/wind | Low | Medium at specific parks |
| 8 | Game totals model | Medium | TBD |
| 9 | Walk-forward backtester | Medium | Process only |
| 10 | Multi-book line shopping | Low | Process only |

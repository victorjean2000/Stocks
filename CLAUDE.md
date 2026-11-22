# CLAUDE.md

## Market data (stockdata package)

Daily panel for backtesting the US momentum strategy: Alpaca prices (SIP) plus
point-in-time shares outstanding and sector from SEC EDGAR. Full details in DATA.md.

**Load it with one call:**

```python
from stockdata import load_panel

df  = load_panel("2020-01-01", "2024-12-31")                    # every kept stock, every day
uni = load_panel("2020-01-01", "2024-12-31", universe_only=True)  # only rows that pass the universe filters
px  = load_panel(tickers=["AAPL", "MSFT"], columns=["close", "mcap"])
```

Columns: `date, ticker, open, high, low, close` (fully adjusted, use for signals), `close_raw`
(as traded), `volume` (raw shares), `dollar_volume`, `adv50`, `split_factor`, `shares` (known on that
date, as-traded units), `shares_asof`, `shares_filed`, `shares_source`, `mcap` (company level),
`sector`, `cik`, `security_type`, `is_primary`, `data_issue`, `in_universe`.

`in_universe` = common stock, primary share class, mcap >= $3bn, raw close >= $10,
50-day average dollar volume >= $25m, no flagged data issue. Thresholds: `stockdata/config.py`.

**Before using the data in a new session**, check it is ready:

```bash
python -m stockdata status        # sync.state should be "ready"; last_session = latest close
```

The SessionStart hook (`.claude/hooks/session-start.sh`) starts `python -m stockdata sync` in the
background on every cloud session: it restores the encrypted snapshot if `STOCKDATA_SNAPSHOT_KEY` is
set (about a minute), otherwise rebuilds from scratch (about 20 minutes), then tops up incrementally.
If `sync.state` is `running`, wait for it (`tail -f data/logs/sync.log`); do not start a second build.

Earnings release dates (SEC 8-K Item 2.02, US filers only): `from stockdata import load_earnings`;
`session` is the first trading session that can react to each release.

Other commands: `python -m stockdata update` (incremental; `--full` re-downloads everything),
`python -m stockdata rebuild` (recompute shares/panel from local files), `python -m stockdata earnings`
(earnings dates from the local submissions.zip), `python -m pytest`.

## Backtests

`python -m backtest.momentum_trend` runs the momentum + trend strategy (golden cross, 3-day SuperTrend(10,3)
entry/exit, top 12 by composite momentum, max 4 per sector, next-open fills, 10 bps per side) plus variants
(SPY 200-day market filter, skip the most volatile 20%, volatility-adjusted ranking, ±5-session earnings
blackout, cost and rule sensitivities) and SPY / equal-weight-universe benchmarks in about 30 s. Results:
`backtest/RESULTS.md`; equity curves, trades (with exit reason) and month-end holdings for every variant in
`data/backtests/momentum_trend/`. Rules are documented at the top of `backtest/momentum_trend.py`; new
variants are `Params(...)` entries in `MAIN` / `SENSITIVITIES`.

`python -m backtest.strategies` (about 40 s) runs the strategy search: about 200 settings of index trend
(plain and leveraged ETFs), stock momentum, earnings drift, buybacks, reversal, mega-caps, sector rotation and
more, on a generic next-open engine with ETF prices from Alpaca and T-bill (BIL) interest on cash. 2017-21 are
the tuning years, 2022-26 the out-of-sample check, plus a risk-matched SPY comparison. Results:
`backtest/STRATEGIES.md`; curves and the full table in `data/backtests/strategies/`.

`python -m backtest.momentum_refine` (about 2 minutes on 4 cores) tests changes to the paper momentum strategy one at a
time: profit taking (sell all or half at a gain, trailing stops, stop losses, selling stretched stocks), rank buffers,
holdings, lookback, pool, SuperTrend exits. Every variant runs at 10 rebalance days (0, 2, ..., 18 sessions after
month end; the result depends a lot on the day) and is judged on 2017-21, checked on 2022-26. Results:
`backtest/MOMENTUM_REFINE.md`. The simulator options it uses (`take_profit`, `trail_stop`, `stop_loss`,
`extension_exit`, `keep_rank`, `reweight`, `offset`, `cash_yield`) are `Params` fields in `backtest/momentum_trend.py`.

`python -m backtest.daily` (about 15 s) tests daily high-win-rate rules: buying short dips in QQQ / SPY (traded 1x,
2x or 3x: QLD, TQQQ, SSO, UPRO) on RSI(2), IBS, N-day lows or down streaks, in nine 3x index and sector funds at once
(`FUNDS`, signalled by their unleveraged ETFs, bars cached in `data/benchmarks/funds_daily.parquet`), and in the 500
largest stocks; signals at the close, fills at the next open, win rate after costs (with 10 / 20 bps checks); picked
on 2017-21, checked on 2022-26. Results:
`backtest/DAILY.md`. Daily bars of a session still trading are never cached (`stockdata.alpaca.drop_unfinished`).

`python -m backtest.lqq3` backtests the trend rule on real LQQ3 prices in pounds (the user's ISA version: LQQ3 while
QQQ closes above its 200-day average, else sterling cash, trading at the next London open). Results:
`backtest/LQQ3.md`. `python -m backtest.lqq3 dips` tests the daily dip rules of `backtest.daily` for the ISA: QQQ
signals at the US close, LQQ3 traded at the next London open or close in pounds, against the S&P 500 in pounds
(`backtest/LQQ3_DAILY.md`).

`python -m backtest.isa` (about 25 s) builds the ISA strategy from the London 3x funds the user can hold (WisdomTree
LQQ3, 3LUS, 3UKL, 3LDE, 3LEU, 3UKS; Leverage Shares 3SMH, 3GDX from April 2024; real prices in pence, splits and
consolidations undone by `adjust_actions`; untraded days whose Yahoo price is stale or false valued from the
index by `fill_stale_prints` in `backtest/lqq3.py`) on the daily dip rules: US-ETF signals at the US close (own London prices
for the FTSE, DAX and EURO STOXX), next London open, sterling cash, S&P 500 in pounds as the benchmark. `RECOMMENDED`
in `backtest/isa.py` is the current answer (LQQ3 + 3LUS half each, IBS < 0.1, sell above the 5-day average). Whether 3x short funds help is tested on the US funds
(TQQQ / UPRO long, SQQQ / SPXU short; bars in `data/benchmarks/longshort_daily.parquet`) because London short-fund
prices are not approved. Results: `backtest/ISA.md`, which also shows the recommendation without 2020 or its
best trades, what it captures, and combined with the 200-day rule (`combination_section`).

`python -m backtest.isa_tests` (about 40 s) tests that recommendation thoroughly and explains why it works: the index's
moves after closes near the day's low (IBS deciles, the 10 days after, hour by hour from Alpaca 30-minute bars cached in
`data/benchmarks/us_30min.parquet` via `stockdata.alpaca.fetch_intraday`), a threshold x exit grid, costs, fill timing,
a random-timing test, a block bootstrap, walk-forward, one-at-a-time extra rules (filters, exits, stops, sizing), and
hourly variants (decisions at the London close, the US 3x funds at other times of day). `london_run` trades the London
funds at their open or close; `event_run` is the same engine on any event axis. Results: `backtest/ISA_TESTS.md`.

`python -m backtest.long_history` (about 15 s) runs the same rule over 2000-2026, through the 2000-02 and 2007-09 bear
markets, before the years it was chosen on: Yahoo's daily SPY / QQQ before 2016 (approved; its SPY closes are wrong on
some extreme days, so Alpaca is used from 2016 and a Nasdaq-100-only version is shown), simulated daily-reset 3x funds
charged what the real TQQQ / UPRO cost against 3x their index in 2016-26 (`leveraged`, `running_cost`), next US open,
in dollars; also 2x / 1x, the 200-day filter, half-size buys below the 200-day average (checked on the London funds),
the trend rule and buy-and-hold. Results: `backtest/LONG_HISTORY.md`.

## Paper trading

`python -m trading status` / `run --dry-run` / `run` paper-trades two strategies from `backtest/strategies.py`,
each in its own Alpaca PAPER account (only `paper-api.alpaca.markets` is ever called):
- `momentum` (keys `ALPACA_MOMENTUM_KEY_ID/_SECRET_KEY`, else the main `ALPACA_API_KEY_ID/_SECRET_KEY`): 10 of the
  500 largest by 6-1 month return, 10% each, rebalanced once after each month end (orders tagged
  `momentum-YYYYMMDD-...`; a decision with filled orders is not traded again). Since 2026-11-13 a holding is kept while
  it ranks in the top 30 (`KEEP_RANK` in `trading/targets.py`; before, it was sold on leaving the top 10) and empty
  places go to the best-ranked stocks not held; `buffered_top` is the same choice as the backtest's `keep_rank=30`.
- `lqq3` (keys `ALPACA_LQQ3_KEY_ID/_SECRET_KEY`, a second paper account): TQQQ, the US stand-in for London-listed
  LQQ3, while QQQ closes above its 200-day average, otherwise BIL.
A run cancels its own open orders, then trades the holdings to the target with day market orders (queued for the
next open after the close). Buys are sized at the last price, so a sale that fills lower at the open (a TQQQ gap) can
leave cash negative; the next run sells the most overweight holding back to the 0.5% cash buffer (`plan_orders`). For lqq3 it also reports what real LQQ3 did in pounds on the same signals since
2026-11-05 (`backtest.lqq3.shadow`) and the account's own return in dollars and pounds. It refuses two strategies
on one account. `.github/workflows/paper-trade.yml` runs it after every successful data-snapshot build; run that
workflow by hand to start a new account.

Rules for this repo:
- Never print, log or commit `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `SEC_USER_AGENT` or
  `STOCKDATA_SNAPSHOT_KEY`. Never commit anything under `data/` (it is git-ignored).
- Data sources are Alpaca (data/paper-api) and SEC EDGAR only; ask before adding another source. One exception:
  daily prices from Yahoo Finance of symbols the user approved, listed in `YAHOO_APPROVED` in `backtest/lqq3.py`:
  London-listed funds for the ISA backtests (LQQ3.L on 2026-11-05; on 2026-11-19 3LUS.L and the WisdomTree / Leverage
  Shares 3x funds the user can hold in their ISA: 3UKL, 3LDE, 3LEU, 3LSI, 3LOI, 3ULS, 3UKS, 3SMH, 3XLF, 3XLE, 3GDX), and
  on 2026-11-22 the long daily history of SPY and QQQ (used before 2016, where Alpaca has none) for
  `backtest/long_history.py`. Fetched only through `fetch_yahoo`, cached in `data/benchmarks/`, never committed. Add a
  symbol only when the user approves it. Pound/dollar rates and sterling cash still come from Alpaca (FXB).
- The repository is public: never publish an unencrypted snapshot (Alpaca data cannot be redistributed).
- Market cap is point-in-time: shares come only from filings filed before the date (one-session lag).
  Never back-fill today's share count into history.

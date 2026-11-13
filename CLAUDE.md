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

`python -m backtest.lqq3` backtests the trend rule on real LQQ3 prices in pounds (the user's ISA version: LQQ3 while
QQQ closes above its 200-day average, else sterling cash, trading at the next London open). Results:
`backtest/LQQ3.md`.

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
next open after the close). For lqq3 it also reports what real LQQ3 did in pounds on the same signals since
2026-11-05 (`backtest.lqq3.shadow`) and the account's own return in dollars and pounds. It refuses two strategies
on one account. `.github/workflows/paper-trade.yml` runs it after every successful data-snapshot build; run that
workflow by hand to start a new account.

Rules for this repo:
- Never print, log or commit `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `SEC_USER_AGENT` or
  `STOCKDATA_SNAPSHOT_KEY`. Never commit anything under `data/` (it is git-ignored).
- Data sources are Alpaca (data/paper-api) and SEC EDGAR only; ask before adding another source. One exception,
  approved by the user on 2026-11-05: daily LQQ3.L prices from Yahoo Finance, for `backtest/lqq3.py` only (cached in
  `data/benchmarks/`, never committed). Pound/dollar rates and sterling cash still come from Alpaca (FXB).
- The repository is public: never publish an unencrypted snapshot (Alpaca data cannot be redistributed).
- Market cap is point-in-time: shares come only from filings filed before the date (one-session lag).
  Never back-fill today's share count into history.

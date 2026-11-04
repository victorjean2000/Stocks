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

Other commands: `python -m stockdata update` (incremental; `--full` re-downloads everything),
`python -m stockdata rebuild` (recompute shares/panel from local files), `python -m pytest`.

## Backtests

`python -m backtest.momentum_trend` runs the momentum + trend strategy (golden cross, 3-day SuperTrend(10,3)
entry/exit, top 12 by composite momentum, max 4 per sector, next-open fills, 10 bps per side) plus variants
and SPY / equal-weight-universe benchmarks in about 20 s. Results: `backtest/RESULTS.md`; equity curves and
trades in `data/backtests/momentum_trend/`. Rules are documented at the top of `backtest/momentum_trend.py`.

Rules for this repo:
- Never print, log or commit `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `SEC_USER_AGENT` or
  `STOCKDATA_SNAPSHOT_KEY`. Never commit anything under `data/` (it is git-ignored).
- Data sources are Alpaca (data/paper-api) and SEC EDGAR only; ask before adding another source.
- The repository is public: never publish an unencrypted snapshot (Alpaca data cannot be redistributed).
- Market cap is point-in-time: shares come only from filings filed before the date (one-session lag).
  Never back-fill today's share count into history.

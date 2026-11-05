# Stocks

Free, automated daily dataset for backtesting a US momentum strategy: Alpaca daily prices (SIP)
joined with point-in-time shares outstanding and sector from SEC EDGAR.

```python
from stockdata import load_panel
df = load_panel("2020-01-01", "2024-12-31", universe_only=True)
```

- `DATA.md`: sources, schema, point-in-time rules, refresh commands, known gaps.
- `python -m stockdata update`: incremental update (needs `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`,
  `SEC_USER_AGENT`).
- `python -m backtest.momentum_trend` and `python -m backtest.strategies`: backtests, results in
  `backtest/RESULTS.md` and `backtest/STRATEGIES.md`.
- `python -m trading status`: the two paper-traded strategies (momentum and LQQ3/TQQQ trend), one Alpaca paper account
  each, traded daily by `.github/workflows/paper-trade.yml`.
- `python -m pytest`: tests.

No market data is stored in this repository. A daily encrypted snapshot is published by the
`data-snapshot` workflow; Alpaca market data may not be redistributed, so it is only readable with
the owner's `STOCKDATA_SNAPSHOT_KEY`.

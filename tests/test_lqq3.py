"""LQQ3 sterling backtest pieces: bad prints, Yahoo parsing, and US signal -> London session timing."""
from __future__ import annotations

import pandas as pd

from backtest.lqq3 import drop_bad_prints, parse_chart, schedule


def test_drop_bad_prints_only_removes_flat_isolated_spikes():
    idx = pd.bdate_range("2025-10-20", periods=6)
    c = [23848.0, 23763.5, 23565.5, 32718.3, 25671.0, 26322.0]
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)
    df.loc[idx[[0, 1, 2, 4, 5]], "high"] += 100  # real sessions have a range; the bad print is flat
    assert list(drop_bad_prints(df).index) == list(idx[[0, 1, 2, 4, 5]])
    real = df.drop(idx[3])
    real.loc[idx[4], "close"] = real.loc[idx[4], "open"] = 30000.0  # a genuine big day (not flat) stays
    assert len(drop_bad_prints(real)) == 5


def _payload(ts_utc, closes, market_time, price, currency="GBp"):
    n = len(ts_utc)
    return {"chart": {"result": [{
        "meta": {"currency": currency, "regularMarketTime": int(pd.Timestamp(market_time, tz="UTC").timestamp()),
                 "regularMarketPrice": price},
        "timestamp": [int(pd.Timestamp(t, tz="UTC").timestamp()) for t in ts_utc],
        "indicators": {"quote": [{"open": [100.0] * n, "high": [110.0] * n, "low": [90.0] * n,
                                  "close": closes, "volume": [1] * n}]}}]}}


def test_parse_chart_fills_a_blank_last_close_after_london_shuts():
    p = _payload(["2026-11-03 08:00", "2026-11-04 08:00"], [101.0, None], "2026-11-04 16:35", 105.0)
    df = parse_chart(p, now=pd.Timestamp("2026-11-05 01:00", tz="UTC"))
    assert df["close"].tolist() == [101.0, 105.0]
    during = parse_chart(p, now=pd.Timestamp("2026-11-04 12:00", tz="UTC"))  # today's bar, London still open
    assert list(during.index) == [pd.Timestamp("2026-11-03")]


def test_signal_trades_at_the_next_london_session():
    lse = pd.DatetimeIndex(["2026-08-27", "2026-08-28", "2026-09-01", "2026-09-02"])  # 31 Aug: UK bank holiday
    on = pd.Series([True, True, False, False, True],
                   index=pd.DatetimeIndex(["2026-08-26", "2026-08-27", "2026-08-28", "2026-08-31", "2026-09-01"]))
    t = schedule(on, lse, i0=1)
    assert t[0] == {0: 1.0}  # in from the start (state at the last US close before 28 Aug)
    # off at the US close of Fri 28 Aug -> sold at London's open on Tue 1 Sep (Monday is a UK holiday);
    # back on at the US close of 1 Sep -> bought at London's open on 2 Sep
    assert t[1] == {} and t[2] == {0: 1.0}

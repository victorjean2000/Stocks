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


def test_parse_chart_keeps_a_day_whose_last_trade_came_before_the_closing_auction():
    # the Friday-night run of 13 Nov 2026 reported LQQ3 only through 12 Nov: a thin day can end before 16:30
    p = _payload(["2026-11-12 08:00", "2026-11-13 08:00"], [101.0, None], "2026-11-13 16:21", 103.0)
    assert parse_chart(p, now=pd.Timestamp("2026-11-14 00:27", tz="UTC"))["close"].tolist() == [101.0, 103.0]
    assert parse_chart(p, now=pd.Timestamp("2026-11-13 16:50", tz="UTC"))["close"].tolist() == [101.0, 103.0]
    early = parse_chart(p, now=pd.Timestamp("2026-11-13 16:40", tz="UTC"))  # auction not over yet: left out
    assert list(early.index) == [pd.Timestamp("2026-11-12")]


def test_dip_state_holds_from_a_buy_close_until_a_sell_close():
    import numpy as np

    from backtest.lqq3 import dip_state
    idx = pd.bdate_range("2026-01-05", periods=8)
    # closes near the day's low (IBS 0.05) on days 1 and 5; close > previous high on days 3 and 6
    c = pd.Series([100, 99, 99.5, 102, 101, 98, 101.5, 101.0], index=idx)
    h = pd.Series([101, 100, 100.5, 102.5, 102, 99.9, 102, 101.5], index=idx)
    l = pd.Series([99, 98.95, 99, 101, 100, 97.9, 100, 100.5], index=idx)
    bars = pd.DataFrame({"o": c, "h": h, "l": l, "c": c})
    state = dip_state(bars, ("ibs", 0.1), "high", False)
    assert state.tolist() == [False, True, True, False, False, True, False, False]


def test_lqq3_trades_compound_to_the_strategy_equity():
    import numpy as np
    import pytest

    from backtest.lqq3 import lqq3_strategy, lqq3_trades
    lse = pd.bdate_range("2026-01-05", periods=12)
    px = pd.Series(np.linspace(100, 130, 12), index=lse)
    lq = pd.DataFrame({"open": px * 0.99, "close": px})
    on = pd.Series([False, True, True, False, False, True, True, True, False, False, False, False], index=lse)
    cash = pd.Series(0.0, index=lse)
    eq, _ = lqq3_strategy(lq, px, cash, start=str(lse[1].date()), on=on)
    t = lqq3_trades(lq, on, start=str(lse[1].date()))
    assert len(t) == 2 and (t["entry"] == [lse[2], lse[6]]).all()  # the US close of day k trades at London k+1
    assert eq.iloc[-1] / eq.iloc[0] == pytest.approx(float(np.prod(1 + t["ret"])), rel=1e-9)


def test_untraded_repeats_and_false_prices_take_the_index_implied_value():
    import numpy as np

    from backtest.lqq3 import fill_stale_prints
    idx = pd.bdate_range("2020-01-01", periods=60)
    under = pd.Series(100 * np.cumprod(1 + np.random.default_rng(5).normal(0, 0.01, 60)), index=idx)
    path = (1 + 3 * under.pct_change().fillna(0.0)).cumprod()
    fair = 1000 * np.sqrt(path * path.shift(1))  # London closes midway between two US closes
    df = pd.DataFrame({"open": fair, "high": fair, "low": fair, "close": fair, "volume": 100.0}, index=idx).iloc[1:]
    df.iloc[20:25, :4] = df["close"].iloc[19]  # a stale run: the last price repeated, nothing traded
    df.iloc[20:25, 4] = 0.0
    df.iloc[40, :4] *= 1.3  # a false price on an untraded day
    df.iloc[40, 4] = 0.0
    df.iloc[30, 4] = 0.0  # an untraded day at the right price
    df.iloc[50, :4] *= 1.3  # a traded day is never changed
    out = fill_stale_prints(df, under, 3)
    want = fair.iloc[1:].copy()
    want.iloc[50] *= 1.3
    for col in ("open", "high", "low", "close"):
        assert np.allclose(out[col], want)
    assert out["volume"].equals(df["volume"])

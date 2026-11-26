"""The long-history test (backtest/long_history.py) and Yahoo's US-dollar prices. No network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.long_history import bad_closes, hold, leveraged, running_cost
from backtest.lqq3 import parse_chart


def _index(closes, opens=None) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    o = pd.Series(opens if opens is not None else closes, index=idx, dtype=float)
    return pd.DataFrame({"o": o, "h": np.maximum(o, c) + 1, "l": np.minimum(o, c) - 1, "c": c})


def test_a_3x_fund_resets_every_close_and_opens_on_the_overnight_move():
    t = _index([100, 110, 99], opens=[100, 105, 100])
    f = leveraged(t, 3, cost=0.0)
    assert f["c"].iloc[1] == pytest.approx(130)  # +10% x 3
    assert f["c"].iloc[2] == pytest.approx(130 * (1 + 3 * (99 / 110 - 1)))  # resets on the new close
    assert f["o"].iloc[1] == pytest.approx(100 * (1 + 3 * 0.05))  # the open: 3x the overnight move
    assert leveraged(t, 3, cost=0.0252)["c"].iloc[1] == pytest.approx(100 * (1.3 - 0.0001))


def test_running_cost_is_the_yearly_gap_to_a_costless_fund():
    t = _index(100 * np.cumprod(np.r_[1.0, np.full(252, 1.001)]))
    sim = leveraged(t, 3, cost=0.0)
    real = leveraged(t, 3, cost=0.06)
    assert running_cost(sim, real) == pytest.approx(0.06, rel=0.01)


def test_hold_buys_part_of_a_place_when_sized_down():
    t = _index([100] * 6)
    state = pd.Series([True] * 6, index=t.index)
    size = pd.Series(0.5, index=t.index)
    eq, tr = hold({"QQQ": state}, {"QQQ": t[["o", "c"]]}, start=str(t.index[1].date()), cost=0.0,
                  size={"QQQ": size})
    assert len(tr) == 0 and eq.iloc[-1] == pytest.approx(1.0)  # flat prices: half invested, nothing changes
    t2 = _index([100, 100, 110, 110, 110, 110])
    eq2, _ = hold({"QQQ": state}, {"QQQ": t2[["o", "c"]]}, start=str(t2.index[1].date()), cost=0.0,
                  size={"QQQ": size})
    assert eq2.iloc[-1] == pytest.approx(1.05)  # half the account caught the +10%


def test_bad_closes_finds_a_close_away_from_the_official_one():
    idx = pd.bdate_range("2020-01-01", periods=80)
    a = pd.DataFrame({"o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0}, index=idx)
    y = a.copy()
    y.loc[idx[40], "c"] = 97.0  # Yahoo's close 3% off on one day
    out = bad_closes({"SPY": y}, {"SPY": a})
    assert out["SPY"]["bad"] == 1 and out["SPY"]["dates"] == [str(idx[40].date())]


def _payload(currency: str, ts: list[int], close: list[float], adj: list[float]) -> dict:
    q = {"open": close, "high": [c + 1 for c in close], "low": [c - 1 for c in close], "close": close,
         "volume": [1000] * len(close)}
    return {"chart": {"result": [{"meta": {"currency": currency, "symbol": "SPY"}, "timestamp": ts,
                                  "indicators": {"quote": [q], "adjclose": [{"adjclose": adj}]}}]}}


def test_parse_chart_reads_dollar_prices_with_the_adjusted_close():
    ny = "America/New_York"
    days = [pd.Timestamp("2026-11-18 09:30", tz=ny), pd.Timestamp("2026-11-19 09:30", tz=ny)]
    ts = [int(d.timestamp()) for d in days]
    p = _payload("USD", ts, [700.0, 710.0], [690.0, 700.0])
    after = parse_chart(p, now=pd.Timestamp("2026-11-19 16:30", tz=ny), currency="USD")
    assert list(after["adjclose"]) == [690.0, 700.0] and after.index[-1] == pd.Timestamp("2026-11-19")
    during = parse_chart(p, now=pd.Timestamp("2026-11-19 15:00", tz=ny), currency="USD")
    assert len(during) == 1  # today's bar waits for the New York close
    with pytest.raises(ValueError):
        parse_chart(p, currency="GBp")


def test_yahoo_us_prices_do_not_share_the_alpaca_spy_cache():
    from backtest.lqq3 import yahoo_cache
    from stockdata.config import paths
    assert yahoo_cache("SPY") != paths().root / "benchmarks" / "SPY.parquet"  # momentum_trend.spy() keeps Alpaca's
    assert yahoo_cache("LQQ3.L").name == "LQQ3.L.parquet"

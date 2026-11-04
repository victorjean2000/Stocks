"""The strategy-search engine: next-open fills, costs, cash yield and rebalance days."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.strategies import Book, blend, key_days, run


def _book(o, c, cost_bps=10.0):
    o, c = np.asarray(o, float), np.asarray(c, float)
    return Book(pd.Index([f"A{i}" for i in range(o.shape[1])]), o, c, cost_bps)


def test_buys_at_next_open_and_pays_costs():
    dates = pd.bdate_range("2024-01-01", periods=3)
    o = [[100.0], [110.0], [121.0]]
    c = [[100.0], [121.0], [133.1]]
    eq, turn, last = run(_book(o, c), {0: {0: 1.0}}, np.zeros(3), 1, dates)
    cost = 10 / 1e4
    # decided at the close of day 0, bought at day 1's open (110), not day 0's close (100)
    assert eq.tolist() == pytest.approx([1.0, 1.1 * (1 - cost), 1.21 * (1 - cost)])
    assert last == {"A0": 1.0}


def test_idle_cash_earns_the_cash_return():
    dates = pd.bdate_range("2024-01-01", periods=4)
    o = c = np.full((4, 1), 50.0)
    eq, _, _ = run(_book(o, c), {0: {}}, np.array([0.0, 0.001, 0.001, 0.001]), 1, dates)
    assert eq.iloc[-1] == pytest.approx(1.001 ** 3)


def test_switching_assets_costs_both_sides():
    dates = pd.bdate_range("2024-01-01", periods=3)
    o = c = np.full((3, 2), 10.0)
    eq, turn, _ = run(_book(o, c, 5.0), {0: {0: 1.0}, 1: {1: 1.0}}, np.zeros(3), 1, dates)
    assert eq.iloc[1] == pytest.approx(1 - 5e-4)  # buy A0
    assert eq.iloc[2] == pytest.approx((1 - 5e-4) * (1 - 2 * 5e-4))  # sell A0, buy A1


def test_key_days_month_ends_and_offset():
    dates = pd.bdate_range("2024-01-01", "2024-04-30")
    i0 = int(dates.searchsorted(pd.Timestamp("2024-02-01")))
    days = key_days(dates, i0, "M")
    assert [str(dates[i].date()) for i in days] == ["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"]
    late = key_days(dates, i0, "M", offset=2)
    assert str(dates[late[1]].date()) == "2024-02-02"  # two sessions after Wednesday 31 January


def test_blend_rebalances_monthly():
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    up = pd.Series(np.linspace(1, 2, len(idx)), index=idx)
    flat = pd.Series(1.0, index=idx)
    mix = blend([up, flat], [0.5, 0.5], idx, cost_bps=0.0)
    assert 1.0 < mix.iloc[-1] < up.iloc[-1] and mix.iloc[0] == 1.0


def test_key_days_final_date_only_if_month_really_ends():
    mid = pd.bdate_range("2026-10-01", "2026-11-03")  # data stops mid-month
    i0 = 5
    assert str(mid[key_days(mid, i0, "M")[-1]].date()) == "2026-10-30"
    end = pd.bdate_range("2026-10-01", "2026-10-30")  # data stops on the month's last session
    assert str(end[key_days(end, i0, "M")[-1]].date()) == "2026-10-30"


def test_leveraged_fund_resets_daily_and_calibration_recovers_the_cost():
    from backtest.strategies import drag_of, leveraged
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.012, len(idx))), index=idx)
    open_ = close.shift(1) * (1 + rng.normal(0, 0.003, len(idx)))
    open_.iloc[0] = close.iloc[0]
    cash = pd.Series(0.0001, index=idx)
    o, c = leveraged(close, open_, cash, 3, 0.0)
    r, rc = close.pct_change().iloc[1:], c.pct_change().iloc[1:]
    assert np.allclose(rc, 3 * r - 2 * 0.0001)  # exactly 3x the day, less borrowing on the extra 2x
    assert np.allclose((o / c.shift(1)).iloc[1:], 1 + 3 * (open_ / close.shift(1) - 1).iloc[1:])
    real = leveraged(close, open_, cash, 3, 0.021)[1]
    assert drag_of(real, close, open_, cash, 3) == pytest.approx(0.021, abs=1e-4)

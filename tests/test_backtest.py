"""Backtest building blocks: SuperTrend, the N-day confirmation, and fill timing (no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.indicators import atr_wilder, confirmed, supertrend_bull
from backtest.momentum_trend import Market, Params, simulate


def _reference_supertrend(h, l, c, atr, mult):
    """Plain per-series loop of the TradingView ta.supertrend recursion."""
    out, fu, fl, bull, c_prev = [], np.nan, np.nan, False, np.nan
    for hi, lo, ci, a in zip(h, l, c, atr):
        if not (np.isfinite(a) and np.isfinite(ci)):
            out.append(np.nan)
            continue
        bu, bl = (hi + lo) / 2 + mult * a, (hi + lo) / 2 - mult * a
        first = np.isnan(fu)
        nfl = bl if (np.isnan(fl) or bl > fl or c_prev < fl) else fl
        nfu = bu if (np.isnan(fu) or bu < fu or c_prev > fu) else fu
        nb = (not ci < nfl) if bull else (ci > nfu)
        bull = False if first else nb
        fl, fu, c_prev = nfl, nfu, ci
        out.append(float(bull))
    return np.array(out)


def test_supertrend_matches_reference_loop():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=400)
    cols = ["A", "B", "C"]
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (400, 3)), axis=0)), index=idx, columns=cols)
    h, l = c * (1 + rng.uniform(0, 0.02, c.shape)), c * (1 - rng.uniform(0, 0.02, c.shape))
    c.iloc[150:155, 1] = np.nan  # a trading gap in B
    h.iloc[150:155, 1] = np.nan
    l.iloc[150:155, 1] = np.nan
    got = supertrend_bull(h, l, c, 10, 3.0)
    atr = atr_wilder(h, l, c, 10)
    for col in cols:
        ref = _reference_supertrend(h[col].to_numpy(), l[col].to_numpy(), c[col].to_numpy(),
                                    atr[col].to_numpy(), 3.0)
        np.testing.assert_array_equal(np.nan_to_num(got[col].to_numpy(), nan=-1), np.nan_to_num(ref, nan=-1))


def test_supertrend_direction_in_clear_trends():
    idx = pd.bdate_range("2020-01-01", periods=120)
    up = pd.DataFrame({"U": np.linspace(100, 200, 120), "D": np.linspace(200, 100, 120)}, index=idx)
    st = supertrend_bull(up * 1.01, up * 0.99, up, 10, 3.0)
    assert st["U"].iloc[-30:].eq(1).all() and st["D"].iloc[-30:].eq(0).all()


def test_confirmation_needs_n_consecutive_days():
    s = pd.DataFrame({"X": [1, 1, 0, 1, 1, 1, np.nan, 1, 1, 1]}, dtype=float)
    assert confirmed(s, 1.0, 3)["X"].tolist() == [False, False, False, False, False, True, False, False, False, True]
    assert confirmed(s, 0.0, 1)["X"].tolist() == [False, False, True] + [False] * 7


def test_orders_fill_at_next_open_not_on_signal_day():
    n = 320
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = pd.DataFrame({"UP": np.linspace(50, 150, n), "FLAT": np.full(n, 80.0)}, index=idx)
    open_ = close * 0.999 + 0.37  # distinct from the close so a same-day fill would show
    m = Market(open=open_, high=close * 1.01, low=close * 0.99, close=close,
               universe=pd.DataFrame(True, index=idx, columns=close.columns),
               sector=pd.Series({"UP": "Industrials", "FLAT": "Utilities"}))
    res = simulate(m, Params(start=str(idx[0].date()), n_positions=2))
    t = res.trades.sort_values("entry").iloc[0]
    assert t["ticker"] == "UP"
    # first day UP is a candidate: golden cross, 3 bullish SuperTrend days and a 12-1 month score
    score = m.score("composite")["UP"]
    golden = (m.sma(50) > m.sma(200))["UP"]
    bull3 = confirmed(m.st(10, 3.0), 1.0, 3)["UP"]
    first_signal = (score.notna() & golden & bull3).idxmax()
    i = idx.get_loc(first_signal)
    assert t["entry"] == idx[i + 1]
    assert t["entry_px"] == pytest.approx(open_["UP"].iloc[i + 1])

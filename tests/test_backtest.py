"""Backtest building blocks: SuperTrend, the N-day confirmation, fill timing (no look-ahead),
the market filter, the volatility skip and the earnings blackout."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from backtest.indicators import atr_wilder, confirmed, event_window, regime, supertrend_bull
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


def _market(close: pd.DataFrame, **kw) -> Market:
    open_ = close * 0.999 + 0.37  # distinct from the close so a same-day fill would show
    return Market(open=open_, high=close * 1.01, low=close * 0.99, close=close,
                  universe=pd.DataFrame(True, index=close.index, columns=close.columns),
                  sector=pd.Series("Industrials", index=close.columns), **kw)


def _trend(n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"UP": np.linspace(50, 150, n), "FLAT": np.full(n, 80.0)}, index=idx)


def test_orders_fill_at_next_open_not_on_signal_day():
    close = _trend(320)
    idx = close.index
    m = _market(close)
    open_ = m.open
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


def test_regime_switches_only_after_three_closes():
    idx = pd.bdate_range("2020-01-01", periods=11)
    spy = pd.Series([100, 100, 100, 100, 100, 90, 85, 80, 95, 100, 105], index=idx, dtype=float)
    # 3-day average: below on days 5, 6, 7 -> off from day 7; above on 8, 9, 10 -> on again on day 10
    assert regime(spy, 3, 3).tolist() == [True] * 7 + [False] * 3 + [True]
    assert regime(spy, 3, 1).tolist() == [True] * 5 + [False] * 3 + [True] * 3


def test_event_window():
    idx = pd.bdate_range("2020-01-01", periods=10)
    ev = pd.DataFrame({"ticker": ["A", "B", "ZZZ"], "session": [idx[5], idx[0], idx[3]]})
    w = event_window(ev, idx, ["A", "B"], 2)
    assert w["A"].tolist() == [False] * 3 + [True] * 5 + [False] * 2
    assert w["B"].tolist() == [True] * 3 + [False] * 7


def test_earnings_blackout_sells_five_sessions_before_and_waits_after():
    close = _trend(420)
    idx = close.index
    e = 360  # first session that reacts to the release
    m = _market(close, earnings=pd.DataFrame({"ticker": ["UP"], "session": [idx[e]]}))
    res = simulate(m, Params(start=str(idx[0].date()), n_positions=2, earnings_days=5))
    t = res.trades[res.trades["ticker"] == "UP"].sort_values("entry")
    assert t.iloc[0]["entry"] < idx[e - 5]
    assert t.iloc[0]["exit"] == idx[e - 5] and t.iloc[0]["reason"] == "earnings"
    assert t.iloc[0]["exit_px"] == pytest.approx(m.open["UP"].iloc[e - 5])
    assert t.iloc[1]["entry"] == idx[e + 6]  # first buy allowed 6 sessions after the reaction session
    # without the blackout the same stock is simply held through
    plain = simulate(m, Params(start=str(idx[0].date()), n_positions=2))
    assert len(plain.trades) == 1 and plain.trades.iloc[0]["reason"] == "open"


def test_market_filter_goes_to_cash_and_comes_back():
    close = _trend(420)
    idx = close.index
    spy = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    spy.iloc[330:360] = 50.0  # a 30-session crash below the 200-day average
    m = _market(close, benchmark=spy)
    res = simulate(m, Params(start=str(idx[0].date()), n_positions=2, market_filter=True))
    t = res.trades[res.trades["ticker"] == "UP"].sort_values("entry")
    assert t.iloc[0]["reason"] == "market" and t.iloc[0]["exit"] == idx[333]  # 3rd close below is day 332
    assert res.invested.loc[idx[333]:idx[359]].eq(0).all()
    on = regime(spy, 200, 3)
    back = on.loc[idx[360]:].idxmax()  # first risk-on day after the crash
    assert t.iloc[1]["entry"] == idx[idx.get_loc(back) + 1]


def test_skip_most_volatile_never_buys_it():
    close = _trend(420)
    wiggle = np.where(np.arange(len(close)) % 2 == 0, 1.01, 0.99)
    close["VOL"] = close["UP"] * wiggle  # same trend, about 30% annualised volatility
    m = _market(close)
    p = Params(start=str(close.index[0].date()), n_positions=3, sector_cap=3)
    assert "VOL" in set(simulate(m, p).trades["ticker"])
    # volatility ranks across the three stocks: FLAT 1/3, UP 2/3, VOL 1
    traded = set(simulate(m, dataclasses.replace(p, max_vol_pct=0.7)).trades["ticker"])
    assert "VOL" not in traded and "UP" in traded

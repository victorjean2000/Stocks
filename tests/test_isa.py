"""ISA strategies: US-close signals traded at the next London open, the short-side mirror rules. No network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.isa import isa_run, london_wanted, rip_signals
from backtest.daily import dip_signals


def test_a_us_close_is_traded_at_the_next_london_session():
    us = pd.Series([False, True, True, False], index=pd.to_datetime(["2026-11-13", "2026-11-16", "2026-11-17",
                                                                      "2026-11-18"]))
    lse = pd.to_datetime(["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19"])
    # London Monday sees Friday's US close, Tuesday sees Monday's, ...
    assert london_wanted(us, lse).tolist() == [False, True, True, False]


def test_isa_trades_compound_to_the_equity():
    lse = pd.bdate_range("2026-01-05", periods=10)
    px = pd.Series(np.linspace(100, 120, 10), index=lse)
    funds = {"LQQ3.L": pd.DataFrame({"open": px * 0.99, "close": px})}
    us = pd.Series([False, True, True, False, False, True, False, False, False, False],
                   index=lse - pd.Timedelta(days=1))
    eq, t = isa_run(funds, {"LQQ3.L": us}, {"LQQ3.L": pd.Series(0.0, index=us.index)}, 1, np.zeros(10), lse, 0)
    assert len(t) == 2 and t["entry"].tolist() == [lse[1], lse[5]]
    assert eq.iloc[-1] == pytest.approx(float(np.prod(1 + t["ret"])))


def test_short_rules_mirror_the_dip_rules_below_the_200_day_average():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2020-01-01", periods=400)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(-0.001, 0.01, 400))), index=idx)
    h, l = c * 1.01, c * 0.99
    e, x = rip_signals(c, h, l, ("low", 5), "nhigh")
    # a short entry is a 5-day high close below the 200-day average; it exits at a 5-day low close
    assert (e == ((c >= c.rolling(5).max()) & (c < c.rolling(200).mean()))).all()
    assert (x == (c <= c.rolling(5).min())).all()
    mirror_e, _ = dip_signals(-c, -l, -h, ("low", 5), "sma5", False)
    assert ((e & (c < c.rolling(200).mean())) == (mirror_e & (c < c.rolling(200).mean()))).all()


def test_a_consolidation_is_undone_and_real_moves_are_kept():
    from backtest.isa import adjust_actions
    idx = pd.bdate_range("2024-01-01", periods=60)
    under = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 60))), index=idx)
    fund = (1 + 3 * under.pct_change().fillna(0)).cumprod() * 2.0
    raw = fund.copy()
    raw.iloc[30:] *= 75  # a 1:75 consolidation on day 30, left in the prices
    df = pd.DataFrame({"open": raw, "high": raw, "low": raw, "close": raw})
    fixed, fixes = adjust_actions(df, under, 3)
    assert [f[2] for f in fixes] == [75.0]
    assert np.allclose(fixed["close"], fund * 75)  # earlier prices scaled up to the new basis
    split = df.copy()
    split.iloc[30:] = (fund.iloc[30:] / 20).to_frame().to_numpy().repeat(4, axis=1)  # a 1:20 split instead
    split.iloc[:30] = fund.iloc[:30].to_frame().to_numpy().repeat(4, axis=1)
    fixed, fixes = adjust_actions(split, under, 3)
    assert [f[2] for f in fixes] == [0.05] and np.allclose(fixed["close"], fund / 20)

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


def test_without_years_chains_the_other_years():
    from backtest.isa import without_years
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    r = pd.Series(0.0, index=idx)
    r[idx.year == 2020] = 0.01  # all the growth in 2020
    eq = (1 + r).cumprod()
    assert without_years(eq, [2020])["CAGR"] == pytest.approx(0.0, abs=1e-12)
    assert without_years(eq)["CAGR"] > 0.5


def test_blend_resets_the_split_each_january():
    from backtest.isa import blend
    idx = pd.to_datetime(["2020-06-01", "2020-12-31", "2021-06-01", "2021-12-31"])
    flat = pd.Series(1.0, index=idx)
    up = pd.Series([1.0, 2.0, 2.0, 4.0], index=idx)  # doubles in each year
    eq = blend(flat, up)
    assert eq.iloc[1] == pytest.approx(1.5)  # half flat, half doubled
    assert eq.iloc[3] == pytest.approx(1.5 * 1.5)  # back to half each in January, the same again (not 2.5)


def test_index_edge_measures_the_moves_after_a_close_near_the_low():
    from backtest.isa import index_edge
    idx = pd.bdate_range("2020-01-01", periods=6)
    c = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=idx)
    bars = pd.DataFrame({"c": c, "h": c + 1, "l": c - 1})
    bars.loc[idx[2], "l"] = 102.0  # day 2 closes at its low; every other day mid-range
    e = index_edge(bars, start="2020-01-01")
    assert e["days"] == 1 and e["next day"] == pytest.approx(103 / 102 - 1)
    assert e["next 3 days"] == pytest.approx(105 / 102 - 1)

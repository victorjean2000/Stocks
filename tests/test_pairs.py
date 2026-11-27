"""Pairs trading (backtest/pairs.py): spread statistics, the z-score and distance rules, and the two-leg engine. No
network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.pairs import (adf_t, closest_pairs, ggr_positions, half_life, positions, run_pair, zscore)


def _ar1(phi: float, n: int, seed: int = 0, sd: float = 1.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, sd)
    return pd.Series(x, index=pd.bdate_range("2016-01-01", periods=n))


def test_half_life_and_stationarity_of_a_mean_reverting_spread():
    s = _ar1(0.9, 3000)
    assert half_life(s) == pytest.approx(-np.log(2) / np.log(0.9), rel=0.15)
    assert adf_t(s) < -2.86  # gaps close
    walk = pd.Series(np.cumsum(np.random.default_rng(1).normal(0, 1, 3000)), index=s.index)
    assert adf_t(walk) > -2.86 and half_life(walk) > 200  # a random walk: no reliable closing


def test_positions_open_beyond_entry_and_close_at_zero_or_the_time_stop():
    idx = pd.bdate_range("2020-01-01", periods=8)
    z = pd.Series([0.0, 2.5, 1.0, -0.1, -2.6, -2.0, -1.0, 0.2], index=idx)
    assert list(positions(z, 2.0)) == [0, -1, -1, 0, 1, 1, 1, 0]
    assert list(positions(z, 2.0, max_hold=1)) == [0, -1, 0, 0, 1, 0, 0, 0]


def _px(a_close, b_close):
    idx = pd.bdate_range("2020-01-01", periods=len(a_close))
    a = pd.DataFrame({"o": a_close, "c": a_close}, index=idx, dtype=float)
    b = pd.DataFrame({"o": b_close, "c": b_close}, index=idx, dtype=float)
    return {"A": a, "B": b}, idx


def test_long_one_leg_short_the_other_pays_the_difference_less_costs():
    px, idx = _px([100, 100, 100, 110, 110, 110], [50, 50, 50, 50, 50, 50])
    pos = np.array([0, 1, 1, 1, 0, 0])  # decided at closes 1-3; long A / short B from open 2 to open 5
    eq, tr = run_pair(px, "A", "B", pos, idx, start=str(idx[1].date()), cost_bps=10, borrow=0.0)
    (t,) = tr.itertuples()
    assert (t.entry, t.exit, t.days) == (idx[2], idx[5], 3)
    # entry: 2 legs x 10 bps; exit: 10 bps on each leg's value (1.10 and 1.00); A +10%, B flat
    assert t.ret == pytest.approx((1 - 0.002) * (1 + 0.10 - 0.001 * 2.10) / 1 - 1, abs=1e-12)
    assert eq.iloc[-1] == pytest.approx(1 + t.ret)


def test_a_truly_mean_reverting_pair_makes_money_before_costs():
    n = 1500
    rng = np.random.default_rng(3)
    common = np.cumsum(rng.normal(0, 0.01, n))
    gap = _ar1(0.85, n, seed=4, sd=0.01).to_numpy()
    px, idx = _px(100 * np.exp(common + gap), 100 * np.exp(common))
    z = zscore(px["A"]["c"], px["B"]["c"], 20)
    eq, tr = run_pair(px, "A", "B", positions(z, 2.0), idx, start=str(idx[30].date()), cost_bps=0.0, borrow=0.0)
    assert eq.iloc[-1] > 1.2 and (tr["ret"] > 0).mean() > 0.6


def test_ggr_rule_opens_beyond_two_sigma_and_closes_when_prices_cross():
    na = np.array([1.0, 1.05, 1.10, 1.02, 0.99, 1.00])
    nb = np.ones(6)
    assert list(ggr_positions(na, nb, sigma=0.03)) == [0, 0, -1, -1, 0, 0]  # 0.10 > 2 x 0.03; crosses on day 4


def test_closest_pairs_picks_the_two_that_move_together():
    idx = pd.bdate_range("2020-01-01", periods=50)
    rng = np.random.default_rng(5)
    base = 100 * np.cumprod(1 + rng.normal(0, 0.01, 50))
    close = pd.DataFrame({"A": base, "B": base * (1 + rng.normal(0, 0.001, 50)),
                          "C": 100 * np.cumprod(1 + rng.normal(0, 0.02, 50))}, index=idx)
    sector = pd.Series("Utilities", index=close.columns)
    (a, b, sd), = closest_pairs(close, sector, n_pairs=1)
    assert {a, b} == {"A", "B"} and sd < 0.01

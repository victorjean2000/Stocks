"""The ISA rule tests (backtest/isa_tests.py): clocks, states, the two-moment London engine, London-close decisions,
the random-timing shuffle, the bootstrap, walk-forward, sizing and the 30-minute session table. No network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import isa
from backtest.isa_tests import (block_bootstrap, event_wants, london_close_mark, london_run, runs_of, session_table,
                                shuffle_runs, us_state, walk_forward)
from backtest.momentum_trend import stats


def test_london_close_mark_follows_both_clocks():
    days = pd.to_datetime(["2026-03-10", "2026-06-10", "2026-10-28", "2026-12-10"])
    # New York changes its clocks earlier in spring and later in autumn than London
    assert list(london_close_mark(days)) == ["1230", "1130", "1230", "1130"]


def test_us_state_stop_and_time_stop():
    idx = pd.bdate_range("2024-01-01", periods=8)
    c = pd.Series([100, 98, 97, 95, 96, 99, 100, 101.0], index=idx)
    entry, leave = pd.Series([True] + [False] * 7, index=idx), pd.Series(False, index=idx)
    assert us_state(entry, leave, c).all()
    assert list(us_state(entry, leave, c, stop=0.04)) == [True, True, True] + [False] * 5  # 95 <= 96
    assert list(us_state(entry, leave, c, max_days=2)) == [True, True] + [False] * 6


def _funds(n: int, seed: int = 1) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    lse = pd.bdate_range("2024-01-01", periods=n)
    out = {}
    for f in ("LQQ3.L", "3LUS.L"):
        c = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
        out[f] = pd.DataFrame({"open": c * (1 + rng.normal(0, 0.01, n)), "close": c}, index=lse)
    return out


def test_two_moment_engine_matches_the_isa_engine():
    funds = _funds(30)
    lse = funds["LQQ3.L"].index
    rng = np.random.default_rng(2)
    states = {f: pd.Series(rng.random(30) < 0.4, index=lse - pd.Timedelta(days=1)) for f in funds}
    eq1, t1 = isa.isa_run(funds, states, {f: pd.Series(0.0, index=lse) for f in funds}, 2, np.full(30, 1e-4), lse, 0)
    W = np.column_stack([isa.london_wanted(states[f], lse) for f in funds])
    eq2, t2, _ = london_run(funds, W, W, np.full(30, 1e-4), lse, 0)
    assert np.allclose(eq1.to_numpy(), eq2.reindex(eq1.index).to_numpy()) and len(t1) == len(t2)


def test_a_buy_at_the_london_close_fills_at_the_close():
    funds = {"LQQ3.L": _funds(8)["LQQ3.L"]}
    lse = funds["LQQ3.L"].index
    wo, wc = np.zeros((8, 1), bool), np.zeros((8, 1), bool)
    wc[2], wo[3], wc[3] = True, True, True  # bought at session 2's close, sold at session 4's open
    _, tr, _ = london_run(funds, wo, wc, np.zeros(8), lse, 0)
    (r,) = tr.itertuples()
    px = funds["LQQ3.L"]
    assert (r.entry, r._3, r.exit, r._5) == (lse[2], "close", lse[4], "open")
    assert r.ret == pytest.approx(0.999 ** 2 * px["open"].iloc[4] / px["close"].iloc[2] - 1)


def _signal_day():
    idx = pd.bdate_range("2024-01-01", periods=10)
    c = pd.Series([100, 100, 100, 100, 95, 96, 97, 101, 102, 103.0], index=idx)
    t = pd.DataFrame({"o": c, "h": c + 1, "l": c - 1, "c": c})
    t.loc[idx[4], ["h", "l"]] = [100.0, 95.0]  # day 4 closes at its low: the signal
    st = pd.DataFrame({"lc": np.nan, "lc_hi": np.nan, "lc_lo": np.nan}, index=idx)
    entry = (t["c"] - t["l"]) / (t["h"] - t["l"]) < 0.1
    leave = t["c"] > t["c"].rolling(5).mean()
    return idx, t, st, entry, leave


def test_an_early_exit_sells_at_the_london_close():
    idx, t, st, entry, leave = _signal_day()
    lse = idx + pd.Timedelta(days=0)
    base_open, base_close = event_wants(t, st, entry, leave, lse)
    assert np.array_equal(base_open, isa.london_wanted(us_state(entry, leave, t["c"]), lse))
    assert np.array_equal(base_close, base_open) and base_open[5] and base_open[7] and not base_open[8]
    st.loc[idx[6], "lc"] = 98.0  # above the mean of the 4 closes before day 6 (97.75)
    wo, wc = event_wants(t, st, entry, leave, lse, early_exit=True)
    assert wo[6] and not wc[6] and not wo[7]  # sold at day 6's London close, not bought back


def test_an_early_entry_buys_at_the_london_close():
    idx, t, st, entry, leave = _signal_day()
    st.loc[idx[4], ["lc", "lc_hi", "lc_lo"]] = [96.0, 100.0, 95.8]  # near the day's low so far, 4% down
    wo, wc = event_wants(t, st, entry, leave, idx, early_entry=0.1)
    assert not wo[4] and wc[4] and wo[5]


def test_shuffled_spells_keep_their_number_and_lengths():
    w = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1], bool)
    r = shuffle_runs(w, np.random.default_rng(0))
    spells = lambda a: sorted(n for v, n in runs_of(a) if v)  # noqa: E731
    gaps = lambda a: sorted(n for v, n in runs_of(a) if not v)  # noqa: E731
    assert len(r) == len(w) and spells(r) == spells(w) and gaps(r) == gaps(w)


def test_bootstrap_of_steady_growth_has_no_spread():
    idx = pd.bdate_range("2020-01-01", periods=600)
    eq = pd.Series(1.0005 ** np.arange(600), index=idx)
    b = block_bootstrap(eq, pd.Series(1.0, index=idx), n=50, block=10)
    assert all(x == pytest.approx(stats(eq)["CAGR"]) for x in b["cagr"])
    assert b["dd"] == (0.0, 0.0, 0.0) and b["beats"] == 1.0


def test_walk_forward_trades_the_best_earlier_setting():
    idx = pd.bdate_range("2019-01-01", "2020-12-31")
    a = pd.Series(np.where(idx.year == 2019, 1.001, 1.0), index=idx).cumprod()
    b = pd.Series(np.where(idx.year == 2019, 1.0, 1.001), index=idx).cumprod()
    wf, chosen = walk_forward({"A": a, "B": b}, first=2020)
    assert chosen == {2020: "A"} and wf.iloc[-1] == pytest.approx(1.0)  # A won 2019, then went flat


def test_run_can_buy_part_of_a_place():
    from backtest.daily import run
    n = 4
    O = C = np.full((n, 1), 10.0)
    enter = np.zeros((n, 1), bool)
    enter[0] = True
    _, inv, _ = run(O, C, enter, np.zeros((n, 1), bool), np.ones((n, 1)), 1, 0.0, np.zeros(n), 0,
                    size=np.full((n, 1), 0.5))
    assert inv[1] == pytest.approx(0.5)


def test_session_table_reads_half_hours_pre_market_and_the_london_close():
    ny = "America/New_York"
    day = pd.Timestamp("2026-06-10")
    starts = [pd.Timestamp("2026-06-10 04:00", tz=ny)] + \
        list(pd.date_range("2026-06-10 09:30", "2026-06-10 15:30", freq="30min", tz=ny)) + \
        [pd.Timestamp("2026-06-10 16:00", tz=ny)]
    c = np.arange(len(starts), dtype=float) + 100
    bars = pd.DataFrame({"ticker": "QQQ", "ts": starts, "o": c - 0.5, "h": c + 1, "l": c - 1, "c": c, "v": 1,
                         "date": day})
    cal = pd.DataFrame({"date": [day], "open_ts": [pd.Timestamp("2026-06-10 09:30", tz=ny)],
                        "close_ts": [pd.Timestamp("2026-06-10 16:00", tz=ny)]})
    daily = pd.DataFrame({"o": [100.5], "h": [115.0], "l": [99.0], "c": [113.0]}, index=[day])
    st = session_table("QQQ", bars, cal, daily).iloc[0]
    # bars from 9:30: closes 101, 102, ...; the bar ending 11:30 is the fourth (close 104)
    assert st["m1130"] == 104 and st["hi1130"] == 105 and st["lo1130"] == 100
    assert st["lc"] == 104 and st["pm"] == 99.5 and st["ah"] == c[-1]

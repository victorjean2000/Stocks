"""The daily strategy engine: next-open fills, costs, position sizing, the time stop. No network."""
from __future__ import annotations

import numpy as np
import pytest

from backtest.daily import run


def _flags(n, days, cols=1, col=0):
    a = np.zeros((n, cols), bool)
    a[list(days), col] = True
    return a


def test_signal_at_a_close_fills_at_the_next_open_with_costs():
    n = 8
    O = np.array([[10.0], [11.0], [12.0], [13.0], [14.0], [15.0], [16.0], [17.0]])
    C = O + 0.5  # closes differ from opens, so a same-day fill would show
    eq, inv, trades = run(O, C, _flags(n, [1]), _flags(n, [4]), np.ones((n, 1)), 1, 10.0, np.zeros(n), 0)
    (j, i_in, i_out, px_in, px_out, ret), = trades
    assert (i_in, i_out, px_in, px_out) == (2, 5, 12.0, 15.0)  # bought at the open after day 1, sold after day 4
    assert ret == pytest.approx(0.999 ** 2 * 15 / 12 - 1)
    assert eq[-1] == pytest.approx(1 + ret) and inv[-1] == 0 and inv[3] == 1
    assert eq[0] == 1.0 and np.isnan(eq[:0]).all()


def test_best_scores_fill_the_free_places_with_equal_slices():
    n, k = 5, 3
    O = np.full((n, k), 10.0)
    C = np.full((n, k), 10.0)
    enter = np.zeros((n, k), bool)
    enter[0] = True
    score = np.tile([1.0, 3.0, 2.0], (n, 1))
    _, inv, _ = run(O, C, enter, np.zeros((n, k), bool), score, 2, 0.0, np.zeros(n), 0)
    eq, _, trades = run(O, C, enter, np.zeros((n, k), bool), score, 1, 0.0, np.zeros(n), 0)
    assert inv[1] == pytest.approx(1.0)  # two places, both filled
    one = run(O, C, enter, _flags(n, [2], k, 1), score, 1, 0.0, np.zeros(n), 0)[2]
    assert [t[0] for t in one] == [1]  # the best score (column 1) took the only place, then sold after day 2


def test_time_stop_and_cash_interest():
    n = 12
    O = C = np.full((n, 1), 10.0)
    _, _, trades = run(O, C, _flags(n, [0]), np.zeros((n, 1), bool), np.ones((n, 1)), 1, 0.0, np.zeros(n), 0, 3)
    assert (trades[0][1], trades[0][2]) == (1, 4)  # held 3 sessions, open to open
    eq, _, _ = run(O, C, np.zeros((n, 1), bool), np.zeros((n, 1), bool), np.ones((n, 1)), 1, 0.0,
                   np.full(n, 1e-4), 0)
    assert eq[-1] == pytest.approx(1.0001 ** (n - 1))  # all cash: T-bill interest from the day after the start


def test_a_signal_on_the_last_day_is_not_traded():
    n = 4
    O = C = np.full((n, 1), 10.0)
    eq, inv, trades = run(O, C, _flags(n, [n - 1]), np.zeros((n, 1), bool), np.ones((n, 1)), 1, 0.0, np.zeros(n), 0)
    assert trades == [] and inv[-1] == 0


def test_todays_bar_is_dropped_until_its_close_is_final():
    import datetime as dt

    import pandas as pd

    from stockdata.alpaca import drop_unfinished
    bars = pd.DataFrame({"date": pd.to_datetime(["2026-11-18", "2026-11-19"]), "c": [1.0, 2.0]})
    at = lambda s: dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)  # noqa: E731
    assert len(drop_unfinished(bars, now=at("2026-11-19 15:30"))) == 1  # 10:30 in New York: session running
    assert len(drop_unfinished(bars, now=at("2026-11-19 21:15"))) == 1  # 16:15: SIP closes not final yet
    assert len(drop_unfinished(bars, now=at("2026-11-19 21:17"))) == 2

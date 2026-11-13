"""The momentum refinement study, and the paper account trading its top-30 buffer exactly as backtested."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from backtest.momentum_refine import BASE, BUFFER, OFFSETS, REFINED, variants
from backtest.momentum_trend import Market, Params, simulate
from backtest.strategies import START
from trading.targets import KEEP_RANK, N_STOCKS, POOL, buffered_top


def test_variants_are_unique_valid_params():
    v = variants()
    labels = [label for _, label, _ in v]
    assert len(labels) == len(set(labels))  # labels key the results
    assert v[0][0] == "Base" and v[0][2] == {}
    assert BUFFER in labels and REFINED in labels
    fields = {f.name for f in dataclasses.fields(Params)}
    for _, _, kw in v:
        assert set(kw) <= fields
        Params(**(BASE | kw))


def test_base_is_the_original_paper_strategy_and_the_buffer_is_what_it_trades_now():
    # the paper account's original rules: top 10 of the 500 largest by 6-1 month return, equal weight, rebalanced
    # after each month end, no trend rules; the strategy search starts trading on START
    p = Params(**BASE)
    assert (p.pool, p.momentum, p.n_positions, p.monthly, p.reweight) == (500, "r6_1", 10, True, True)
    assert not (p.golden_cross or p.trend_entry or p.trend_exit or p.market_filter)
    assert p.sector_cap >= p.n_positions and p.keep_rank is None and p.take_profit is None
    assert START == "2017-02-01" and OFFSETS[0] == 0
    # since 2026-11-13 the account keeps a holding while it ranks in the top 30 (trading/targets.py)
    live = Params(**(BASE | dict(variants_by_label()[BUFFER])))
    assert (live.keep_rank, live.n_positions, live.pool) == (KEEP_RANK, N_STOCKS, POOL)


def variants_by_label() -> dict[str, dict]:
    return {label: kw for _, label, kw in variants()}


def test_the_paper_rule_picks_what_the_backtest_holds():
    """trading.targets.buffered_top, applied at each month end to the previous picks, gives the same holdings
    as simulate(keep_rank=30) on a market whose ranking changes every day."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2019-01-01", periods=200)
    names = [f"S{k:02d}" for k in range(60)]
    close = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(0, 0.02, (200, 60)), axis=0)), index=idx, columns=names)
    scores = pd.DataFrame(np.cumsum(rng.normal(0, 1, (200, 60)), axis=0), index=idx, columns=names)
    m = Market(open=close * 0.999 + 0.37, high=close * 1.01, low=close * 0.99, close=close,
               universe=pd.DataFrame(True, index=idx, columns=names), sector=pd.Series("Tech", index=names))
    m._cache[("score", "r6_1")] = scores
    p = Params(**(BASE | dict(pool=None, cash_yield=False, keep_rank=KEEP_RANK, start=str(idx[0].date()))))
    trades = simulate(m, p).trades
    held: set[int] = set()
    month_ends = np.flatnonzero(idx.month[1:] != idx.month[:-1])
    for i in month_ends:
        held = set(buffered_top(scores.to_numpy()[i], np.ones(60, bool), N_STOCKS, KEEP_RANK, held))
        fill = idx[i + 1]
        sim = {t.ticker for t in trades.itertuples() if t.entry <= fill and (pd.isna(t.exit) or t.exit > fill)}
        assert sim == {names[j] for j in held}, fill
    assert len(month_ends) >= 8

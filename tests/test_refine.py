"""The momentum refinement study: its base matches the paper strategy and every variant is a valid Params."""
from __future__ import annotations

import dataclasses

from backtest.momentum_refine import BASE, BUFFER, OFFSETS, REFINED, variants
from backtest.momentum_trend import Params
from backtest.strategies import START


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


def test_base_is_the_paper_momentum_strategy():
    # trading/targets.py momentum_target: top 10 of the 500 largest by 6-1 month return, equal weight, rebalanced
    # after each month end, no trend rules; the strategy search starts trading on START
    p = Params(**BASE)
    assert (p.pool, p.momentum, p.n_positions, p.monthly, p.reweight) == (500, "r6_1", 10, True, True)
    assert not (p.golden_cross or p.trend_entry or p.trend_exit or p.market_filter)
    assert p.sector_cap >= p.n_positions and p.keep_rank is None and p.take_profit is None
    assert START == "2017-02-01" and OFFSETS[0] == 0

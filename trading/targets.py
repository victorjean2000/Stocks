"""Today's target for each paper-traded strategy, computed with the backtest's own code."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockdata import alpaca as market_data

from .broker import Paper

NY = "America/New_York"
# Momentum: 10 stocks from the 500 largest by 6-1 month return. Since 2026-11-13 a holding is sold only when it
# ranks below KEEP_RANK, not below 10 (backtest/MOMENTUM_REFINE.md: 'Keep a holding while it ranks in the top 30').
N_STOCKS, POOL, KEEP_RANK = 10, 500, 30


class StaleData(RuntimeError):
    pass


@dataclass
class Target:
    strategy: str
    weights: dict[str, float]  # symbol -> fraction of the account
    decided: pd.Timestamp  # close the target is based on
    data_through: pd.Timestamp  # latest close in the data used
    tag: str  # prefix of the client_order_id of orders acting on this decision
    once: bool  # True: a full rebalance once per decision; False: keep holdings matched to the target
    info: dict = field(default_factory=dict)


def sessions(paper: Paper, back_days: int = 450) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
    """Trading sessions from the account's calendar (a few weeks ahead too) and the last one whose
    closing prices are final (16 minutes after the close, for the SIP feed)."""
    now = pd.Timestamp.now(tz=NY)
    cal = paper.calendar((now - pd.Timedelta(days=back_days)).strftime("%Y-%m-%d"),
                         (now + pd.Timedelta(days=21)).strftime("%Y-%m-%d"))
    days = pd.DatetimeIndex([pd.Timestamp(c["date"]) for c in cal])
    closes = [pd.Timestamp(f"{c['date']} {c['close']}", tz=NY) for c in cal]
    done = [d for d, c in zip(days, closes) if c + pd.Timedelta(minutes=16) <= now]
    return days, done[-1]


def month_end_decision(days: pd.DatetimeIndex, upto: pd.Timestamp) -> pd.Timestamp:
    """The last session of the most recent month that has ended by `upto` (its close is the decision)."""
    nxt = days[1:]
    ends = days[:-1][(nxt.month != days[:-1].month) & (days[:-1] <= upto)]
    return ends[-1]


def lqq3_target(paper: Paper) -> Target:
    """TQQQ (3x Nasdaq-100, the US stand-in for London-listed LQQ3) while QQQ's close is above its 200-day
    average, else BIL (backtest: 'TQQQ when QQQ > 200-day avg' / 'LQQ3 (sim) when QQQ > 200-day avg')."""
    from backtest.indicators import regime
    days, last = sessions(paper)
    bars = market_data.fetch_bars(paper.session, ["QQQ"], (last - pd.Timedelta(days=440)).strftime("%Y-%m-%d"),
                                  market_data.safe_end(), "all", "sip")
    close = bars.set_index("date")["c"].astype(float).sort_index()
    close = close[close.index <= last]
    if close.index[-1] != last:
        raise StaleData(f"QQQ bars end {close.index[-1].date()}, last completed session is {last.date()}")
    on = bool(regime(close, 200, 1).iloc[-1])
    avg = float(close.rolling(200).mean().iloc[-1])
    return Target("lqq3", {"TQQQ" if on else "BIL": 1.0}, last, last, f"lqq3-{last:%Y%m%d}", once=False,
                  info={"QQQ close": round(float(close.iloc[-1]), 2), "QQQ 200-day average": round(avg, 2),
                        "signal": "above: hold TQQQ (LQQ3 in the ISA)" if on else "below: hold BIL (cash in the ISA)"})


def ranking(score: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Columns of the ok stocks with a score, best first (ties keep column order, as in the backtest)."""
    idx = np.flatnonzero(ok & np.isfinite(score))
    return idx[np.argsort(-score[idx], kind="stable")]


def buffered_top(score: np.ndarray, ok: np.ndarray, n: int, keep: int, held: set[int]) -> list[int]:
    """The n stocks to hold after a monthly decision: current holdings that still rank in the top `keep`
    (best first), then the best-ranked stocks not held. backtest.momentum_trend.simulate with keep_rank
    makes the same choice."""
    order = [int(j) for j in ranking(score, ok)]
    kept = [j for j in order[:keep] if j in held][:n]
    return kept + [j for j in order if j not in held][:n - len(kept)]


def momentum_target(paper: Paper) -> Target:
    """At the last month end, rank the POOL largest stocks by 6-1 month return. Keep each holding that still
    ranks in the top KEEP_RANK and fill up to N_STOCKS with the best-ranked others, equal weight (backtest:
    'Keep a holding while it ranks in the top 30', backtest/MOMENTUM_REFINE.md). Needs the stockdata panel up
    to the last completed session."""
    from backtest.strategies import build_lab, momentum_pool
    days, last = sessions(paper)
    lab = build_lab()
    if lab.dates[-1] != last:
        raise StaleData(f"stockdata panel ends {lab.dates[-1].date()}, last completed session is {last.date()}: "
                        "run `python -m stockdata update` first")
    decision = month_end_decision(days, last)
    i = lab.dates.get_loc(decision)
    names = [str(s) for s in lab.m.close.columns]
    col = {s: j for j, s in enumerate(names)}
    held = {p["symbol"] for p in paper.positions() if float(p["qty"]) > 0}
    score, ok = lab.x["r6_1"][i], momentum_pool(lab, POOL)[i]
    picks = buffered_top(score, ok, N_STOCKS, KEEP_RANK, {col[s] for s in held if s in col})
    rank = {int(j): k + 1 for k, j in enumerate(ranking(score, ok))}
    where = lambda s: f"#{rank[col[s]]}" if s in col and col[s] in rank else "not ranked"  # noqa: E731
    weights = {names[j]: 1.0 / N_STOCKS for j in picks}
    return Target("momentum", weights, decision, last, f"momentum-{decision:%Y%m%d}", once=True,
                  info={"ranked at close of": str(decision.date()),
                        "rank and 6-1 month return": {names[j]: f"#{rank[j]} ({score[j]:+.1%})" for j in picks},
                        f"kept (still in the top {KEEP_RANK})": sorted(s for s in held if s in weights),
                        f"sold (below #{KEEP_RANK})": {s: where(s) for s in sorted(held - set(weights))},
                        "bought": sorted(set(weights) - held)})


TARGETS = {"momentum": momentum_target, "lqq3": lqq3_target}

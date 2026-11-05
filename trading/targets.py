"""Today's target for each paper-traded strategy, computed with the backtest's own code."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from stockdata import alpaca as market_data

from .broker import Paper

NY = "America/New_York"


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


def momentum_target(paper: Paper) -> Target:
    """Top 10 of the 500 largest by 6-1 month return at the last month end (backtest: 'Momentum 6-1 month,
    top 10 of largest 500'). Needs the stockdata panel up to the last completed session."""
    from backtest.strategies import build_lab, momentum_pool, top_n
    days, last = sessions(paper)
    lab = build_lab()
    if lab.dates[-1] != last:
        raise StaleData(f"stockdata panel ends {lab.dates[-1].date()}, last completed session is {last.date()}: "
                        "run `python -m stockdata update` first")
    decision = month_end_decision(days, last)
    i = lab.dates.get_loc(decision)
    picks = top_n(lab.x["r6_1"][i], momentum_pool(lab, 500)[i], 10)
    names = lab.m.close.columns
    weights = {str(names[j]): w for j, w in picks.items()}
    r = lab.x["r6_1"][i]
    return Target("momentum", weights, decision, last, f"momentum-{decision:%Y%m%d}", once=True,
                  info={"ranked at close of": str(decision.date()),
                        "6-1 month returns": {str(names[j]): round(float(r[j]), 3) for j in picks}})


TARGETS = {"momentum": momentum_target, "lqq3": lqq3_target}

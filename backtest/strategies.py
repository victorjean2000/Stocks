"""Strategy search: many strategy ideas on the same data, judged on years they were not tuned on.

Every strategy here
- decides at a close and trades at the next open (no look-ahead),
- pays 10 bps per side on stocks and 5 bps per side on ETFs,
- earns T-bill returns (BIL) on cash it does not use,
- uses the published standard settings of its idea; other settings are shown as a grid.

The tuning years are 2017-2021; 2022-2026 is the out-of-sample check. Picks for the
combinations at the end were made on the tuning years only.

Run: python -m backtest.strategies   (writes backtest/STRATEGIES.md, about 2 minutes)
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockdata import alpaca
from stockdata.config import REPO_ROOT, paths

from .indicators import regime
from .momentum_trend import Market, Params, load_market, md, simulate, stats

log = logging.getLogger(__name__)

START = "2017-02-01"  # first trade; the first decision is at the close before
SPLIT = "2021-12-31"  # last day of the tuning years
STOCK_COST, ETF_COST = 10.0, 5.0
ETFS = ["SPY", "QQQ", "SSO", "UPRO", "QLD", "TQQQ", "BIL", "TLT", "IEF", "GLD", "EFA", "IWM", "RSP", "MTUM",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
# Yearly fees used to simulate LQQ3 (WisdomTree NASDAQ 100 3x Daily Leveraged, London-listed, not on
# Alpaca) from QQQ. LQQ3's 0.75% is not verified from WisdomTree's documents: check the KID.
LQQ3_FEE, TQQQ_FEE, QQQ_FEE = 0.0075, 0.0084, 0.0020
LQQ3, LQQ3_UPPER = "LQQ3 (sim)", "LQQ3 (sim, upper bound)"


# ------------------------------------------------------------------ data

def etf_bars(symbols=ETFS, start: str = "2016-01-01") -> pd.DataFrame:
    """Adjusted daily bars (splits and dividends) from Alpaca, cached in data/benchmarks/."""
    path = paths().root / "benchmarks" / "etf_daily.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_parquet(path)
        fresh = (pd.Timestamp.today().normalize() - df["date"].max()).days <= 3
        if fresh and set(symbols) <= set(df["ticker"]):
            return df
    df = alpaca.fetch_bars(alpaca.session(), list(symbols), start, alpaca.safe_end(), "all", "sip")
    df.to_parquet(path, index=False)
    return df


@dataclass
class Book:
    """Prices of the assets a strategy can hold, on the common calendar."""
    names: pd.Index
    open: np.ndarray
    close: np.ndarray
    cost_bps: float

    @classmethod
    def build(cls, o: pd.DataFrame, c: pd.DataFrame, cost_bps: float) -> "Book":
        c = c.ffill()
        o = o.where(o.notna(), c.shift(1))  # no trade that day: no overnight move
        return cls(c.columns, o.to_numpy(), c.to_numpy(), cost_bps)

    def col(self, name: str) -> int:
        return int(self.names.get_loc(name))


@dataclass
class Lab:
    m: Market
    dates: pd.DatetimeIndex
    stocks: Book
    etfs: Book
    etf_close: pd.DataFrame
    cash_r: np.ndarray
    i0: int
    x: dict = field(default_factory=dict)  # signal arrays, computed once


def leveraged(close: pd.Series, open_: pd.Series, cash_r: pd.Series, lev: float,
              cost: float) -> tuple[pd.Series, pd.Series]:
    """Open and close of a fund that resets to `lev` x the daily return of `close` at each close,
    borrowing (lev - 1) at the T-bill rate and paying `cost` a year on top (fees, funding spread)."""
    r = close.pct_change().fillna(0.0)
    c = 100 * (1 + lev * r - (lev - 1) * cash_r - cost / 252).cumprod()
    o = c.shift(1) * (1 + lev * (open_ / close.shift(1) - 1))  # overnight: lev x the index's gap
    o.iloc[0] = c.iloc[0]
    return o, c


def drag_of(real: pd.Series, close: pd.Series, open_: pd.Series, cash_r: pd.Series, lev: float) -> float:
    """The yearly cost at which leveraged(close) grows exactly like the real fund (bisection)."""
    years = (real.index[-1] - real.index[0]).days / 365.25
    target = (real.iloc[-1] / real.iloc[0]) ** (1 / years)
    lo, hi = -0.05, 0.10
    for _ in range(50):
        mid = (lo + hi) / 2
        c = leveraged(close, open_, cash_r, lev, mid)[1]
        lo, hi = (mid, hi) if (c.iloc[-1] / c.iloc[0]) ** (1 / years) > target else (lo, mid)
    return (lo + hi) / 2


def add_lqq3(eo: pd.DataFrame, ec: pd.DataFrame, cash: pd.Series) -> float:
    """Simulated LQQ3 columns from QQQ: one with TQQQ's measured all-in cost (adjusted for the lower
    fee), one with index math only (dividends kept, borrowing at the T-bill rate) as an upper bound.
    Returns TQQQ's measured yearly drag."""
    q = ec["QQQ"].notna() & ec["TQQQ"].notna()
    qc, qo = ec["QQQ"].ffill(), eo["QQQ"].fillna(ec["QQQ"].ffill())
    drag = drag_of(ec.loc[q, "TQQQ"], qc[q], qo[q], cash[q], 3)
    for name, cost in ((LQQ3, drag - (TQQQ_FEE - LQQ3_FEE)), (LQQ3_UPPER, LQQ3_FEE - 3 * QQQ_FEE)):
        eo[name], ec[name] = leveraged(qc, qo, cash, 3, cost)
    return drag


def build_lab() -> Lab:
    m = load_market()
    dates = m.close.index
    bars = etf_bars()
    eo = bars.pivot(index="date", columns="ticker", values="o").reindex(dates)
    ec = bars.pivot(index="date", columns="ticker", values="c").reindex(dates)
    cash_r = ec["BIL"].ffill().pct_change().fillna(0.0).to_numpy()
    tqqq_drag = add_lqq3(eo, ec, pd.Series(cash_r, index=dates))
    i0 = int(dates.searchsorted(pd.Timestamp(START)))
    lab = Lab(m, dates, Book.build(m.open, m.close, STOCK_COST), Book.build(eo, ec, ETF_COST), ec, cash_r, i0)
    extras = load_extras(m)
    c = m.close
    f = m.features()
    lab.x.update({
        "r12_1": f["r12_1"].to_numpy(), "r6_1": (c.shift(21) / c.shift(126) - 1).to_numpy(),
        "composite": m.score("composite").to_numpy(), "vol126": f["vol126"].to_numpy(),
        "hi52": (c / c.rolling(252, min_periods=240).max()).to_numpy(),
        "r5": (c / c.shift(5) - 1).to_numpy(), "above200": (c > c.rolling(200, min_periods=200).mean()).to_numpy(),
        "uni": m.universe.to_numpy(),
        "mcap_rank": extras["mcap"].where(m.universe).rank(axis=1, ascending=False).to_numpy(),
        "issuance": (extras["shares_adj"] / extras["shares_adj"].shift(252) - 1).to_numpy(),
        "spy_on": regime(ec["SPY"].ffill(), 200, 1).to_numpy(),
        "spy_on3": regime(ec["SPY"].ffill(), 200, 3).to_numpy(), "tqqq_drag": tqqq_drag,
    })
    return lab


def load_extras(m: Market) -> dict[str, pd.DataFrame]:
    from stockdata import load_panel
    df = load_panel(m.close.index[0].strftime("%Y-%m-%d"), tickers=list(m.close.columns),
                    columns=["mcap", "shares", "split_factor"])
    df["shares_adj"] = df["shares"] * df["split_factor"]  # today's share units, so splits do not count

    def piv(col):
        return df.pivot(index="date", columns="ticker", values=col).reindex(index=m.close.index,
                                                                            columns=m.close.columns)
    return {"mcap": piv("mcap"), "shares_adj": piv("shares_adj")}


# ------------------------------------------------------------------ engine

def run(book: Book, targets: dict[int, dict[int, float]], cash_r: np.ndarray, i0: int,
        dates: pd.DatetimeIndex) -> tuple[pd.Series, float, dict]:
    """Daily value of a strategy that moves to `targets[i]` (asset -> weight, rest in cash)
    at the open after the close of day i. Returns equity, one-way turnover per year, last target."""
    O, C = book.open, book.close
    cost = book.cost_bps / 1e4
    n_t = len(dates)
    eq = np.full(n_t, np.nan)
    value, cash_w, traded = 1.0, 1.0, 0.0
    hold: dict[int, float] = {}
    eq[i0 - 1] = 1.0
    pending = targets.get(i0 - 1)
    last = pending or {}
    for i in range(i0, n_t):
        if pending is not None:
            g = cash_w
            for a in hold:
                hold[a] *= O[i, a] / C[i - 1, a]
                g += hold[a]
            value *= g
            hold = {a: w / g for a, w in hold.items()}
            cash_w /= g
            tgt = {a: w for a, w in pending.items() if w > 0 and np.isfinite(O[i, a]) and O[i, a] > 0}
            turn = sum(abs(tgt.get(a, 0.0) - hold.get(a, 0.0)) for a in set(tgt) | set(hold))
            value *= 1 - cost * turn
            traded += turn
            hold, cash_w = dict(tgt), 1.0 - sum(tgt.values())
            ref = O[i]
        else:
            ref = C[i - 1]
        cash_w *= 1 + cash_r[i]
        g = cash_w
        for a in hold:
            hold[a] *= C[i, a] / ref[a]
            g += hold[a]
        value *= g
        hold = {a: w / g for a, w in hold.items()}
        cash_w /= g
        eq[i] = value
        pending = targets.get(i)
        if pending is not None:
            last = pending
    equity = pd.Series(eq[i0 - 1:], index=dates[i0 - 1:])
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return equity, traded / 2 / years, {book.names[a]: w for a, w in last.items()}


# ------------------------------------------------------------------ helpers

def key_days(dates: pd.DatetimeIndex, i0: int, freq: str, offset: int = 0) -> list[int]:
    """Last session of each month ("M") or week ("W"), moved `offset` sessions later,
    from the first decision day on. The final date counts only if the next weekday is in a
    new period (the data may stop mid-month)."""
    f = "M" if freq == "M" else "W-FRI"
    per = dates.to_period(f)
    final = (dates[-1] + pd.offsets.BDay(1)).to_period(f) != per[-1]
    last = np.flatnonzero(np.r_[per[1:] != per[:-1], final]) + offset
    return sorted({i0 - 1} | {int(i) for i in last if i0 - 1 <= i < len(dates)})


def flips(on: np.ndarray, i0: int) -> list[int]:
    return [i for i in range(i0, len(on)) if on[i] != on[i - 1]]


def pick(lab: Lab, score: np.ndarray, ok: np.ndarray, n: int, freq: str = "M", on: np.ndarray | None = None,
         sector_cap: int | None = None, offset: int = 0) -> dict[int, dict[int, float]]:
    """Equal-weight top-n by score among ok stocks on each key day; cash while `on` is False."""
    days = key_days(lab.dates, lab.i0, freq, offset)
    if on is not None:
        days = sorted(set(days) | set(flips(on, lab.i0)))
    codes = pd.factorize(lab.m.sector)[0]
    out = {}
    for i in days:
        out[i] = {} if on is not None and not on[i] else top_n(score[i], ok[i], n, codes, sector_cap)
    return out


def top_n(score: np.ndarray, ok: np.ndarray, n: int, codes: np.ndarray | None = None,
          sector_cap: int | None = None) -> dict[int, float]:
    """One day's equal-weight top-n by score among ok stocks (ties keep column order)."""
    idx = np.flatnonzero(ok & np.isfinite(score))
    order = idx[np.argsort(-score[idx], kind="stable")]
    if sector_cap:
        picks, cnt = [], {}
        for j in order:
            if cnt.get(codes[j], 0) < sector_cap:
                picks.append(j)
                cnt[codes[j]] = cnt.get(codes[j], 0) + 1
            if len(picks) == n:
                break
    else:
        picks = order[:n]
    return {int(j): 1.0 / n for j in picks}


def etf_trend(lab: Lab, signal: str, hold: str, n: int = 200, confirm: int = 1) -> dict:
    """Hold `hold` while `signal` is above its n-day average (confirm closes to switch), else T-bills."""
    on = regime(lab.etf_close[signal].ffill(), n, confirm).to_numpy()
    j = lab.etfs.col(hold)
    days = [lab.i0 - 1] + flips(on, lab.i0)
    return {i: ({j: 1.0} if on[i] else {}) for i in days}


def etf_hold(lab: Lab, weights: dict[str, float]) -> dict:
    """Buy and hold, rebalanced to the weights each month."""
    w = {lab.etfs.col(k): v for k, v in weights.items()}
    return {i: dict(w) for i in key_days(lab.dates, lab.i0, "M")}


def etf_rotation(lab: Lab, universe: list[str], top: int, lookback: str = "12-1", filt: bool = False) -> dict:
    c = lab.etf_close[universe]
    score = (c.shift(21) / c.shift(252) - 1) if lookback == "12-1" else (c / c.shift(126) - 1)
    on = lab.x["spy_on"] if filt else None
    days = key_days(lab.dates, lab.i0, "M")
    if on is not None:
        days = sorted(set(days) | set(flips(on, lab.i0)))
    cols = [lab.etfs.col(s) for s in universe]
    out = {}
    for i in days:
        if on is not None and not on[i]:
            out[i] = {}
            continue
        row = score.iloc[i].to_numpy()
        ok = np.flatnonzero(np.isfinite(row))
        best = ok[np.argsort(-row[ok], kind="stable")][:top]
        out[i] = {cols[k]: 1.0 / top for k in best}
    return out


def dual_momentum(lab: Lab) -> dict:
    """Antonacci's global equity momentum: SPY or EFA by 12-month return, IEF when SPY lags T-bills."""
    c = lab.etf_close
    r12 = c / c.shift(252) - 1
    out = {}
    for i in key_days(lab.dates, lab.i0, "M"):
        if r12["SPY"].iloc[i] > r12["BIL"].iloc[i]:
            best = "SPY" if r12["SPY"].iloc[i] >= r12["EFA"].iloc[i] else "EFA"
        else:
            best = "IEF"
        out[i] = {lab.etfs.col(best): 1.0}
    return out


def earnings_score(lab: Lab, window: int = 60) -> np.ndarray:
    """Abnormal return over the reaction session and the next (vs SPY), known at the close of E+1
    and kept for `window` sessions."""
    m = lab.m
    C = lab.stocks.close
    spy = lab.etf_close["SPY"].ffill().to_numpy()
    ev = m.earnings
    e = lab.dates.get_indexer(pd.DatetimeIndex(ev["session"]).as_unit(lab.dates.unit))
    j = m.close.columns.get_indexer(ev["ticker"])
    keep = (e >= 1) & (j >= 0) & (e + 1 < len(lab.dates))
    e, j = e[keep], j[keep]
    order = np.argsort(e, kind="stable")
    e, j = e[order], j[order]
    ear = (C[e + 1, j] / C[e - 1, j]) / (spy[e + 1] / spy[e - 1]) - 1
    out = np.full(C.shape, np.nan)
    for ei, ji, v in zip(e, j, ear):
        out[ei + 1: ei + 1 + window, ji] = v  # later releases overwrite earlier ones
    return out


def reports_soon(lab: Lab, lo: int = 2, hi: int = 6) -> np.ndarray:
    """True on day t when the stock's next earnings reaction session is t+lo .. t+hi, so a buy at the
    open of t+1 is in before the release and a weekly rebalance (open of t+6) sells after it."""
    ev = lab.m.earnings
    e = lab.dates.get_indexer(pd.DatetimeIndex(ev["session"]).as_unit(lab.dates.unit))
    j = lab.m.close.columns.get_indexer(ev["ticker"])
    keep = (e >= 0) & (j >= 0)
    out = np.zeros(lab.stocks.close.shape, dtype=bool)
    for k in range(lo, hi + 1):
        r = e[keep] - k
        ok = r >= 0
        out[r[ok], j[keep][ok]] = True
    return out


# ------------------------------------------------------------------ the strategies

BENCH = "SPY (buy and hold)"
YOURS = "Your rules (golden cross + 3-day SuperTrend, top 12)"
LABEL = {"r12_1": "12-1 month", "r6_1": "6-1 month", "composite": "3/6/12 month"}


@dataclass
class Strat:
    name: str
    family: str
    book: str  # stocks | etfs
    targets: dict | None = None
    kind: str = "grid"  # standard (published default) | grid | reference
    cash: str = "BIL"  # what idle cash earns: BIL (T-bills) or SPY (parked in the index, no costs)
    equity: pd.Series | None = None
    turnover: float = np.nan
    last: dict = field(default_factory=dict)


def momentum_name(score: str, n: int, size: int | None, filt: bool = False, cap: int | None = None) -> str:
    pool = "$3bn+" if size is None else f"largest {size}"
    return (f"Momentum {LABEL[score]}, top {n} of {pool}" + (f", max {cap} per sector" if cap else "") +
            (", SPY filter" if filt else ""))


def momentum_pool(lab: Lab, size: int | None) -> np.ndarray:
    """Stocks a momentum strategy may hold each day: the universe, or its `size` largest companies."""
    x = lab.x
    return x["uni"] if size is None else x["uni"] & (x["mcap_rank"] <= size)


def momentum(lab: Lab, score: str, n: int, size: int | None, filt: bool = False, cap: int | None = None,
             offset: int = 0) -> dict:
    """Each month, the n stocks with the best momentum score, equal weight."""
    x = lab.x
    return pick(lab, x[score], momentum_pool(lab, size), n, "M", x["spy_on"] if filt else None, sector_cap=cap,
                offset=offset)


def strategies(lab: Lab) -> list[Strat]:
    x = lab.x
    uni, mrank = x["uni"], x["mcap_rank"]
    S: list[Strat] = []

    def add(name, family, book, targets, kind="grid", cash="BIL"):
        S.append(Strat(name, family, book, targets, kind, cash))

    add(BENCH, "Benchmark", "etfs", etf_hold(lab, {"SPY": 1.0}), "reference")
    add("QQQ (buy and hold)", "Index funds", "etfs", etf_hold(lab, {"QQQ": 1.0}), "reference")
    add("RSP (equal-weight S&P 500)", "Index funds", "etfs", etf_hold(lab, {"RSP": 1.0}), "reference")
    add("MTUM (momentum ETF)", "Index funds", "etfs", etf_hold(lab, {"MTUM": 1.0}), "reference")
    # Same data, no strategy: how much do the missing (delisted) stocks flatter stock results? Compare with RSP.
    for n in (500, 100):
        add(f"Our largest {n}, equal weight", "Survivorship check", "stocks",
            pick(lab, -mrank, uni & (mrank <= n), n, "M"), "reference")
    for sym, lev in (("SSO", 2), ("UPRO", 3), ("QLD", 2), ("TQQQ", 3)):
        add(f"{sym} buy and hold ({lev}x)", "Leveraged buy and hold", "etfs", etf_hold(lab, {sym: 1.0}),
            "standard" if sym == "SSO" else "grid")

    # Trend filter on the index, plain and leveraged (Faber 2007; Gayed and Bilello 2016)
    for sig, holds in (("SPY", ("SPY", "SSO", "UPRO")), ("QQQ", ("QQQ", "QLD", "TQQQ"))):
        for h, lev in zip(holds, (1, 2, 3)):
            for n in (100, 150, 200, 250):
                for conf in (1, 3):
                    add(f"{h} when {sig} > {n}-day avg" + ("" if conf == 1 else f", {conf}-day confirm"),
                        f"Index trend {lev}x ({sig})", "etfs", etf_trend(lab, sig, h, n, conf),
                        "standard" if (n, conf) == (200, 1) else "grid")

    # LQQ3 (London-listed 3x Nasdaq-100), simulated from QQQ: Alpaca has no London prices
    for n in (100, 150, 200, 250):
        for conf in (1, 3):
            add(f"{LQQ3} when QQQ > {n}-day avg" + ("" if conf == 1 else f", {conf}-day confirm"),
                "Index trend 3x (QQQ), LQQ3", "etfs", etf_trend(lab, "QQQ", LQQ3, n, conf),
                "standard" if (n, conf) == (200, 1) else "grid")
    add(f"{LQQ3_UPPER} when QQQ > 200-day avg", "Index trend 3x (QQQ), LQQ3", "etfs",
        etf_trend(lab, "QQQ", LQQ3_UPPER, 200, 1))
    add(f"{LQQ3} buy and hold", "Leveraged buy and hold", "etfs", etf_hold(lab, {LQQ3: 1.0}))

    # Stock momentum (Jegadeesh and Titman 1993), monthly, equal weight
    for score in ("r12_1", "r6_1", "composite"):
        for size in (None, 500, 200, 100):
            for n in (10, 20, 30, 50):
                for filt in (False, True):
                    std = score == "r12_1" and n == 20 and not filt and size is None
                    add(momentum_name(score, n, size, filt), "Stock momentum", "stocks",
                        momentum(lab, score, n, size, filt), "standard" if std else "grid")
    for score in ("r6_1", "r12_1"):
        for n, cap in ((10, 3), (20, 4)):
            add(momentum_name(score, n, 500, cap=cap), "Stock momentum", "stocks", momentum(lab, score, n, 500, cap=cap))

    # Near the 52-week high (George and Hwang 2004)
    for n in (20, 50):
        add(f"Closest to 52-week high, top {n} of $3bn+", "52-week high", "stocks",
            pick(lab, x["hi52"] + 1e-6 * np.nan_to_num(x["r12_1"]), uni, n, "M"), "standard" if n == 20 else "grid")

    # Earnings announcement drift (Chan, Jegadeesh and Lakonishok 1996; Brandt et al. 2008)
    for window in (40, 60):
        ear = earnings_score(lab, window)
        for n in (10, 20, 30):
            for mom in (False, True):
                ok = uni & (ear > 0) & ((x["r12_1"] > 0) if mom else True)
                add(f"Earnings drift, top {n} reactions in last {window} sessions" + (", momentum > 0" if mom else ""),
                    "Earnings drift", "stocks", pick(lab, ear, ok, n, "W"),
                    "standard" if (window, n, mom) == (60, 20, False) else "grid")

    # Share buybacks (Pontiff and Woodgate 2008): largest 12-month fall in share count
    for n in (20, 50):
        for mom in (False, True):
            ok = uni & ((x["r12_1"] > 0) if mom else True)
            add(f"Buybacks, top {n} share-count reducers" + (", momentum > 0" if mom else ""), "Buybacks", "stocks",
                pick(lab, -x["issuance"], ok, n, "M"), "standard" if (n, mom) == (20, False) else "grid")

    # Short-term reversal in uptrends (Jegadeesh 1990; Lehmann 1990), weekly
    for size in (500, None):
        ok = uni & x["above200"] & (True if size is None else (mrank <= size))
        add(f"Weekly losers in uptrends, top 20 of {'$3bn+' if size is None else f'largest {size}'}",
            "Short-term reversal", "stocks", pick(lab, -x["r5"], ok, 20, "W"), "standard" if size == 500 else "grid")

    # Biggest companies (the period's winners: hindsight-prone)
    for n in (5, 10, 20):
        add(f"Largest {n} companies, equal weight", "Mega-caps", "stocks",
            pick(lab, -mrank, uni & (mrank <= n), n, "M"), "standard" if n == 10 else "grid")

    # Sector rotation (Moskowitz and Grinblatt 1999) and dual momentum (Antonacci 2014)
    for top in (2, 3, 4):
        for lb in ("12-1", "6m"):
            for filt in (False, True):
                add(f"Sector ETFs, top {top} by {lb}" + (", SPY filter" if filt else ""), "Sector rotation", "etfs",
                    etf_rotation(lab, SECTOR_ETFS, top, lb, filt),
                    "standard" if (top, lb, filt) == (3, "12-1", False) else "grid")
    add("Dual momentum (SPY / EFA / IEF)", "Dual momentum", "etfs", dual_momentum(lab), "standard")

    # Earnings-season momentum (idea from this data: see RESULTS.md, so a weaker out-of-sample check).
    # Each week, the highest-momentum stocks that report in the coming week; idle cash in T-bills or SPY.
    soon = reports_soon(lab)
    for score in ("r12_1", "r6_1"):
        for size in (500, None):
            for n in (10, 20):
                for cash in ("BIL", "SPY"):
                    ok = uni & soon & (True if size is None else (mrank <= size))
                    pool = "$3bn+" if size is None else f"largest {size}"
                    add(f"Reporting next week, top {n} by {LABEL[score]} of {pool}, cash in {cash}",
                        "Earnings-season momentum", "stocks", pick(lab, x[score], ok, n, "W"),
                        "standard" if (score, size, n, cash) == ("r12_1", 500, 20, "BIL") else "grid", cash)
    return S


# ------------------------------------------------------------------ evaluation

def risk_matched(spy: pd.Series, cash: pd.Series, vol: float) -> pd.Series:
    """SPY scaled to `vol` (borrowing or lending at the T-bill rate), rebalanced daily."""
    r = spy.pct_change().dropna()
    lev = vol / (r.std() * np.sqrt(252))
    m = lev * r - (lev - 1) * cash.reindex(r.index).fillna(0.0)
    return pd.concat([spy.iloc[:1] * 0 + 1.0, (1 + m).cumprod()])


def row(s: Strat, spy: pd.Series, cash: pd.Series) -> dict:
    eq = s.equity
    split = pd.Timestamp(SPLIT)
    a, b = eq[eq.index <= split], eq[eq.index >= split]
    sa, sb = spy[spy.index <= split], spy[spy.index >= split]
    st = stats(eq)
    r, rs = eq.pct_change().dropna(), spy.pct_change().dropna()
    beta = r.cov(rs) / rs.var()
    yret = yearly_returns(eq)
    ysret = yearly_returns(spy)
    return {"Strategy": s.name, "Family": s.family, "Kind": s.kind, "CAGR": st["CAGR"],
            "vs SPY": st["CAGR"] - stats(spy)["CAGR"],
            "vs risk-matched SPY": st["CAGR"] - stats(risk_matched(spy, cash, st["Volatility"]))["CAGR"],
            "Volatility": st["Volatility"], "Sharpe": st["Sharpe (rf=0)"], "Max drawdown": st["Max drawdown"],
            "2017-21": stats(a)["CAGR"], "2017-21 vs SPY": stats(a)["CAGR"] - stats(sa)["CAGR"],
            "2022-26": stats(b)["CAGR"], "2022-26 vs SPY": stats(b)["CAGR"] - stats(sb)["CAGR"],
            "Years beating SPY": int((yret > ysret.reindex(yret.index)).sum()), "Beta": beta,
            "Alpha": (r.mean() - beta * rs.mean()) * 252, "Turnover": s.turnover}


def yearly_returns(eq: pd.Series) -> pd.Series:
    y = eq.groupby(eq.index.year).last()
    return y / y.shift(1).fillna(eq.iloc[0]) - 1


def evaluate(lab: Lab, S: list[Strat]) -> pd.DataFrame:
    spy_r = lab.etf_close["SPY"].ffill().pct_change().fillna(0.0).to_numpy()
    for s in S:
        book = lab.stocks if s.book == "stocks" else lab.etfs
        cash_r = spy_r if s.cash == "SPY" else lab.cash_r
        s.equity, s.turnover, s.last = run(book, s.targets, cash_r, lab.i0, lab.dates)
        s.targets = None  # free memory
    spy = next(s.equity for s in S if s.name == BENCH)
    cash = pd.Series(lab.cash_r, index=lab.dates)
    return pd.DataFrame([row(s, spy, cash) for s in S]).set_index("Strategy")


def shortlist(tbl: pd.DataFrame) -> list[str]:
    """Per family: the published default setting and the best setting on the tuning years only."""
    names: list[str] = []
    for fam, g in tbl[tbl["Kind"] != "reference"].groupby("Family", sort=False):
        for n in g.index[g["Kind"] == "standard"].tolist() + [g["2017-21 vs SPY"].idxmax()]:
            if n not in names:
                names.append(n)
    return names


def blend(curves: list[pd.Series], weights: list[float], dates: pd.DatetimeIndex, cost_bps: float = 10.0) -> pd.Series:
    """Fixed-weight mix of strategies, rebalanced to the weights at each month end."""
    r = pd.concat([c.pct_change() for c in curves], axis=1).fillna(0.0).to_numpy()
    idx = curves[0].index
    month_end = np.r_[idx.to_period("M")[1:] != idx.to_period("M")[:-1], True]
    w = np.array(weights, dtype=float)
    v = w.copy()
    out = np.empty(len(idx))
    out[0] = 1.0
    for i in range(1, len(idx)):
        v = v * (1 + r[i])
        total = v.sum()
        if month_end[i]:
            turn = np.abs(w * total - v).sum() / total
            total *= 1 - cost_bps / 1e4 * turn
            v = w * total
        out[i] = total
    return pd.Series(out, index=idx)


def timing_spread(lab: Lab, spy: pd.Series, configs: list[tuple]) -> pd.DataFrame:
    """Excess return over SPY when the monthly rebalance happens 0, 2, ..., 20 sessions after month end."""
    rows = {}
    base = stats(spy)["CAGR"]
    for score, n, size in configs:
        ex = []
        for off in range(0, 21, 2):
            eq, _, _ = run(lab.stocks, momentum(lab, score, n, size, offset=off), lab.cash_r, lab.i0, lab.dates)
            ex.append(stats(eq)["CAGR"] - base)
        rows[momentum_name(score, n, size)] = {"Worst": min(ex), "Median": float(np.median(ex)), "Best": max(ex)}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------ report

PCT = ["CAGR", "vs SPY", "vs risk-matched SPY", "Volatility", "Max drawdown", "2017-21", "2017-21 vs SPY",
       "2022-26", "2022-26 vs SPY", "Alpha", "Worst", "Median", "Best"]


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    for c in df.columns:
        if c in PCT:
            out[c] = [f"{v:+.1%}" if c not in ("CAGR", "Volatility", "2017-21", "2022-26") else f"{v:.1%}"
                      for v in df[c]]
        elif c in ("Sharpe", "Beta", "Turnover"):
            out[c] = ["" if pd.isna(v) else f"{v:.2f}" if c != "Turnover" else f"{v:.1f}" for v in df[c]]
    return out


HEAD = ["CAGR", "vs SPY", "vs risk-matched SPY", "Sharpe", "Max drawdown", "2017-21 vs SPY", "2022-26 vs SPY",
        "Years beating SPY", "Beta", "Turnover"]


YEARLY = ["QLD when QQQ > 200-day avg", "TQQQ when QQQ > 200-day avg", "Momentum 6-1 month, top 10 of largest 500",
          "Momentum 6-1 month, top 20 of largest 500"]


def report(lab: Lab, S: list[Strat], tbl: pd.DataFrame, blends: pd.DataFrame, spread: pd.DataFrame,
           corr: float, blend_curves: dict[str, pd.Series]) -> str:
    by = {s.name: s for s in S}
    short = shortlist(tbl)
    both = [n for n in short if tbl.loc[n, "2017-21 vs SPY"] > 0 and tbl.loc[n, "2022-26 vs SPY"] > 0
            and tbl.loc[n, "vs risk-matched SPY"] > 0]
    refs = [BENCH, "QQQ (buy and hold)", YOURS, "RSP (equal-weight S&P 500)", "MTUM (momentum ETF)"]
    g = tbl[tbl["Kind"] != "reference"].groupby("Family", sort=False)
    fam = pd.DataFrame({"Settings tested": g.size(), "Median vs SPY": g["vs SPY"].median(),
                        "Worst": g["vs SPY"].min(), "Best": g["vs SPY"].max(),
                        "Beat SPY in 2022-26": g["2022-26 vs SPY"].apply(lambda s: f"{(s > 0).mean():.0%}")})
    fam = fam.sort_values("Median vs SPY", ascending=False)
    fam_f = fam.copy().astype(object)
    for c in ("Median vs SPY", "Worst", "Best"):
        fam_f[c] = [f"{v:+.1%}" for v in fam[c]]
    spy_eq = by[BENCH].equity
    curves = {n: by[n].equity for n in refs[:3] + YEARLY if n in by}
    best_mix = blends["vs SPY"].idxmax()
    curves[best_mix] = blend_curves[best_mix]
    yr = pd.DataFrame({n: yearly_returns(c) for n, c in curves.items()}).T.map(lambda v: f"{v:+.0%}")
    surv = tbl.loc[["Our largest 500, equal weight", "RSP (equal-weight S&P 500)"], ["CAGR", "2017-21", "2022-26"]]
    gap = surv.iloc[0] - surv.iloc[1]
    holdings = []
    for n in both:
        last = by[n].last
        held = ", ".join(sorted(last)) if last else "cash (T-bills)"
        holdings.append(f"- {n}: {held}")
    n_tested = int((tbl["Kind"] != "reference").sum())
    lines = [
        f"# Strategy search ({spy_eq.index[0].date()} to {spy_eq.index[-1].date()})", "",
        f"Generated by `python -m backtest.strategies` on {dt.date.today()}. {n_tested} strategy settings in "
        f"{fam.shape[0]} families, plus reference funds. Rules and code: backtest/strategies.py.", "",
        "How it was run: decisions at the close, trades at the next open, costs of 10 bps per side for stocks and "
        "5 bps for ETFs, idle cash earns T-bills (BIL). Each idea was first run with its published default "
        "setting; other settings are a grid. 2017-21 are the tuning years and 2022-26 the out-of-sample check: "
        "the shortlist below is each family's default setting plus its best setting on 2017-21 only. "
        "'vs risk-matched SPY' compares with SPY scaled (borrowing or lending at the T-bill rate) to the same "
        "volatility, so it shows what the strategy adds beyond simply taking more market risk.", "",
        "## Shortlist", "", md(fmt(tbl.loc[refs[:3] + short, HEAD]).rename_axis("Strategy"), "Strategy"), "",
        "Beat SPY in both the tuning years and the out-of-sample years, and beat SPY scaled to the same risk "
        "over the whole period: " + "; ".join(both) + ".", "",
        "## Every family", "", md(fam_f, "Family"), "",
        "## Robustness", "",
        "Momentum, by the day of the month the portfolio is rebalanced (0 = last session of the month, 20 = "
        "about three weeks later); excess return over SPY per year:", "", md(fmt(spread), "Strategy"), "",
        f"Survivorship: the data holds only stocks listed today. Our largest 500 at equal weight returned "
        f"{surv.iloc[0]['CAGR']:.1%} a year against {surv.iloc[1]['CAGR']:.1%} for RSP, the real equal-weight "
        f"S&P 500 fund ({gap['2017-21']:+.1%} in 2017-21, {gap['2022-26']:+.1%} in 2022-26). Treat stock results as "
        "flattered by at least that much, more for concentrated portfolios. ETF results have no such bias.", "",
        "## LQQ3 instead of QLD (simulated)", "",
        "LQQ3 (WisdomTree NASDAQ 100 3x Daily Leveraged) trades in London, and Alpaca has no London prices, so it "
        "is simulated from QQQ: 3x the daily return, (3 - 1) x the T-bill rate on the borrowed part, and a yearly "
        f"cost. '{LQQ3}' uses the all-in cost measured on TQQQ, the US 3x fund ({lab.x['tqqq_drag']:.2%} a year "
        f"beyond T-bill borrowing), less the fee difference ({TQQQ_FEE:.2%} vs an assumed {LQQ3_FEE:.2%} for LQQ3); "
        f"'{LQQ3_UPPER}' uses index math only (dividends kept, borrowing at the T-bill rate). Real LQQ3 should "
        "land between the two, plus currency moves (it is unhedged; a sterling S&P 500 fund moves the same way) "
        "and London trading hours (the signal comes at the US close, trades fill in the next London session).", "",
        md(fmt(tbl.loc[[n for n in ("QLD when QQQ > 200-day avg", "TQQQ when QQQ > 200-day avg",
                                    f"{LQQ3} when QQQ > 200-day avg", f"{LQQ3_UPPER} when QQQ > 200-day avg",
                                    f"{LQQ3} buy and hold", BENCH) if n in tbl.index], HEAD]), "Strategy"), "",
        "## Combinations (50/50, rebalanced monthly)", "",
        md(fmt(blends[[c for c in HEAD if c in blends.columns]]), "Mix"), "",
        f"Daily correlation between large-cap momentum (top 20) and the 2x Nasdaq trend strategy: {corr:.2f}.", "",
        "## Calendar-year returns", "", md(yr, "Strategy"), "",
        "## What the qualifying strategies hold now (signal at the last close)", "", *holdings, "",
        "## Caveats", "",
        "- Ten years is short and mostly a bull market led by large US tech. The Nasdaq-100 itself beat SPY by "
        "about 5% a year, which lifts every tech-heavy strategy. Momentum crashes (2009) and a dot-com style "
        "bear market are not in the sample.",
        "- Leveraged ETFs reset daily. A fast crash hits them before a 200-day filter can react: when the "
        "Nasdaq-100 fell 30% before crossing its 200-day average in 2000, a 3x fund would have lost about "
        "two-thirds.",
        "- Stock strategies use today's listed stocks only (see Survivorship). No market-impact model; taxes "
        "ignored (monthly turnover creates short-term gains).",
        "- Many settings were tried; the best of many backtests is always flattered by luck. Prefer families that "
        "work across most of their grid over the single best row.", "",
        "<details><summary>All settings tested</summary>", "",
        md(fmt(tbl[HEAD + ["Family"]]), "Strategy"), "", "</details>",
    ]
    return "\n".join(lines)


BLENDS = [
    ("Momentum 6-1 month, top 10 of largest 500", "QLD when QQQ > 200-day avg"),
    ("Momentum 6-1 month, top 20 of largest 500", "QLD when QQQ > 200-day avg"),
    ("Momentum 6-1 month, top 20 of largest 500", "SSO when SPY > 200-day avg"),
    ("Momentum 6-1 month, top 20 of largest 500", BENCH),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    t0 = time.time()
    lab = build_lab()
    log.info("lab ready (%.0fs)", time.time() - t0)
    S = strategies(lab)
    log.info("%d strategies defined (%.0fs)", len(S), time.time() - t0)
    tbl = evaluate(lab, S)
    # Your current rules (backtest/momentum_trend.py), same start date; idle cash there earns nothing
    yours = simulate(lab.m, Params()).equity
    yours = yours[yours.index >= lab.dates[lab.i0 - 1]]
    s_y = Strat(YOURS, "Your rules", "stocks", kind="reference", equity=yours / yours.iloc[0])
    S.append(s_y)
    tbl = pd.concat([tbl, pd.DataFrame([row(s_y, next(s.equity for s in S if s.name == BENCH),
                                            pd.Series(lab.cash_r, index=lab.dates))]).set_index("Strategy")])
    log.info("evaluated (%.0fs)", time.time() - t0)
    by = {s.name: s for s in S}
    spy = by[BENCH].equity
    cash = pd.Series(lab.cash_r, index=lab.dates)
    rows, blend_curves = [], {}
    for a, b in BLENDS:
        eq = blend([by[a].equity, by[b].equity], [0.5, 0.5], lab.dates)
        blend_curves[f"50% {a} + 50% {b}"] = eq
        rows.append(row(Strat(f"50% {a} + 50% {b}", "Blend", "mix", kind="blend", equity=eq,
                              turnover=np.nan), spy, cash))
    blends = pd.DataFrame(rows).set_index("Strategy")
    corr = by[BLENDS[1][0]].equity.pct_change().corr(by[BLENDS[1][1]].equity.pct_change())
    spread = timing_spread(lab, spy, [("r6_1", 10, 500), ("r6_1", 20, 500), ("r12_1", 20, 500), ("r12_1", 20, None)])
    log.info("robustness done (%.0fs)", time.time() - t0)
    out = paths().root / "backtests" / "strategies"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([tbl, blends]).to_csv(out / "summary.csv")
    pd.concat({s.name: s.equity for s in S}, axis=1).to_csv(out / "equity_curves.csv")
    text = report(lab, S, tbl, blends, spread, corr, blend_curves)
    (REPO_ROOT / "backtest" / "STRATEGIES.md").write_text(text + "\n")
    log.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()

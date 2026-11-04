"""Momentum + trend strategy on the stockdata panel.

Rules (signals at the close of day t, orders filled at the open of day t+1):
  Buy candidates: in the $3bn+ universe (stockdata `in_universe`), golden cross
    (SMA50 > SMA200), and SuperTrend(10, 3) bullish on each of the last 3 days.
  Ranking: composite momentum = mean of cross-sectional percentile ranks of the 3-month,
    6-month and 12-1-month returns (12 months excluding the latest month), computed over
    the universe that day.
  Portfolio: up to 12 stocks, at most 4 per sector. Free slots are filled with the best
    ranked candidates, each funded with cash / free slots (equal weight at entry);
    positions then drift with the market.
  Sell: SuperTrend bearish on each of the last 3 days.
  Costs: 10 bps per side on traded value. Idle cash earns nothing.

Optional rules (Params fields), run as variants:
  market_filter: when SPY closes below its 200-day average 3 days in a row, sell everything
    at the next open and stay in cash; buying resumes after 3 closes in a row above it.
  max_vol_pct: skip candidates whose 6-month volatility is above this percentile of the
    universe that day (0.8 = never buy the most volatile 20%). Holdings are not sold for it.
  momentum="vol_adjusted": rank on each return divided by 6-month volatility.
  earnings_days=N: with E the first session that can react to an earnings release (SEC
    8-K Item 2.02, stockdata/earnings.py), hold nothing from the open of E-N to the close
    of E+N: a holding is sold at the open of E-N, and the stock cannot be bought again
    before the open of E+N+1. Uses the actual release dates, so it assumes they are known
    N+1 sessions ahead (companies usually announce the date weeks before).

Run: python -m backtest.momentum_trend
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockdata import alpaca, load_earnings, load_panel
from stockdata.config import REPO_ROOT, paths
from stockdata.earnings import reaction_session

from .indicators import confirmed, event_window, momentum_features, regime, sma, supertrend_bull

log = logging.getLogger(__name__)
NICE = np.array([2, 3, 4, 5, 8, 10, 20, 25, 50, 100, 1.5, 2.5], dtype=float)
NICE = np.concatenate([NICE, 1 / NICE])


@dataclass(frozen=True)
class Params:
    name: str = "Your rules (3-day SuperTrend)"
    n_positions: int = 12
    sector_cap: int = 4
    st_period: int = 10
    st_mult: float = 3.0
    confirm_days: int = 3
    fast: int = 50
    slow: int = 200
    cost_bps: float = 10.0
    momentum: str = "composite"  # composite | vol_adjusted
    death_cross_exit: bool = False
    market_filter: bool = False  # SPY vs its market_sma-day average, confirm_days closes to switch
    market_sma: int = 200
    max_vol_pct: float | None = None  # skip candidates above this universe percentile of 6-month vol
    earnings_days: int = 0  # hold nothing within this many sessions of an earnings release
    start: str = "2017-01-03"
    capital: float = 100_000.0


@dataclass
class Market:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    universe: pd.DataFrame
    sector: pd.Series
    benchmark: pd.Series | None = None  # SPY close on the same dates (market filter)
    earnings: pd.DataFrame | None = None  # ticker, session: first session reacting to each release
    jumps_fixed: list = field(default_factory=list)
    _cache: dict = field(default_factory=dict)

    def st(self, period: int, mult: float) -> pd.DataFrame:
        key = ("st", period, mult)
        if key not in self._cache:
            self._cache[key] = supertrend_bull(self.high, self.low, self.close, period, mult)
        return self._cache[key]

    def sma(self, n: int) -> pd.DataFrame:
        key = ("sma", n)
        if key not in self._cache:
            self._cache[key] = sma(self.close, n)
        return self._cache[key]

    def features(self) -> dict[str, pd.DataFrame]:
        if "feats" not in self._cache:
            self._cache["feats"] = momentum_features(self.close)
        return self._cache["feats"]

    def score(self, kind: str) -> pd.DataFrame:
        key = ("score", kind)
        if key not in self._cache:
            f = self.features()
            if kind == "composite":
                parts = [f["r3"], f["r6"], f["r12_1"]]
            elif kind == "vol_adjusted":
                parts = [f["r3"] / f["vol126"], f["r6"] / f["vol126"], f["r12_1"] / f["vol126"]]
            else:
                raise ValueError(kind)
            ranks = [x.where(self.universe).rank(axis=1, pct=True) for x in parts]
            s = sum(ranks) / len(ranks)
            self._cache[key] = s.where(self.universe)
        return self._cache[key]

    def vol_rank(self) -> pd.DataFrame:
        """Percentile of 6-month volatility within the universe that day (1.0 = most volatile)."""
        if "vol_rank" not in self._cache:
            self._cache["vol_rank"] = self.features()["vol126"].where(self.universe).rank(axis=1, pct=True)
        return self._cache["vol_rank"]

    def risk_on(self, n: int, days: int) -> pd.Series:
        key = ("risk_on", n, days)
        if key not in self._cache:
            if self.benchmark is None:
                raise ValueError("the market filter needs Market.benchmark (SPY closes)")
            self._cache[key] = regime(self.benchmark.reindex(self.close.index).ffill(), n, days)
        return self._cache[key]

    def blackout(self, days: int) -> pd.DataFrame:
        key = ("blackout", days)
        if key not in self._cache:
            if self.earnings is None:
                raise ValueError("the earnings blackout needs Market.earnings (stockdata.load_earnings)")
            self._cache[key] = event_window(self.earnings, self.close.index, self.close.columns, days)
        return self._cache[key]


def load_market(start: str = "2016-01-01", benchmark: bool = True, earnings: bool = True) -> Market:
    """Wide price frames for every ticker that is ever in the universe, plus SPY and earnings dates."""
    df = load_panel(start, columns=["open", "high", "low", "close", "in_universe", "sector"])
    ever = df.loc[df["in_universe"], "ticker"].unique()
    df = df[df["ticker"].isin(ever)]

    def piv(col):
        return df.pivot(index="date", columns="ticker", values=col).sort_index()

    o, h, l, c = (piv(x).astype("float64") for x in ("open", "high", "low", "close"))
    uni = piv("in_universe").fillna(False).astype(bool)
    sector = df.groupby("ticker", observed=True)["sector"].last().astype(object)
    sector = sector.reindex(c.columns).fillna("Unknown")
    m = Market(o, h, l, c, uni, sector)
    # Corporate actions Alpaca did not adjust (stockdata flags them): scale earlier prices so a
    # position held across that day does not book a fake jump.
    jp = paths().reports / "price_jumps.csv"
    if jp.exists():
        for r in pd.read_csv(jp, parse_dates=["date"]).itertuples(index=False):
            if r.ticker not in c.columns:
                continue
            R = float(NICE[np.argmin(np.abs(np.log(NICE) - np.log(r.price_ratio)))])
            mask = c.index < r.date
            for frame in (m.open, m.high, m.low, m.close):
                frame.loc[mask, r.ticker] *= R
            m.jumps_fixed.append((r.ticker, str(r.date.date()), R))
    if benchmark:
        m.benchmark = spy().reindex(c.index).ffill()
    if earnings:
        try:
            ev = load_earnings(c.columns)
        except FileNotFoundError as e:
            log.warning("%s (earnings variants will fail)", e)
        else:
            sess, _ = reaction_session(ev["accepted"], c.index)  # on this calendar
            m.earnings = pd.DataFrame({"ticker": ev["ticker"].to_numpy(), "session": sess.to_numpy()})
    return m


TRADE_COLUMNS = ["ticker", "sector", "entry", "exit", "entry_px", "exit_px", "return", "days", "reason"]


@dataclass
class Result:
    params: Params
    equity: pd.Series
    invested: pd.Series
    trades: pd.DataFrame
    holdings: pd.DataFrame
    turnover: float


def simulate(m: Market, p: Params) -> Result:
    bull = m.st(p.st_period, p.st_mult)
    fast, slow = m.sma(p.fast), m.sma(p.slow)
    golden = (fast > slow).fillna(False)
    entry_ok = confirmed(bull, 1.0, p.confirm_days)
    exit_sig = confirmed(bull, 0.0, p.confirm_days)
    if p.death_cross_exit:
        exit_sig = exit_sig | (fast < slow).fillna(False)
    score = m.score(p.momentum)
    eligible = m.universe & golden & entry_ok & score.notna()
    if p.max_vol_pct is not None:
        eligible = eligible & (m.vol_rank() <= p.max_vol_pct)

    dates = m.close.index
    tick = m.close.columns.to_numpy()
    O = m.open.to_numpy()
    C = m.close.ffill().to_numpy()
    E, X, S = eligible.to_numpy(), exit_sig.to_numpy(), score.to_numpy()
    n_t, n_s = C.shape
    on = m.risk_on(p.market_sma, p.confirm_days).to_numpy() if p.market_filter else np.ones(n_t, bool)
    B = m.blackout(p.earnings_days).to_numpy() if p.earnings_days else None
    sec_codes, sec_names = pd.factorize(m.sector)
    i0 = int(dates.searchsorted(pd.Timestamp(p.start)))
    cost = p.cost_bps / 1e4

    cash = p.capital
    shares = np.zeros(n_s)
    held: set[int] = set()
    entry: dict[int, tuple] = {}
    pend_sell: dict[int, str] = {}  # ticker column -> exit reason
    pend_buy: list[int] = []
    eq = np.full(n_t, np.nan)
    inv = np.full(n_t, np.nan)
    traded = 0.0
    trades = []
    snap = []

    for i in range(i0, n_t):
        # 1) orders decided at yesterday's close fill at today's open
        if pend_sell:
            retry = {}
            for j, why in pend_sell.items():
                px = O[i, j]
                if not np.isfinite(px):
                    retry[j] = why
                    continue
                value = shares[j] * px
                cash += value * (1 - cost)
                traded += value
                ei, epx, _ = entry[j]
                trades.append({"ticker": tick[j], "sector": sec_names[sec_codes[j]], "entry": dates[ei],
                               "exit": dates[i], "entry_px": epx, "exit_px": px,
                               "return": px / epx - 1, "days": i - ei, "reason": why})
                shares[j] = 0.0
                held.discard(j)
            pend_sell = retry
        if pend_buy:
            n_free = p.n_positions - len(held)
            if n_free > 0:
                alloc = cash / n_free
                for j in pend_buy[:n_free]:
                    px = O[i, j]
                    if not np.isfinite(px) or alloc <= 0:
                        continue
                    shares[j] = alloc * (1 - cost) / px
                    cash -= alloc
                    traded += alloc
                    held.add(j)
                    entry[j] = (i, px, alloc)
            pend_buy = []
        # 2) mark to market at the close
        pos = float(sum(shares[j] * C[i, j] for j in held))
        eq[i] = cash + pos
        inv[i] = pos / eq[i] if eq[i] > 0 else 0.0
        if dates[i].is_month_end or i == n_t - 1:
            snap.append({"date": dates[i], "holdings": ",".join(sorted(tick[j] for j in held))})
        # 3) signals at today's close -> orders for tomorrow's open
        if i + 1 >= n_t:
            break
        for j in held:
            if j in pend_sell:
                continue
            if X[i, j]:
                pend_sell[j] = "trend"
            elif B is not None and B[i + 1, j]:
                pend_sell[j] = "earnings"
            elif not on[i]:
                pend_sell[j] = "market"
        remaining = [j for j in held if j not in pend_sell]
        n_free = p.n_positions - len(remaining)
        if n_free > 0 and on[i]:
            ok = E[i] if B is None else E[i] & ~B[i + 1]
            cand = np.flatnonzero(ok)
            cand = cand[~np.isin(cand, list(held))]
            if cand.size:
                order = cand[np.argsort(-S[i, cand], kind="stable")]
                counts = Counter(sec_codes[j] for j in remaining)
                picks = []
                for j in order:
                    if counts[sec_codes[j]] >= p.sector_cap:
                        continue
                    picks.append(int(j))
                    counts[sec_codes[j]] += 1
                    if len(picks) == n_free:
                        break
                pend_buy = picks

    for j in held:  # positions still open at the end
        ei, epx, _ = entry[j]
        trades.append({"ticker": tick[j], "sector": sec_names[sec_codes[j]], "entry": dates[ei], "exit": pd.NaT,
                       "entry_px": epx, "exit_px": C[-1, j], "return": C[-1, j] / epx - 1, "days": n_t - 1 - ei,
                       "reason": "open"})
    equity = pd.Series(eq[i0:], index=dates[i0:], name=p.name).dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    turnover = traded / 2 / equity.mean() / years  # one-way, per year
    return Result(p, equity, pd.Series(inv[i0:], index=dates[i0:]), pd.DataFrame(trades, columns=TRADE_COLUMNS),
                  pd.DataFrame(snap), turnover)


# ------------------------------------------------------------------ benchmarks

def spy(start: str = "2016-01-01") -> pd.Series:
    """SPY adjusted close from Alpaca (same source), cached in data/benchmarks/."""
    path = paths().root / "benchmarks" / "SPY.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        s = pd.read_parquet(path)["close"]
        if (pd.Timestamp.today().normalize() - s.index.max()).days <= 3:
            return s
    sess = alpaca.session()
    bars = alpaca.fetch_bars(sess, ["SPY"], start, alpaca.safe_end(), "all", "sip")
    s = bars.set_index("date")["c"].rename("close").astype(float)
    s.to_frame().to_parquet(path)
    return s


def equal_weight_universe(m: Market, start: str) -> pd.Series:
    """Every universe stock at equal weight, membership reset each month (no costs)."""
    r = m.close.pct_change(fill_method=None)
    member = m.universe.copy()
    month = member.index.to_period("M")
    first = ~month.duplicated()
    fixed = member.where(pd.Series(first, index=member.index), np.nan).ffill().shift(1)  # set at month start
    fixed = fixed.fillna(False).astype(bool)
    daily = r.where(fixed).mean(axis=1).fillna(0.0)
    daily = daily[daily.index >= pd.Timestamp(start)]
    return (1 + daily).cumprod().rename("Equal-weight $3bn+ universe")


# ------------------------------------------------------------------ variants

BASE = Params()
FIXES = dict(market_filter=True, max_vol_pct=0.8, momentum="vol_adjusted")
ALL3, ALL3_EARN = "All three fixes", "All three fixes + earnings blackout (±5 days)"
MAIN = [
    BASE,
    Params(name="+ market filter (SPY above 200-day)", market_filter=True),
    Params(name="+ skip 20% most volatile", max_vol_pct=0.8),
    Params(name="+ volatility-adjusted ranking", momentum="vol_adjusted"),
    Params(name=ALL3, **FIXES),
    Params(name="Your rules + earnings blackout (±5 days)", earnings_days=5),
    Params(name=ALL3_EARN, earnings_days=5, **FIXES),
]
SENSITIVITIES = [
    Params(name="1-day SuperTrend (no 3-day rule)", confirm_days=1),
    Params(name="+ sell on death cross", death_cross_exit=True),
    Params(name="Costs 25 bps per side", cost_bps=25.0),
    Params(name="All three fixes + earnings blackout (±2 days)", earnings_days=2, **FIXES),
    Params(name="All three fixes + earnings (±5), costs 25 bps", earnings_days=5, cost_bps=25.0, **FIXES),
]
VARIANTS = MAIN + SENSITIVITIES


def run_all(variants: list[Params] | None = None) -> dict:
    variants = variants or VARIANTS
    t0 = time.time()
    m = load_market()
    log.info("market: %d dates x %d tickers (%.0fs)", *m.close.shape, time.time() - t0)
    res = [simulate(m, p) for p in variants]
    base = res[0]
    start = str(base.equity.index[0].date())
    bm = {"SPY": spy().reindex(base.equity.index).ffill(), "EW": equal_weight_universe(m, start)}
    out_dir = paths().root / "backtests" / "momentum_trend"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.concat([r.equity / r.equity.iloc[0] for r in res] +
                       [bm["SPY"] / bm["SPY"].iloc[0], bm["EW"] / bm["EW"].iloc[0]], axis=1)
    curves.columns = [r.params.name for r in res] + ["SPY", "Equal-weight $3bn+ universe"]
    curves.to_csv(out_dir / "equity_curves.csv")
    pd.concat([r.trades.assign(variant=r.params.name) for r in res]).to_csv(out_dir / "trades.csv", index=False)
    pd.concat([r.holdings.assign(variant=r.params.name) for r in res]).to_csv(
        out_dir / "month_end_holdings.csv", index=False)
    return {"market": m, "results": res, "benchmarks": bm, "curves": curves, "seconds": time.time() - t0}


# ------------------------------------------------------------------ reporting

def stats(eq: pd.Series) -> dict:
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    dd = eq / eq.cummax() - 1
    down = r[r < 0].std() * np.sqrt(252)
    return {"CAGR": cagr, "Volatility": vol, "Sharpe (rf=0)": r.mean() / r.std() * np.sqrt(252),
            "Sortino": r.mean() * 252 / down if down > 0 else np.nan, "Max drawdown": dd.min(),
            "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
            "Total return": eq.iloc[-1] / eq.iloc[0] - 1}


def yearly(eq: pd.Series) -> pd.Series:
    y = eq.groupby(eq.index.year).last()
    first = eq.iloc[0]
    return y / y.shift(1).fillna(first) - 1


def spells(flag: pd.Series) -> list[tuple]:
    """(first day, last day, sessions) of each run of True."""
    d = np.diff(np.concatenate([[0], flag.to_numpy().astype(int), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return [(flag.index[s], flag.index[e], int(e - s + 1)) for s, e in zip(starts, ends)]


PCT = ["CAGR", "Volatility", "Max drawdown", "Total return", "Win rate", "Avg invested"]
SHOW = ["CAGR", "Volatility", "Sharpe (rf=0)", "Max drawdown", "Calmar", "Total return", "Beta to SPY",
        "Trades per year", "Win rate", "Avg holding (trading days)", "Avg invested"]


def summary(res: list[Result], curves: pd.DataFrame) -> pd.DataFrame:
    spy_r = curves["SPY"].pct_change()
    rows = {}
    for r in res:
        s = stats(r.equity)
        closed = r.trades.dropna(subset=["exit"])
        rr = r.equity.pct_change()
        s.update({"Trades per year": len(closed) / ((r.equity.index[-1] - r.equity.index[0]).days / 365.25),
                  "Win rate": (closed["return"] > 0).mean(), "Avg holding (trading days)": closed["days"].mean(),
                  "Avg invested": r.invested.mean(), "Turnover per year": r.turnover,
                  "Beta to SPY": rr.cov(spy_r) / spy_r.var()})
        rows[r.params.name] = s
    for name in ("SPY", "Equal-weight $3bn+ universe"):
        eq = curves[name].dropna()
        rows[name] = stats(eq) | {"Beta to SPY": eq.pct_change().cov(spy_r) / spy_r.var()}
    return pd.DataFrame(rows).T


COUNTS = ["Trades per year", "Avg holding (trading days)"]


def fmt(tbl: pd.DataFrame) -> pd.DataFrame:
    out = tbl.copy().astype(object)
    for c in tbl.columns:
        out[c] = [("" if pd.isna(v) else f"{v:.1%}" if c in PCT else f"{v:,.0f}" if c in COUNTS or abs(v) >= 100
                   else f"{v:.2f}") for v in tbl[c]]
    return out


def earnings_split(m: Market, r: Result, days: int = 5) -> pd.DataFrame:
    """Daily returns of held stocks on the earnings reaction session, on the other sessions
    within `days` of it, and on all other sessions (open-to-close on entry, close-to-open on exit)."""
    cols, dates = m.close.columns, m.close.index
    O, C = m.open.to_numpy(), m.close.ffill().to_numpy()
    W, E0 = m.blackout(days).to_numpy(), m.blackout(0).to_numpy()
    kind, rets = [], []
    for t in r.trades.itertuples(index=False):
        j, a = cols.get_loc(t.ticker), dates.get_loc(t.entry)
        b = dates.get_loc(t.exit) if pd.notna(t.exit) else None
        i = np.arange(a + 1, b if b is not None else len(dates))
        rows = [np.array([a]), i] + ([np.array([b])] if b is not None else [])
        parts = [np.array([C[a, j] / O[a, j] - 1]), C[i, j] / C[i - 1, j] - 1]
        if b is not None:
            parts.append(np.array([O[b, j] / C[b - 1, j] - 1]))
        ii = np.concatenate(rows)
        kind.append(np.where(E0[ii, j], 0, np.where(W[ii, j], 1, 2)))
        rets.append(np.concatenate(parts))
    kind, rets = np.concatenate(kind), np.concatenate(rets)
    names = ["Earnings reaction session", f"Other sessions within ±{days}", "All other sessions"]
    total = np.nansum(rets)
    return pd.DataFrame({"Share of holding days": [np.mean(kind == k) for k in range(3)],
                         "Avg daily return": [np.nanmean(rets[kind == k]) for k in range(3)],
                         "Share of summed returns": [np.nansum(rets[kind == k]) / total for k in range(3)]},
                        index=names)


def reaction_spike(m: Market, offsets=(-1, 0, 1)) -> dict[int, float]:
    """Mean absolute daily return of universe stocks around earnings, relative to each stock's median."""
    a = m.close.pct_change(fill_method=None).abs()
    med = a.where(m.universe).median().to_numpy()
    A, U = a.to_numpy(), m.universe.to_numpy()
    ri = m.close.index.get_indexer(pd.DatetimeIndex(m.earnings["session"]).as_unit(m.close.index.unit))
    ci = m.close.columns.get_indexer(m.earnings["ticker"])
    k = (ri >= 0) & (ci >= 0)
    out = {}
    for off in offsets:
        r_, c_ = ri[k] + off, ci[k]
        ok = (r_ >= 1) & (r_ < len(A))
        r_, c_ = r_[ok], c_[ok]
        sel = U[r_, c_]
        out[off] = float(np.nanmean(A[r_, c_][sel] / med[c_][sel]))
    return out


def trade_line(r: Result) -> str:
    closed = r.trades.dropna(subset=["exit"])
    best = closed.sort_values("return", ascending=False).head(5)
    worst = closed.sort_values("return").head(3)
    return (f"{len(closed)} closed trades; median return {closed['return'].median():+.1%}, mean "
            f"{closed['return'].mean():+.1%}; best: " + ", ".join(f"{t.ticker} {t['return']:+.0%}" for _, t in
                                                                   best.iterrows()) +
            "; worst: " + ", ".join(f"{t.ticker} {t['return']:+.0%}" for _, t in worst.iterrows()) + ".")


def report(out: dict) -> str:
    res, curves, m = out["results"], out["curves"], out["market"]
    by = {r.params.name: r for r in res}
    base = res[0]
    start = base.equity.index[0]
    tbl = fmt(summary(res, curves))
    main_rows = [p.name for p in MAIN if p.name in by] + ["SPY", "Equal-weight $3bn+ universe"]
    sens_rows = [n for n in by if n not in main_rows]
    yr_cols = {"Your rules": base.equity}
    for name, label in ((ALL3, "All three fixes"), (ALL3_EARN, "All three + earnings")):
        if name in by:
            yr_cols[label] = by[name].equity
    yr_cols |= {"SPY": curves["SPY"].dropna(), "EW universe": curves["Equal-weight $3bn+ universe"].dropna()}
    yr = pd.DataFrame({k: yearly(v) for k, v in yr_cols.items()}).map(lambda v: f"{v:+.1%}")

    reasons = pd.DataFrame({r.params.name: r.trades.dropna(subset=["exit"])["reason"].value_counts()
                            for r in res if r.params.name in main_rows}).T.fillna(0).astype(int)
    reasons = reasons[[c for c in ("trend", "earnings", "market") if c in reasons.columns]]
    reasons.columns = [{"trend": "SuperTrend exit", "earnings": "Earnings blackout",
                        "market": "Market filter"}[c] for c in reasons.columns]

    lines = [
        f"# Momentum + trend backtest ({curves.index[0].date()} to {curves.index[-1].date()})", "",
        "Generated by `python -m backtest.momentum_trend` on "
        f"{dt.date.today()} from the stockdata panel. See backtest/momentum_trend.py for the rules.", "",
        "## Summary", "", md(tbl.loc[main_rows, SHOW], "Variant"), "",
        "Fixes: market filter = cash when SPY closes below its 200-day average 3 days in a row, back in after "
        "3 closes above; skip volatile = never buy a stock in the most volatile 20% of the universe "
        "(6-month volatility); volatility-adjusted ranking = momentum returns divided by 6-month volatility. "
        "Earnings blackout = sell 5 sessions before the first session that reacts to the results and do not "
        "buy again until 6 sessions after it.", "",
        "## Other sensitivities", "", md(tbl.loc[sens_rows, SHOW], "Variant"), "",
        "## Calendar-year returns", "", md(yr, "Year"), "",
        "## Why positions were sold (closed trades)", "", md(reasons, "Variant"), "",
    ]
    if m.benchmark is not None:
        off = ~m.risk_on(BASE.market_sma, BASE.confirm_days)
        off = off[off.index >= start]
        sp = spells(off)
        lines += ["## Market filter", "",
                  f"Out of the market (SPY filter) on {off.mean():.0%} of sessions, in {len(sp)} spells: " +
                  ", ".join(f"{a.date()} to {b.date()} ({n})" for a, b, n in sp) + ".", ""]
    if m.earnings is not None:
        near = event_window(m.earnings, m.close.index, m.close.columns, 70)
        u = m.universe[m.universe.index >= start]
        covered = (near.loc[u.index] & u).to_numpy().sum() / u.to_numpy().sum()
        miss = (u & ~near.loc[u.index]).sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        split = earnings_split(m, base, 5)
        sp_tbl = pd.DataFrame({"Share of holding days": split["Share of holding days"].map(lambda v: f"{v:.1%}"),
                               "Avg daily return": split["Avg daily return"].map(lambda v: f"{v:+.2%}"),
                               "Share of summed returns": split["Share of summed returns"].map(
                                   lambda v: f"{v:.0%}")})
        inside = 1 - split["Share of holding days"].iloc[2]
        inside_ret = 1 - split["Share of summed returns"].iloc[2]
        spike = reaction_spike(m)
        lines += ["## Earnings: where your rules make their money", "",
                  "Daily returns of the stocks held under your rules, split by distance to the nearest earnings "
                  "release:", "", md(sp_tbl, "Sessions"), "",
                  f"The ±5-session windows are {inside:.0%} of holding days but produced {inside_ret:.0%} of the "
                  "summed daily returns. The blackout swaps those days for ordinary ones (and adds trading costs), "
                  "which is why the blackout variants lag.", "",
                  "## Earnings dates", "",
                  f"From SEC 8-K Item 2.02 filings (stockdata/earnings.py). On the reaction session universe "
                  f"stocks move {spike[0]:.1f}x their median daily move (day before {spike[-1]:.1f}x, day after "
                  f"{spike[1]:.1f}x), which confirms the timing. {covered:.1%} of universe stock-days have a "
                  "release within 70 sessions; the rest get no blackout. The gap is mostly foreign companies "
                  "that report on 6-K or 40-F. Largest uncovered names (universe days): " +
                  ", ".join(f"{t} {n}" for t, n in miss.head(12).items()) + ".", ""]
    lines += ["## Trades", ""]
    for name in (base.params.name, ALL3, ALL3_EARN):
        if name in by:
            lines += [f"- {name}: " + trade_line(by[name])]
    sect = base.trades.groupby("sector").size().sort_values(ascending=False)
    lines += ["", "Trades by sector (your rules): " + ", ".join(f"{k} {v}" for k, v in sect.items()) + ".", "",
              "Price artifacts neutralised (Alpaca did not adjust): " +
              ", ".join(f"{t} {d} x{R:g}" for t, d, R in m.jumps_fixed) + ".", "",
              "## Caveats", "",
              "- Survivorship bias: only stocks listed today are in the data; companies that were acquired, went "
              "bankrupt or were delisted are missing, which flatters results (most for momentum). Compare with the "
              "equal-weight universe, which has the same bias, rather than with SPY alone.",
              "- Fills at the next open with fixed costs; no market-impact model, no borrow, cash earns 0% "
              "(this understates the market-filter variants, which sit in cash for long spells).",
              "- The earnings blackout uses the actual release dates, i.e. assumes each date is known at least "
              "6 sessions ahead. Companies usually announce it 2 to 4 weeks before, but not always.",
              "- Sector is today's SIC-based sector for every date."]
    return "\n".join(lines)


def md(df: pd.DataFrame, first: str = "") -> str:
    d = df.reset_index()
    if first:
        d = d.rename(columns={d.columns[0]: first})
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    out = run_all()
    text = report(out)
    (REPO_ROOT / "backtest" / "RESULTS.md").write_text(text + "\n")
    print(text)
    log.info("done in %.0fs", out["seconds"])


if __name__ == "__main__":
    main()

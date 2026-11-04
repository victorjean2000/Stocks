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

Run: python -m backtest.momentum_trend
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockdata import alpaca, load_panel
from stockdata.config import REPO_ROOT, paths

from .indicators import confirmed, momentum_features, sma, supertrend_bull

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

    def score(self, kind: str) -> pd.DataFrame:
        key = ("score", kind)
        if key not in self._cache:
            f = self._cache.get("feats") or momentum_features(self.close)
            self._cache["feats"] = f
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


def load_market(start: str = "2016-01-01") -> Market:
    """Wide price frames for every ticker that is ever in the universe."""
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
    return m


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

    dates = m.close.index
    tick = m.close.columns.to_numpy()
    O = m.open.to_numpy()
    C = m.close.ffill().to_numpy()
    E, X, S = eligible.to_numpy(), exit_sig.to_numpy(), score.to_numpy()
    sec_codes, sec_names = pd.factorize(m.sector)
    i0 = int(dates.searchsorted(pd.Timestamp(p.start)))
    n_t, n_s = C.shape
    cost = p.cost_bps / 1e4

    cash = p.capital
    shares = np.zeros(n_s)
    held: set[int] = set()
    entry: dict[int, tuple] = {}
    pend_sell: list[int] = []
    pend_buy: list[int] = []
    eq = np.full(n_t, np.nan)
    inv = np.full(n_t, np.nan)
    traded = 0.0
    trades = []
    snap = []

    for i in range(i0, n_t):
        # 1) orders decided at yesterday's close fill at today's open
        if pend_sell:
            retry = []
            for j in pend_sell:
                px = O[i, j]
                if not np.isfinite(px):
                    retry.append(j)
                    continue
                value = shares[j] * px
                cash += value * (1 - cost)
                traded += value
                ei, epx, _ = entry[j]
                trades.append({"ticker": tick[j], "sector": sec_names[sec_codes[j]], "entry": dates[ei],
                               "exit": dates[i], "entry_px": epx, "exit_px": px,
                               "return": px / epx - 1, "days": i - ei})
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
        pend_sell = sorted(set(pend_sell) | {j for j in held if X[i, j]})
        remaining = [j for j in held if j not in pend_sell]
        n_free = p.n_positions - len(remaining)
        if n_free > 0:
            cand = np.flatnonzero(E[i])
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
                       "entry_px": epx, "exit_px": C[-1, j], "return": C[-1, j] / epx - 1, "days": n_t - 1 - ei})
    equity = pd.Series(eq[i0:], index=dates[i0:], name=p.name).dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    turnover = traded / 2 / equity.mean() / years  # one-way, per year
    return Result(p, equity, pd.Series(inv[i0:], index=dates[i0:]), pd.DataFrame(trades),
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


VARIANTS = [
    Params(),
    Params(name="1-day SuperTrend (no 3-day rule)", confirm_days=1),
    Params(name="+ sell on death cross", death_cross_exit=True),
    Params(name="Volatility-adjusted momentum", momentum="vol_adjusted"),
    Params(name="Costs 25 bps per side", cost_bps=25.0),
]


def run_all() -> dict:
    t0 = time.time()
    m = load_market()
    log.info("market: %d dates x %d tickers (%.0fs)", *m.close.shape, time.time() - t0)
    res = [simulate(m, p) for p in VARIANTS]
    base = res[0]
    start = str(base.equity.index[0].date())
    bm = {"SPY": spy().reindex(base.equity.index).ffill(), "EW": equal_weight_universe(m, start)}
    out_dir = paths().root / "backtests" / "momentum_trend"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.concat([r.equity / r.equity.iloc[0] for r in res] +
                       [bm["SPY"] / bm["SPY"].iloc[0], bm["EW"] / bm["EW"].iloc[0]], axis=1)
    curves.columns = [r.params.name for r in res] + ["SPY", "Equal-weight $3bn+ universe"]
    curves.to_csv(out_dir / "equity_curves.csv")
    base.trades.to_csv(out_dir / "trades.csv", index=False)
    base.holdings.to_csv(out_dir / "month_end_holdings.csv", index=False)
    return {"market": m, "results": res, "benchmarks": bm, "curves": curves, "seconds": time.time() - t0}


def report(out: dict) -> str:
    res, curves = out["results"], out["curves"]
    base = res[0]
    rows = {}
    for r in res:
        s = stats(r.equity)
        closed = r.trades.dropna(subset=["exit"])
        s.update({"Trades per year": len(closed) / ((r.equity.index[-1] - r.equity.index[0]).days / 365.25),
                  "Win rate": (closed["return"] > 0).mean(), "Avg holding (trading days)": closed["days"].mean(),
                  "Avg invested": r.invested.mean(), "Turnover per year": r.turnover})
        rows[r.params.name] = s
    for name in ("SPY", "Equal-weight $3bn+ universe"):
        rows[name] = stats(curves[name].dropna())
    tbl = pd.DataFrame(rows).T
    pct = ["CAGR", "Volatility", "Max drawdown", "Total return", "Win rate", "Avg invested"]
    fmt = tbl.copy().astype(object)
    for c in tbl.columns:
        fmt[c] = [("" if pd.isna(v) else (f"{v:.1%}" if c in pct else (f"{v:.2f}" if abs(v) < 100 else f"{v:,.0f}")))
                  for v in tbl[c]]
    yr = pd.DataFrame({"Strategy": yearly(base.equity), "SPY": yearly(curves["SPY"].dropna()),
                       "EW universe": yearly(curves["Equal-weight $3bn+ universe"].dropna())})
    yr = yr.map(lambda v: f"{v:+.1%}")
    closed = base.trades.dropna(subset=["exit"])
    best = closed.sort_values("return", ascending=False).head(5)
    sect = base.trades.groupby("sector").size().sort_values(ascending=False)
    lines = [
        f"# Momentum + trend backtest ({curves.index[0].date()} to {curves.index[-1].date()})", "",
        "Generated by `python -m backtest.momentum_trend` on "
        f"{dt.date.today()} from the stockdata panel. See backtest/momentum_trend.py for the rules.", "",
        "## Summary", "", md(fmt[["CAGR", "Volatility", "Sharpe (rf=0)", "Max drawdown", "Calmar", "Total return",
                                  "Trades per year", "Win rate", "Avg holding (trading days)", "Avg invested"]]), "",
        "## Calendar-year returns (your rules)", "", md(yr), "",
        "## Trades (your rules)", "",
        f"{len(closed)} closed trades; median return {closed['return'].median():+.1%}, mean {closed['return'].mean():+.1%};"
        f" best: " + ", ".join(f"{r.ticker} {r['return']:+.0%}" for _, r in best.iterrows()) + ".", "",
        "Trades by sector: " + ", ".join(f"{k} {v}" for k, v in sect.items()) + ".", "",
        "Price artifacts neutralised (Alpaca did not adjust): " +
        ", ".join(f"{t} {d} x{R:g}" for t, d, R in out["market"].jumps_fixed) + ".", "",
        "## Caveats", "",
        "- Survivorship bias: only stocks listed today are in the data; companies that were acquired, went "
        "bankrupt or were delisted are missing, which flatters results (most for momentum). Compare with the "
        "equal-weight universe, which has the same bias, rather than with SPY alone.",
        "- Fills at the next open with fixed costs; no market-impact model, no borrow, cash earns 0%.",
        "- Sector is today's SIC-based sector for every date.",
    ]
    return "\n".join(lines)


def md(df: pd.DataFrame) -> str:
    d = df.reset_index()
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

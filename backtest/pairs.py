"""Pairs trading: when two similar stocks drift apart, short the one that rose and buy the one that fell, and close
when they come back together.

Two tests on the stockdata panel (Alpaca prices adjusted for splits and dividends, 2016-2026):

1. Famous pairs, picked by hand (KO / PEP and twelve others). For each: how closely the two move, how far apart they
   drifted, how fast gaps closed (half-life of the log price ratio) and whether the ratio is stationary (an augmented
   Dickey-Fuller test, 5% level). The rule: the spread is the log price ratio; z = its distance from its N-day average
   in N-day standard deviations. When z rises above +entry, short A and buy B (equal amounts); below -entry the
   opposite; close when z gets back to 0, or after 60 sessions. Signals at the close, trades at the next open, 10 bps a
   side on each leg, 0.5% a year to borrow the short leg, no interest on cash. Lookback and entry are picked on
   2017-21 and checked on 2022-26. Choosing pairs that are known today to have stayed close flatters this test.

2. Pairs picked by the data, without hindsight (the method of Gatev, Goetzmann and Rouwenhorst, 2006): every six
   months, among the 100 largest stocks (point-in-time market cap), the 20 same-sector pairs whose normalised prices
   stayed closest over the past year; trade each for the next six months, opening when the gap passes two standard
   deviations of the past year's gap and closing when the prices cross (or at the period end).

Returns are per unit of capital with that amount long and the same amount short (as a margin account allows); a
portfolio gives each pair an equal share. The panel holds only companies still listed today, so pairs whose stock
later collapsed are missing, which also flatters the results.

Run: python -m backtest.pairs (about 1 minute). Writes backtest/PAIRS.md.
"""
from __future__ import annotations

import datetime as dt
import logging
import time

import numpy as np
import pandas as pd

from stockdata import load_panel
from stockdata.config import REPO_ROOT, paths

from .momentum_trend import md, spy, stats, tbill_returns, yearly

log = logging.getLogger(__name__)

PAIRS = [("KO", "PEP", "Coca-Cola / Pepsi"), ("V", "MA", "Visa / Mastercard"), ("HD", "LOW", "Home Depot / Lowe's"),
         ("XOM", "CVX", "Exxon / Chevron"), ("JPM", "BAC", "JPMorgan / Bank of America"),
         ("C", "WFC", "Citigroup / Wells Fargo"), ("UPS", "FDX", "UPS / FedEx"), ("T", "VZ", "AT&T / Verizon"),
         ("PG", "CL", "Procter & Gamble / Colgate"), ("WMT", "TGT", "Walmart / Target"),
         ("MCD", "YUM", "McDonald's / Yum! Brands"), ("LMT", "NOC", "Lockheed Martin / Northrop Grumman"),
         ("DUK", "SO", "Duke Energy / Southern Co")]
START, SPLIT = "2017-01-03", "2021-12-31"
COST, BORROW = 10.0, 0.005  # bps a side on each leg; yearly fee on the short leg
LOOKBACKS, ENTRIES = (20, 60, 120), (1.5, 2.0, 2.5)
MAX_HOLD = 60
ADF_5PC = -2.86  # 5% critical value, constant, no trend (large samples)


# ------------------------------------------------------------------ statistics of a spread

def half_life(s: pd.Series) -> float:
    """Sessions for half of a gap to close: from ds_t = a + b s_{t-1} (np.inf when b >= 0)."""
    s = s.dropna()
    x, y = s.shift(1).iloc[1:].to_numpy(), s.diff().iloc[1:].to_numpy()
    b = np.polyfit(x, y, 1)[0]
    return float(-np.log(2) / np.log(1 + b)) if b < 0 else np.inf


def adf_t(s: pd.Series, lags: int = 1) -> float:
    """Augmented Dickey-Fuller t statistic (constant, `lags` lagged differences): below about -2.86 the series is
    stationary at the 5% level (gaps close), above it a random walk cannot be ruled out."""
    s = s.dropna().to_numpy(float)
    d = np.diff(s)
    rows = len(d) - lags
    X = np.column_stack([np.ones(rows), s[lags:-1]] + [d[lags - k:-k] for k in range(1, lags + 1)])
    y = d[lags:]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / (rows - X.shape[1])
    se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(beta[1] / se)


# ------------------------------------------------------------------ the z-score rule on one pair

def zscore(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    s = np.log(a) - np.log(b)
    return (s - s.rolling(n).mean()) / s.rolling(n).std()


def positions(z: pd.Series, entry: float, exit_z: float = 0.0, max_hold: int | None = MAX_HOLD,
              stop: float | None = None) -> np.ndarray:
    """Position decided at each close: +1 = long A / short B (A cheap: z below -entry), -1 = short A / long B, 0 =
    flat; closed when z gets back to exit_z, after max_hold sessions, or (stop) when |z| passes `stop`."""
    zz = z.to_numpy(float)
    pos, cur, held = np.zeros(len(zz), int), 0, 0
    for i, v in enumerate(zz):
        if not np.isfinite(v):
            pos[i] = cur
            continue
        if cur == 0:
            if v > entry:
                cur, held = -1, 0
            elif v < -entry:
                cur, held = 1, 0
        else:
            held += 1
            back = (cur == 1 and v >= -exit_z) or (cur == -1 and v <= exit_z)
            if back or (max_hold is not None and held >= max_hold) or (stop is not None and abs(v) >= stop):
                cur = 0
        pos[i] = cur
    return pos


def run_pair(px: dict[str, pd.DataFrame], a: str, b: str, pos: np.ndarray, dates: pd.DatetimeIndex,
             start: str = START, cost_bps: float = COST, borrow: float = BORROW) -> tuple[pd.Series, pd.DataFrame]:
    """Equity of one pair (1 at the close before `start`) and its trades. pos[i] is decided at close i and traded at
    open i + 1: long one leg and short the other, equal amounts of the equity at entry, shares fixed until the exit."""
    oa, ca = px[a]["o"].reindex(dates).to_numpy(), px[a]["c"].reindex(dates).to_numpy()
    ob, cb = px[b]["o"].reindex(dates).to_numpy(), px[b]["c"].reindex(dates).to_numpy()
    cost, fee = cost_bps / 1e4, borrow / 252
    i0 = int(dates.searchsorted(pd.Timestamp(start))) - 1
    eq = np.full(len(dates), np.nan)
    eq[i0] = level = 1.0
    cur, entry, trades = 0, None, []
    for i in range(i0 + 1, len(dates)):
        want = pos[i - 1]
        if want != cur and cur != 0:  # close at today's open
            e, pa, pb, k, before = entry
            gain = cur * ((oa[i] / pa - 1) - (ob[i] / pb - 1))
            level = e * (1 + gain - fee * (i - k) - cost * (oa[i] / pa + ob[i] / pb))
            trades.append({"side": "long A / short B" if cur == 1 else "short A / long B", "entry": dates[k],
                           "exit": dates[i], "days": i - k, "ret": level / before - 1})  # both legs' costs in
            cur, entry = 0, None
        if want != 0 and cur == 0 and np.isfinite(oa[i]) and np.isfinite(ob[i]):  # open at today's open
            before = level
            level *= 1 - 2 * cost
            cur, entry = want, (level, oa[i], ob[i], i, before)
        if cur != 0:
            e, pa, pb, k, _ = entry
            eq[i] = e * (1 + cur * ((ca[i] / pa - 1) - (cb[i] / pb - 1)) - fee * (i - k + 1))
        else:
            eq[i] = level
    return pd.Series(eq, index=dates).dropna(), pd.DataFrame(trades, columns=["side", "entry", "exit", "days", "ret"])


# ------------------------------------------------------------------ data

def load(tickers: list[str]) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    df = load_panel(tickers=tickers, columns=["open", "close"])
    px = {t: g.set_index("date")[["open", "close"]].rename(columns={"open": "o", "close": "c"}).sort_index()
          for t, g in df.groupby("ticker")}
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    return px, dates


def measure(eq: pd.Series, trades: pd.DataFrame | None = None, bench: pd.Series | None = None) -> dict:
    split = pd.Timestamp(SPLIT)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    row = {"CAGR": stats(eq)["CAGR"], "2017-21": stats(eq[eq.index <= split])["CAGR"],
           "2022-26": stats(eq[eq.index >= split])["CAGR"], "Sharpe": stats(eq)["Sharpe (rf=0)"],
           "Max drawdown": stats(eq)["Max drawdown"], "Worst year": yearly(eq).min()}
    if trades is not None:
        row.update({"Trades a year": len(trades) / years,
                    "Win rate": (trades["ret"] > 0).mean() if len(trades) else np.nan,
                    "Average trade": trades["ret"].mean() if len(trades) else np.nan,
                    "Average hold (sessions)": trades["days"].mean() if len(trades) else np.nan})
    if bench is not None:
        r, rb = eq.pct_change(), bench.reindex(eq.index).ffill().pct_change()
        row["Correlation with the S&P 500"] = r.corr(rb)
    return row


# ------------------------------------------------------------------ pairs picked by the data (no hindsight)

FORMATION, N_PAIRS, TOP, GGR_SIGMA = 252, 20, 100, 2.0


def closest_pairs(close: pd.DataFrame, sector: pd.Series, n_pairs: int = N_PAIRS, same_sector: bool = True):
    """The n_pairs pairs (same sector) whose normalised prices (1 at the window start) stayed closest over the
    window: smallest sum of squared differences. Returns [(a, b, std of the difference)]."""
    x = (close / close.iloc[0]).to_numpy().T  # stocks x days
    sq = (x ** 2).sum(axis=1)
    dist = sq[:, None] + sq[None, :] - 2 * x @ x.T
    names = close.columns.to_numpy()
    sec = sector.reindex(close.columns).to_numpy()
    iu, ju = np.triu_indices(len(names), k=1)
    ok = (sec[iu] == sec[ju]) if same_sector else np.ones(len(iu), bool)
    iu, ju, dd = iu[ok], ju[ok], dist[iu[ok], ju[ok]]
    best = np.argsort(dd)[:n_pairs]
    return [(names[iu[k]], names[ju[k]], float(np.std(x[iu[k]] - x[ju[k]]))) for k in best]


def reverting_pairs(close: pd.DataFrame, sector: pd.Series, n_pairs: int = N_PAIRS):
    """The n_pairs same-sector pairs whose log price ratio was most clearly mean-reverting over the window (most
    negative ADF t statistic, all below the 5% level). Returns [(a, b, std of the normalised price difference)]."""
    logp = np.log(close)
    x = (close / close.iloc[0]).to_numpy().T
    names = close.columns.to_numpy()
    sec = sector.reindex(close.columns).to_numpy()
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if sec[i] != sec[j]:
                continue
            t = adf_t(logp.iloc[:, i] - logp.iloc[:, j])
            if t < ADF_5PC:
                rows.append((t, i, j))
    rows.sort()
    return [(names[i], names[j], float(np.std(x[i] - x[j]))) for _, i, j in rows[:n_pairs]]


def ggr_positions(na: np.ndarray, nb: np.ndarray, sigma: float, k: float = GGR_SIGMA) -> np.ndarray:
    """Position at each close of a trading period from the normalised prices (1 at the period start): open when the
    gap passes k sigma (short the higher one), close when the prices cross; flat on the last day."""
    pos, cur = np.zeros(len(na), int), 0
    for i in range(len(na)):
        d = na[i] - nb[i]
        if not np.isfinite(d):
            pos[i] = cur
            continue
        if cur == 0:
            if d > k * sigma:
                cur = -1
            elif d < -k * sigma:
                cur = 1
        elif (cur == 1 and d >= 0) or (cur == -1 and d <= 0):
            cur = 0
        pos[i] = cur
    pos[-1] = 0
    return pos


def data_picked(m, top: int = TOP, n_pairs: int = N_PAIRS, same_sector: bool = True, cost_bps: float = COST,
                borrow: float = BORROW, first: str = "2017-01-01", select: str = "distance"
                ) -> tuple[pd.Series, pd.DataFrame, list]:
    """Every six months: pick the pairs on the past year (the `top` largest stocks then, point-in-time), trade them for
    six months, each with 1 / n_pairs of the capital. Returns the chained equity, the trades and the pairs chosen."""
    dates = m.close.index
    starts = [d for d in pd.date_range(first, dates[-1], freq="6MS") if d <= dates[-1]]
    level, parts, trades, chosen = 1.0, [], [], []
    px_cache: dict[str, pd.DataFrame] = {}
    for k, s0 in enumerate(starts):
        i_start = int(dates.searchsorted(s0))
        i_end = int(dates.searchsorted(starts[k + 1])) if k + 1 < len(starts) else len(dates)
        if i_start - FORMATION < 0 or i_start >= len(dates) - 1:
            continue
        form = m.close.iloc[i_start - FORMATION:i_start]
        last = i_start - 1
        big = m.universe.iloc[last] & (m.size_rank.iloc[last] <= top)
        cols = [c for c in big[big].index if form[c].notna().all()]
        pairs = (reverting_pairs(form[cols], m.sector, n_pairs) if select == "adf" else
                 closest_pairs(form[cols], m.sector, n_pairs, same_sector))
        if not pairs:
            continue
        chosen.append((dates[i_start], pairs))
        window = dates[last:min(i_end + 1, len(dates))]  # the close before, the period, the next open to exit
        curves = []
        for a, b, sd in pairs:
            for t in (a, b):
                if t not in px_cache:
                    px_cache[t] = pd.DataFrame({"o": m.open[t], "c": m.close[t]})
            ca, cb = px_cache[a]["c"].reindex(window), px_cache[b]["c"].reindex(window)
            na, nb = (ca / ca.iloc[0]).to_numpy(), (cb / cb.iloc[0]).to_numpy()
            pos = ggr_positions(na, nb, sd)
            pos[0] = 0
            if len(window) > i_end - last:  # the extra session only closes what is still open
                pos[-2:] = 0
            eq, tr = run_pair(px_cache, a, b, pos, window, start=str(window[1].date()), cost_bps=cost_bps,
                              borrow=borrow)
            curves.append(eq.reindex(window).ffill())
            if len(tr):
                trades.append(tr.assign(pair=f"{a} / {b}", period=str(dates[i_start].date())))
        period = pd.concat(curves, axis=1).mean(axis=1) * level
        parts.append(period.iloc[1:] if parts else period)
        level = float(period.iloc[-1])
    eq = pd.concat(parts)
    eq = eq[~eq.index.duplicated(keep="last")]
    return eq, (pd.concat(trades) if trades else pd.DataFrame()), chosen


# ------------------------------------------------------------------ report

def famous(px: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = {}
    split = pd.Timestamp(SPLIT)
    for a, b, name in PAIRS:
        ca, cb = px[a]["c"].reindex(dates), px[b]["c"].reindex(dates)
        sp = np.log(ca) - np.log(cb)
        drift = float(np.exp(sp.iloc[-1] - sp.iloc[0]) - 1)
        hl = half_life(sp)
        rows[name] = {"Daily moves correlated": f"{ca.pct_change().corr(cb.pct_change()):.2f}",
                      "How far apart they drifted": f"{a} {'+' if drift >= 0 else ''}{drift:.0%} against {b}",
                      "Half of a gap closes in": "never" if not np.isfinite(hl) else f"{hl:.0f} sessions",
                      "Gaps close (ADF test), 2016-26": "yes" if adf_t(sp) < ADF_5PC else "no",
                      "2016-21": "yes" if adf_t(sp[sp.index <= split]) < ADF_5PC else "no",
                      "2022-26": "yes" if adf_t(sp[sp.index > split]) < ADF_5PC else "no"}
    return pd.DataFrame(rows).T


def run_famous(px, dates, sp, n: int, e: float, cost_bps: float = COST, borrow: float = BORROW):
    curves, trades, per = [], [], {}
    for a, b, name in PAIRS:
        z = zscore(px[a]["c"].reindex(dates), px[b]["c"].reindex(dates), n)
        eq, tr = run_pair(px, a, b, positions(z, e), dates, cost_bps=cost_bps, borrow=borrow)
        curves.append(eq)
        trades.append(tr.assign(pair=name))
        per[name] = measure(eq, tr) | {"Worst trade": tr["ret"].min() if len(tr) else np.nan}
    port = pd.concat(curves, axis=1).ffill().mean(axis=1)
    alltr = pd.concat(trades)
    return port, alltr, pd.DataFrame(per).T


def run_all() -> dict:
    t0 = time.time()
    from .momentum_trend import load_market
    tick = sorted({t for a, b, _ in PAIRS for t in (a, b)})
    px, dates = load(tick)
    sp = spy().reindex(dates).ffill()
    cash = (1 + tbill_returns(dates)).cumprod()
    grid = {}
    for n in LOOKBACKS:
        for e in ENTRIES:
            port, tr, _ = run_famous(px, dates, sp, n, e)
            grid[(n, e)] = measure(port, tr, sp)
    grid = pd.DataFrame(grid).T
    pick = grid["2017-21"].idxmax()
    port, tr, per = run_famous(px, dates, sp, *pick)
    gross, gtr, _ = run_famous(px, dates, sp, *pick, cost_bps=0.0, borrow=0.0)
    ko = px["KO"]["c"].reindex(dates) / px["PEP"]["c"].reindex(dates)
    m = load_market(earnings=False)
    picked = {}
    for label, kw in (("The 20 closest same-sector pairs among the 100 largest", {}),
                      ("... before costs", {"cost_bps": 0.0, "borrow": 0.0}),
                      ("The 20 closest pairs of any sector among the 100 largest", {"same_sector": False}),
                      ("The 20 closest same-sector pairs among the 300 largest", {"top": 300}),
                      ("The 20 same-sector pairs whose gap closed most reliably in the past year (ADF test)",
                       {"select": "adf"})):
        eq, gtr2, chosen = data_picked(m, **kw)
        picked[label] = (eq, gtr2, chosen)
    return {"px": px, "dates": dates, "spy": sp, "cash": cash, "famous": famous(px, dates), "grid": grid,
            "pick": pick, "port": port, "trades": tr, "per": per, "gross": gross, "gross_trades": gtr, "ko": ko,
            "picked": picked, "seconds": time.time() - t0}


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    for c in df.columns:
        if c in ("Sharpe", "Correlation with the S&P 500"):
            out[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        elif c in ("Trades a year", "Average hold (sessions)"):
            out[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
        else:
            out[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}" if c in ("Average trade", "Worst trade")
                               else f"{v:.1%}")
    return out


def report(out: dict) -> str:
    grid, pick, port, tr, per = out["grid"], out["pick"], out["port"], out["trades"], out["per"]
    sp, cash = out["spy"], out["cash"]
    first, last = port.index[0].date(), port.index[-1].date()
    sp_p = sp[sp.index >= port.index[0]]
    c_p = cash[cash.index >= port.index[0]]
    bench = pd.DataFrame({"S&P 500 held (SPY)": measure(sp_p / sp_p.iloc[0]),
                          "Cash (T-bills, BIL)": measure(c_p / c_p.iloc[0])}).T
    g = grid.copy()
    g.index = [f"{n}-day z-score, open beyond {e:g}" for n, e in g.index]
    best = grid.loc[pick]
    gross = measure(out["gross"], out["gross_trades"])
    fam = out["famous"]
    stationary = int((fam["Gaps close (ADF test), 2016-26"] == "yes").sum())
    picked = out["picked"]
    ptab = pd.DataFrame({k: measure(eq, t, sp) for k, (eq, t, _) in picked.items()}).T
    main_pick = next(iter(picked))
    ex_first = picked[main_pick][2][0]
    ex_last = picked[main_pick][2][-1]
    ko = out["ko"]
    ko_years = ko.groupby(ko.index.year).last() / ko.iloc[0]
    ko_tr = tr[tr["pair"] == "Coca-Cola / Pepsi"]
    cols = ["CAGR", "2017-21", "2022-26", "Sharpe", "Max drawdown", "Trades a year", "Win rate", "Average trade",
            "Average hold (sessions)", "Correlation with the S&P 500"]
    rates = pd.concat([grid["Win rate"], ptab["Win rate"]])
    wr_lo, wr_hi = rates.min(), rates.max()
    lines = [
        f"# Pairs trading ({first} to {last})", "",
        f"Generated by `python -m backtest.pairs` on {dt.date.today()} in {out['seconds']:.0f} s. Short the stock "
        "that pulled ahead of its partner, buy the one that fell behind, close when they meet again. Alpaca prices "
        "adjusted for dividends, signals at the close, trades at the next open, 10 bps a side on each leg, 0.5% a "
        "year to borrow the short leg, no interest on cash. Returns per unit of capital with that much long and as "
        "much short. See the module docstring.", "",
        "## Summary", "",
        f"- **Famous pairs mostly drift apart:** of {len(fam)} well-known pairs, only {stationary} had a price ratio "
        "that reliably came back over 2016-26 (augmented Dickey-Fuller test, 5% level). Coca-Cola gained "
        f"{ko.iloc[-1] / ko.iloc[0] - 1:+.0%} against Pepsi and the gap never closed.",
        f"- **The rule on all {len(fam)} famous pairs:** {len(grid)} settings made between {grid['CAGR'].min():+.1%} "
        f"and {grid['CAGR'].max():+.1%} a year. The best on 2017-21 ({pick[0]}-day z-score, open beyond {pick[1]:g}) "
        f"made {best['2017-21']:+.1%} a year then and {best['2022-26']:+.1%} in 2022-26; before costs and borrowing "
        f"{gross['CAGR']:+.1%} a year. It won {best['Win rate']:.0%} of its trades, but the losers were bigger.",
        f"- **Pairs picked by the data, without hindsight:** {ptab.iloc[0]['CAGR']:+.1%} a year after costs "
        f"({ptab.iloc[1]['CAGR']:+.1%} before), against {bench.loc['Cash (T-bills, BIL)', 'CAGR']:.1%} for T-bills "
        f"and {bench.loc['S&P 500 held (SPY)', 'CAGR']:.1%} for the S&P 500.",
        "- **Verdict:** the idea is real (the spread does tend to narrow a little more often than not: win rates of "
        f"{wr_lo:.0%}-{wr_hi:.0%}), but in large US stocks in 2017-26 the edge was too small to pay for trading both "
        "legs and "
        "borrowing the short one, and a pair that splits for good (a company's business changing) wipes out many "
        "small wins. It was much more profitable decades ago, before many funds traded it (the classic study, "
        "Gatev, Goetzmann and Rouwenhorst 2006, found up to about 11% a year above cash in 1962-2002, with smaller "
        "profits in its later years).", "",
        "## The famous pairs", "",
        md(fam, "Pair"), "",
        "A ratio that never comes back is a trend, not a gap: shorting the winner of such a pair loses steadily.",
        "",
        "## The rule on the famous pairs", "",
        "All 13 pairs together, equal capital each (z-score of the log price ratio over the lookback; open beyond "
        f"the entry level, close at 0 or after {MAX_HOLD} sessions):", "",
        md(_fmt(g[cols]), "Setting"), "",
        f"Each pair with the setting picked on 2017-21 ({pick[0]}-day z-score, open beyond {pick[1]:g}):", "",
        md(_fmt(per[["CAGR", "2017-21", "2022-26", "Trades a year", "Win rate", "Average trade", "Worst trade",
                     "Max drawdown"]]), "Pair"), "",
        "## Coca-Cola and Pepsi", "",
        "Coca-Cola's price against Pepsi's (1.00 at the start of 2016), at each year end:", "",
        md(pd.DataFrame({"KO / PEP": ko_years.map(lambda v: f"{v:.2f}")}).T.rename(columns=str), "Ratio"), "",
        f"The rule traded the pair {len(ko_tr)} times; {(ko_tr['ret'] > 0).mean():.0%} won, the average trade made "
        f"{ko_tr['ret'].mean():+.2%} and the worst {ko_tr['ret'].min():+.1%}.", "",
        "## Pairs picked by the data (no hindsight)", "",
        "Every six months: the pairs whose normalised prices stayed closest over the past year, traded for six "
        "months; open when the gap passes two standard deviations of the past year's gap, close when the prices "
        "cross (Gatev, Goetzmann and Rouwenhorst, 2006).", "",
        md(_fmt(ptab[cols]), "Pairs"), "",
        f"First pick ({ex_first[0]:%b %Y}): " + ", ".join(f"{a}/{b}" for a, b, _ in ex_first[1][:8]) + ". Latest "
        f"({ex_last[0]:%b %Y}): " + ", ".join(f"{a}/{b}" for a, b, _ in ex_last[1][:8]) + ".", "",
        "## For comparison", "",
        md(_fmt(bench[["CAGR", "2017-21", "2022-26", "Max drawdown"]]), "Holding"), "",
        "## Notes", "",
        "- An ISA cannot hold short positions; pairs trading needs a margin account, CFDs or spread bets, whose "
        "financing costs would come on top.",
        "- The short leg pays the dividends of the stock borrowed; the adjusted prices used here include that.",
        "- Survivorship: the panel holds only companies still listed today, so pairs in which one stock collapsed "
        "(the worst case for this strategy) are missing; real results would be worse.",
        "- No interest on cash: in a margin account the collateral could earn some, which would bring the results "
        "up to about the T-bill rate at best, still nothing for the risk taken."]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    out = run_all()
    text = report(out)
    (REPO_ROOT / "backtest" / "PAIRS.md").write_text(text + "\n")
    out_dir = paths().root / "backtests" / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out["grid"].to_csv(out_dir / "grid.csv")
    out["trades"].to_csv(out_dir / "famous_trades.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()

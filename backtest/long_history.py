"""The ISA dip rule over 2000-2026: through the 2000-02 and 2007-09 bear markets, before the years it was chosen on.

The rule (RECOMMENDED in backtest/isa.py): half in a 3x Nasdaq-100 fund and half in a 3x S&P 500 fund; buy after the
index (QQQ / SPY) closes in the bottom 10% of the day's range (IBS < 0.1), sell after it closes above its 5-day average.

Data: Yahoo daily prices of QQQ (from March 1999) and SPY (from 1993), approved by the user on 2026-11-22 for this test,
used before 2016; Alpaca's from 2016 (the official closing auction prices). Where both exist Yahoo's QQQ closes match,
but its SPY closes are wrong on a few extreme days (7 in March 2020), so SPY's earlier crash days may hold such errors;
a Nasdaq-100-only version is shown too. Dividends are included (Yahoo's adjusted close; open, high and low scaled by the
same factor).

3x funds did not exist before 2009-10, so they are simulated the way LQQ3, 3LUS, TQQQ and UPRO work: each day three
times the index's return, reset every close, less their running cost; the open is three times the overnight move. The
running cost is what the real TQQQ and UPRO lost against three times their index in 2016-26 (fees and the cost of
borrowing for the leverage, about 6.7% a year while held; T-bill rates averaged about the same in 2000-15 as in 2016-26);
a check uses the 0.75% fee alone. 2x funds pay half of it. Idle cash earns nothing (conservative for this rule, which is
in cash four fifths of the time). Everything is in dollars (the approved pound rates start in 2016). Trades at the next
US open, 10 bps a side (the ISA trades at the London open, about 3:00 in New York, which earned about the same as the US
open in 2016-26: backtest/ISA_TESTS.md).

Run: python -m backtest.long_history (about 10 s). Writes backtest/LONG_HISTORY.md.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from stockdata.config import REPO_ROOT, paths

from . import isa
from .daily import run
from .indicators import regime
from .isa_tests import exit_flags, ibs, us_state
from .lqq3 import fetch_yahoo
from .momentum_trend import md, stats
from .strategies import yearly_returns

log = logging.getLogger(__name__)

INDICES = ("QQQ", "SPY")
FEE = 0.0075  # WisdomTree's yearly fee on LQQ3 / 3LUS
COST = 10.0
START = "2000-01-03"  # QQQ's first day with a 200-day average; two months before the dot-com peak
SPY_START = "1994-01-03"
BEARS = {"Dot-com bust (24 Mar 2000 to 9 Oct 2002)": ("2000-03-24", "2002-10-09"),
         "Financial crisis (9 Oct 2007 to 9 Mar 2009)": ("2007-10-09", "2009-03-09"),
         "Covid crash (19 Feb to 23 Mar 2020)": ("2020-02-19", "2020-03-23"),
         "2022 bear market (3 Jan to 12 Oct 2022)": ("2022-01-03", "2022-10-12")}
PERIODS = {"2000-2015 (before the years the rule was chosen on)": ("2000-01-01", "2015-12-31"),
           "2016-2026 (the years tested on real London prices)": ("2016-01-01", "2026-12-31"),
           "2000-2026": ("2000-01-01", "2026-12-31")}


# ------------------------------------------------------------------ data

SPLICE = "2016-01-04"  # Alpaca's first day


def yahoo(symbol: str, max_age_days: int = 4) -> pd.DataFrame:
    """Daily o, h, l, c of a US ETF from Yahoo, adjusted for dividends and splits (scaled by adjclose / close)."""
    y = fetch_yahoo(symbol, max_age_days=max_age_days)
    f = y["adjclose"] / y["close"]
    return pd.DataFrame({k[0]: y[k] * f for k in ("open", "high", "low", "close")}).dropna()


def history(symbol: str, b: dict[str, pd.DataFrame], max_age_days: int = 4) -> pd.DataFrame:
    """Yahoo before 2016 (scaled to meet Alpaca on its first day), Alpaca from 2016."""
    y, a = yahoo(symbol, max_age_days), b[symbol][["o", "h", "l", "c"]]
    d0 = pd.Timestamp(SPLICE)
    k = a.loc[d0, "c"] / y.loc[d0, "c"]
    return pd.concat([y[y.index < d0] * k, a[a.index >= d0]])


def running_cost(sim0: pd.DataFrame, real: pd.DataFrame) -> float:
    """Yearly return a real 3x fund lost against a costless 3x of its index on the days both exist."""
    j = sim0[["c"]].join(real[["c"]], rsuffix="_r", how="inner")
    return float((j["c"].pct_change() - j["c_r"].pct_change()).mean() * 252)


def alpaca_check(hist: dict[str, pd.DataFrame], b: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How closely Yahoo matches Alpaca on the days both have: signals and daily returns."""
    rows = {}
    for s, y in hist.items():
        j = y.join(b[s], how="inner", rsuffix="_a")
        ry, ra = j["c"].pct_change(), j["c_a"].pct_change()
        sy, sa = ibs(j) < 0.1, (j["c_a"] - j["l_a"]) / (j["h_a"] - j["l_a"]) < 0.1
        rows[s] = {"Days compared": len(j), "From": str(j.index[0].date()), "Signals agree": (sy == sa).mean(),
                   "Daily return gap (median)": (ry - ra).abs().median(),
                   "Total return, Yahoo": j["c"].iloc[-1] / j["c"].iloc[0] - 1,
                   "Total return, Alpaca": j["c_a"].iloc[-1] / j["c_a"].iloc[0] - 1}
    return pd.DataFrame(rows).T


def leveraged(t: pd.DataFrame, lev: float = 3.0, cost: float = FEE) -> pd.DataFrame:
    """Open and close of a fund that returns lev x the index each day (reset at every close), less a running cost of
    `cost` a year. Starts at 100."""
    c, o = t["c"].to_numpy(float), t["o"].to_numpy(float)
    drag = cost / 252
    fc, fo = np.empty(len(c)), np.empty(len(c))
    fc[0] = fo[0] = 100.0
    for i in range(1, len(c)):
        fo[i] = fc[i - 1] * max(1 + lev * (o[i] / c[i - 1] - 1), 1e-6)
        fc[i] = fc[i - 1] * max(1 + lev * (c[i] / c[i - 1] - 1) - drag, 1e-6)
    return pd.DataFrame({"o": fo, "c": fc}, index=t.index)


# ------------------------------------------------------------------ rules

def dip_state(t: pd.DataFrame, x: float = 0.1, trend: bool = False) -> pd.Series:
    e = ibs(t) < x
    if trend:
        e &= t["c"] > t["c"].rolling(200).mean()
    return us_state(e, exit_flags(t, "sma", 5), t["c"])


def hold(states: dict[str, pd.Series], funds: dict[str, pd.DataFrame], start: str = START, cost: float = COST,
         rate: float = 0.0, size: dict[str, pd.Series] | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Each fund one place (1 / n of the account): held from the open after a close with state True until the open
    after a close with state False. size: share of the place to buy, decided at the close before (default all).
    Idle cash earns `rate` a year."""
    dates = next(iter(funds.values())).index
    for f in funds.values():
        dates = dates.intersection(f.index)
    names = list(funds)
    O = np.column_stack([funds[n]["o"].reindex(dates).to_numpy() for n in names])
    C = np.column_stack([funds[n]["c"].reindex(dates).to_numpy() for n in names])
    W = np.column_stack([states[n].reindex(dates, fill_value=False).to_numpy(bool) for n in names])
    i0 = int(dates.searchsorted(pd.Timestamp(start))) - 1
    cash = np.full(len(dates), rate / 252)
    S = None
    if size is not None:  # a buy at the open of day i is sized at the close of day i - 1
        S = np.column_stack([size[n].reindex(dates).shift(1).fillna(1.0).to_numpy() for n in names])
    eq, _, tr = run(O, C, W, ~W, np.ones(W.shape), len(names), cost, cash, i0, size=S)
    trades = pd.DataFrame([{"fund": names[j], "entry": dates[a], "exit": dates[b_], "ret": r}
                           for j, a, b_, _, _, r in tr], columns=["fund", "entry", "exit", "ret"])
    return pd.Series(eq, index=dates).dropna(), trades


RULE = "Dip rule, 3x funds (the recommendation)"
QQQ_ONLY = "Dip rule, 3x Nasdaq-100 only (the cleaner data)"
FILTERED = "Dip rule, 3x, buying only above the 200-day average"
HALF = "Dip rule, 3x, half-size buys below the 200-day average"
TREND = "Your 200-day rule: 3x Nasdaq-100 while QQQ is above its 200-day average"
SP = "S&P 500 held (SPY)"


def strategies(hist: dict[str, pd.DataFrame], costs: dict[str, float]) -> dict[str, tuple[pd.Series, pd.DataFrame]]:
    """The rule (3x, 2x, 1x; Nasdaq-100 only; with the 200-day filter), the trend rule and buy-and-hold, from START.
    costs: the 3x funds' running cost a year, by index (2x funds pay half)."""
    f3 = {s: leveraged(hist[s], 3, costs[s]) for s in INDICES}
    f2 = {s: leveraged(hist[s], 2, costs[s] / 2) for s in INDICES}
    f1 = {s: hist[s][["o", "c"]] for s in INDICES}
    dip = {s: dip_state(hist[s]) for s in INDICES}
    dip_up = {s: dip_state(hist[s], trend=True) for s in INDICES}
    trend = {"QQQ": regime(hist["QQQ"]["c"], 200, 1)}
    always = {s: pd.Series(True, index=hist[s].index) for s in INDICES}
    return {
        RULE: hold(dip, f3),
        QQQ_ONLY: hold({"QQQ": dip["QQQ"]}, {"QQQ": f3["QQQ"]}),
        "Dip rule, 2x funds": hold(dip, f2),
        "Dip rule, 1x (QQQ and SPY themselves)": hold(dip, f1),
        FILTERED: hold(dip_up, f3),
        HALF: hold(dip, f3, size={s: pd.Series(np.where(hist[s]["c"] > hist[s]["c"].rolling(200).mean(), 1.0, 0.5),
                                                index=hist[s].index) for s in INDICES}),
        TREND: hold(trend, {"QQQ": f3["QQQ"]}),
        "3x Nasdaq-100 held": hold({"QQQ": always["QQQ"]}, {"QQQ": f3["QQQ"]}, cost=0.0),
        "3x S&P 500 held": hold({"SPY": always["SPY"]}, {"SPY": f3["SPY"]}, cost=0.0),
        "Nasdaq-100 held (QQQ)": hold({"QQQ": always["QQQ"]}, {"QQQ": f1["QQQ"]}, cost=0.0),
        SP: hold({"SPY": always["SPY"]}, {"SPY": f1["SPY"]}, cost=0.0),
    }


# ------------------------------------------------------------------ measures

def part(eq: pd.Series, a: str, z: str) -> pd.Series:
    s = eq[(eq.index >= pd.Timestamp(a)) & (eq.index <= pd.Timestamp(z))]
    prev = eq[eq.index < pd.Timestamp(a)]
    return pd.concat([prev.iloc[-1:], s]) if len(prev) else s


def period_table(runs: dict, periods: dict) -> pd.DataFrame:
    rows = {}
    for name, (eq, _) in runs.items():
        row = {}
        for label, (a, z) in periods.items():
            p = part(eq, a, z)
            row[f"{label}: a year"] = stats(p)["CAGR"]
            row[f"{label}: max drawdown"] = stats(p)["Max drawdown"]
        rows[name] = row
    return pd.DataFrame(rows).T


def bear_table(runs: dict) -> pd.DataFrame:
    rows = {}
    for name, (eq, _) in runs.items():
        rows[name] = {label: part(eq, a, z).iloc[-1] / part(eq, a, z).iloc[0] - 1 for label, (a, z) in BEARS.items()}
    return pd.DataFrame(rows).T


def drawdowns(eq: pd.Series, k: int = 4) -> pd.DataFrame:
    """The k deepest falls that do not overlap: peak, trough, depth, and when the peak was regained."""
    dd = eq / eq.cummax() - 1
    rows, used = [], pd.Series(False, index=eq.index)
    while len(rows) < k:
        d = dd[~used]
        if d.empty or d.min() >= 0:
            break
        trough = d.idxmin()
        peak = eq[:trough].idxmax()
        back = eq[trough:][eq[trough:] >= eq[peak]]
        end = back.index[0] if len(back) else None
        used[(used.index >= peak) & (used.index <= (end or eq.index[-1]))] = True
        rows.append({"Peak": f"{peak:%b %Y}", "Trough": f"{trough:%b %Y}", "Fall": f"{dd[trough]:.0%}",
                     "Back to the peak": f"{end:%b %Y}" if end is not None else "not yet",
                     "Years under water": f"{((end or eq.index[-1]) - peak).days / 365.25:.1f}"})
    return pd.DataFrame(rows)


def index_edge_by_year(hist: dict[str, pd.DataFrame], x: float = 0.1) -> tuple[pd.DataFrame, dict]:
    """Extra 3-day return after a close in the bottom x of the day's range, against any 3 days, by year; and the
    t statistic before and after 2016."""
    rows, tstats = {}, {}
    for s, t in hist.items():
        c = t["c"]
        n3 = c.shift(-3) / c - 1
        sig = ibs(t) < x
        yr = c.index.year
        rows[s] = n3[sig].groupby(yr[sig]).mean() - n3.groupby(yr).mean()
        for label, m in (("before 2016", c.index < "2016-01-01"), ("2016 on", c.index >= "2016-01-01")):
            a, b_ = n3[sig & m].dropna(), n3[~sig & m].dropna()
            tstats[(s, label)] = (a.mean() - b_.mean(), (a.mean() - b_.mean()) / np.sqrt(a.var() / len(a) +
                                                                                           b_.var() / len(b_)), len(a))
    return pd.DataFrame(rows).T, tstats


# ------------------------------------------------------------------ report

def london_versions() -> dict[str, dict]:
    """The full rule, half-size buys below the 200-day average and buying only above it, on the real London funds
    (backtest/isa_tests.py set-up: next London open, 10 bps a side, sterling cash)."""
    from . import isa_tests as T
    env = T.Env()
    out = {}
    for label, mode in ((RULE, "full"), (HALF, "half"), (FILTERED, "above")):
        wo, size = [], []
        for s in T.FUNDS.values():
            t = env.b[s]
            up = t["c"] > t["c"].rolling(200).mean()
            e = T.ibs(t) < 0.1
            if mode == "above":
                e &= up
            wo.append(T.london_wanted(T.us_state(e, T.exit_flags(t, "sma", 5), t["c"]), env.lse))
            pos = np.clip(up.index.searchsorted(env.lse, side="left") - 1, 0, None)
            size.append(np.where(up.to_numpy()[pos], 1.0, 0.5) if mode == "half" else np.ones(len(env.lse)))
        W = np.column_stack(wo)
        eq, tr, inv = T.london_run(env.funds, W, W, env.cash_r, env.lse, env.i0, size_open=np.column_stack(size))
        out[label] = T.measure(eq, tr, inv)
    return out


def run_all() -> dict:
    b = isa.us_bars()
    hist = {s: history(s, b) for s in INDICES}
    real = {"QQQ": b["TQQQ"], "SPY": b["UPRO"]}
    costs = {s: running_cost(leveraged(hist[s], 3, 0.0), real[s]) for s in INDICES}
    runs = strategies(hist, costs)
    runs_fee = strategies(hist, {s: FEE for s in INDICES})
    spy3 = leveraged(hist["SPY"], 3, costs["SPY"])
    spy_runs = {"Dip rule, 3x S&P 500 only": hold({"SPY": dip_state(hist["SPY"])}, {"SPY": spy3}, start=SPY_START),
                "Dip rule, 1x (SPY itself)": hold({"SPY": dip_state(hist["SPY"])}, {"SPY": hist["SPY"][["o", "c"]]},
                                                  start=SPY_START),
                "S&P 500 held (SPY)": hold({"SPY": pd.Series(True, index=hist["SPY"].index)},
                                           {"SPY": hist["SPY"][["o", "c"]]}, start=SPY_START, cost=0.0)}
    edge, tstats = index_edge_by_year(hist)
    ys = {s: yahoo(s) for s in INDICES}
    yahoo_hist = {s: pd.concat([ys[s][ys[s].index < pd.Timestamp(SPLICE)] * (hist[s].loc[SPLICE, "c"] /
                                                                              ys[s].loc[SPLICE, "c"]),
                                ys[s][ys[s].index >= pd.Timestamp(SPLICE)] * (hist[s].loc[SPLICE, "c"] /
                                                                               ys[s].loc[SPLICE, "c"])])
                  for s in INDICES}
    covid = {name: bear_table({RULE: strategies(h, costs)[RULE]})
             for name, h in (("Alpaca from 2016 (used)", hist), ("Yahoo throughout", yahoo_hist))}
    return {"hist": hist, "check": alpaca_check(ys, b), "bad_closes": bad_closes(ys, b), "covid": covid,
            "runs": runs, "runs_fee": runs_fee, "costs": costs, "spy_runs": spy_runs, "edge": edge, "tstats": tstats,
            "london": london_versions()}


def bad_closes(ys: dict[str, pd.DataFrame], b: dict[str, pd.DataFrame], tol: float = 0.003) -> dict:
    """Days on which Yahoo's close is more than `tol` away from Alpaca's official close (after removing the slow drift
    from different dividend adjustments)."""
    out = {}
    for s, y in ys.items():
        j = y.join(b[s], rsuffix="_a", how="inner")
        rel = j["c"] / j["c_a"]
        rel = rel / rel.rolling(61, center=True, min_periods=10).median() - 1
        bad = rel[rel.abs() > tol]
        out[s] = {"days": len(j), "bad": len(bad), "dates": [str(d.date()) for d in bad.abs().sort_values().index[::-1]]}
    return out


def _pct(df: pd.DataFrame, digits: int = 1) -> pd.DataFrame:
    return df.map(lambda v: "" if pd.isna(v) else f"{v:+.{digits}%}")


SHORT = {"2000-2015 (before the years the rule was chosen on)": "2000-15",
         "2016-2026 (the years tested on real London prices)": "2016-26", "2000-2026": "2000-26"}


def _periods(runs: dict) -> pd.DataFrame:
    t = period_table(runs, PERIODS)
    cols = {}
    for long, short in SHORT.items():
        cols[f"{short} a year"] = t[f"{long}: a year"].map(lambda v: f"{v:+.1%}")
        cols[f"{short} max drawdown"] = t[f"{long}: max drawdown"].map(lambda v: f"{v:.0%}")
    return pd.DataFrame(cols)


def report(out: dict) -> str:
    runs, runs_fee, spy_runs, costs = out["runs"], out["runs_fee"], out["spy_runs"], out["costs"]
    eq, tr = runs[RULE]
    sp = runs[SP][0]
    first, last = eq.index[1].date(), eq.index[-1].date()
    t = period_table(runs, PERIODS)
    bears = bear_table(runs)
    yrs = pd.DataFrame({name: yearly_returns(runs[name][0]) for name in
                        (RULE, HALF, FILTERED, QQQ_ONLY, TREND, SP, "Nasdaq-100 held (QQQ)")})
    yrs.columns = ["Dip rule 3x", "Half size below 200-day", "No buys below 200-day", "Nasdaq-100 only",
                   "Your 200-day rule (3x)", "S&P 500", "Nasdaq-100"]
    yrs = yrs[yrs.index >= 2000]
    early, sp_early = part(eq, "2000-01-01", "2015-12-31"), part(sp, "2000-01-01", "2015-12-31")
    late = part(eq, "2016-01-01", "2026-12-31")
    tr_early = tr[tr["entry"] < pd.Timestamp("2016-01-01")]
    check = out["check"].copy()
    for c in ("Signals agree", "Total return, Yahoo", "Total return, Alpaca"):
        check[c] = check[c].map(lambda v: f"{v:.2%}")
    check["Daily return gap (median)"] = check["Daily return gap (median)"].map(lambda v: f"{v:.4%}")
    check["Days compared"] = check["Days compared"].astype(int)
    bc = out["bad_closes"]
    cov = {k: v.loc[RULE, list(BEARS)[2]] for k, v in out["covid"].items()}
    edge, ts = out["edge"], out["tstats"]
    fee_t = period_table(runs_fee, {"2000-2026": ("2000-01-01", "2026-12-31")})
    cost_tab = pd.DataFrame({"Real funds' running cost: a year": t["2000-2026: a year"],
                             "0.75% fee only: a year": fee_t["2000-2026: a year"],
                             "Real funds' running cost: max drawdown": t["2000-2026: max drawdown"],
                             "0.75% fee only: max drawdown": fee_t["2000-2026: max drawdown"]})
    spy_tab = period_table(spy_runs, {"1994-2026": ("1994-01-01", "2026-12-31"),
                                      "1994-2015": ("1994-01-01", "2015-12-31")})
    dd = drawdowns(eq)
    dot, gfc = list(BEARS)[0], list(BEARS)[1]
    ed = {k: v for k, v in ts.items()}
    lines = [
        f"# The ISA dip rule over {first.year}-{last.year} ({first} to {last})", "",
        f"Generated by `python -m backtest.long_history` on {pd.Timestamp.today().date()}. The rule as in "
        "backtest/isa.py (IBS < 0.1 on QQQ / SPY, sell above the 5-day average, half in each 3x fund) on daily prices "
        "since 1999 (Yahoo before 2016, approved 2026-11-22; Alpaca from 2016), with simulated 3x funds that pay what "
        f"the real TQQQ and UPRO cost ({costs['QQQ']:.1%} and {costs['SPY']:.1%} a year while held), in dollars, at "
        "the next US open, 10 bps a side, no interest on cash. See the module docstring.", "",
        "## Summary", "",
        f"- **2000-2015, before the years the rule was chosen on:** {stats(early)['CAGR']:.1%} a year against "
        f"{stats(sp_early)['CAGR']:.1%} for the S&P 500 ({(tr_early['ret'] > 0).mean():.0%} of {len(tr_early)} trades "
        f"won), with a max drawdown of {stats(early)['Max drawdown']:.0%} ({stats(sp_early)['Max drawdown']:.0%} for the "
        f"S&P 500). 2016-2026: {stats(late)['CAGR']:.1%} a year.",
        f"- **The edge was there before 2016 too:** QQQ's extra 3-day return after a close at the day's low was "
        f"{ed[('QQQ', 'before 2016')][0]:+.2%} before 2016 (t = {ed[('QQQ', 'before 2016')][1]:.1f}, "
        f"{ed[('QQQ', 'before 2016')][2]} signals) and {ed[('QQQ', '2016 on')][0]:+.2%} since; SPY's "
        f"{ed[('SPY', 'before 2016')][0]:+.2%} (t = {ed[('SPY', 'before 2016')][1]:.1f}) and "
        f"{ed[('SPY', '2016 on')][0]:+.2%}.",
        f"- **Bear markets:** the dot-com bust {bears.loc[RULE, dot]:+.0%} (S&P 500 {bears.loc[SP, dot]:+.0%}, your "
        f"200-day rule in a 3x fund {bears.loc[TREND, dot]:+.0%}, a 3x Nasdaq-100 fund held "
        f"{bears.loc['3x Nasdaq-100 held', dot]:+.0%}); the financial crisis {bears.loc[RULE, gfc]:+.0%} (S&P 500 "
        f"{bears.loc[SP, gfc]:+.0%}, your rule {bears.loc[TREND, gfc]:+.0%}).",
        f"- **The price: deep drawdowns.** Worst fall {stats(eq)['Max drawdown']:.0%} ({dd.iloc[0]['Peak']} to "
        f"{dd.iloc[0]['Trough']}, {dd.iloc[0]['Years under water']} years to recover), far deeper than the "
        "2016-26 London test showed. Buying less, or nothing, while the index is below its 200-day average cuts "
        "that:", "",
        md(_versions(t, bears, out["london"]), "Version"), "",
        "## Results by period", "",
        md(_periods(runs), "Rule"), "",
        "## The bear markets (from the index's peak to its low)", "",
        md(_pct(bears, 0), "Rule"), "",
        "## Year by year", "",
        md(_pct(yrs.rename(index=str), 0), "Year"), "",
        "## The rule's deepest falls", "",
        md(dd.set_index("Peak"), "Peak"), "",
        "## The edge in the index, year by year", "",
        "Extra 3-day return after a close in the bottom 10% of the day's range, against any 3 days that year:", "",
        md(_pct(edge.rename(columns=str), 2).T, "Year"), "",
        f"Positive in {int((edge.loc['QQQ'].dropna() > 0).sum())} of {int(edge.loc['QQQ'].notna().sum())} years for QQQ "
        f"and {int((edge.loc['SPY'].dropna() > 0).sum())} of {int(edge.loc['SPY'].notna().sum())} for SPY.", "",
        "## The S&P 500 alone from 1994", "",
        md(_pct(spy_tab), "Rule"), "",
        "## Fund costs", "",
        "The simulated 3x funds pay what the real ones cost (main runs) or only the 0.75% fee (the funds' borrowing "
        "free, which flatters holding 3x funds most):", "",
        md(_pct(cost_tab), "Rule"), "",
        "## The data", "",
        "Yahoo against Alpaca on the days both have (2016-2026):", "",
        md(check, "Index"), "",
        f"Yahoo's QQQ closes are within 0.3% of Alpaca's official closes on all but {bc['QQQ']['bad']} of "
        f"{bc['QQQ']['days']} days. Its SPY closes are off by more on {bc['SPY']['bad']} days ("
        f"{', '.join(bc['SPY']['dates'][:8])}): extreme days, when SPY's closing auction moved away from its last "
        f"trades. On Yahoo's prices the rule's Covid-crash result would be {cov['Yahoo throughout']:+.0%} instead of "
        f"{cov['Alpaca from 2016 (used)']:+.0%}: two wrong closes made false signals. So the runs use Alpaca from 2016; "
        "before 2016 such errors may sit on SPY's crash days, which the Nasdaq-100-only rows avoid. Prices before 2001 "
        "are in fractions of a dollar, as traded then.", "",
        "## Verdict", "",
        _verdict(runs, bears, early, sp_early, t, out["london"])]
    return "\n".join(lines)


def _versions(t: pd.DataFrame, bears: pd.DataFrame, london: dict) -> pd.DataFrame:
    dot = next(iter(BEARS))
    rows = {}
    for label, short in ((RULE, "Full size always (the rule)"), (HALF, "Half-size buys below the 200-day average"),
                         (FILTERED, "No buys below the 200-day average"), (SP, "S&P 500 held")):
        lon = london.get(label)
        rows[short] = {"2000-26 a year": f"{t.loc[label, '2000-2026: a year']:+.1%}",
                       "2000-26 worst fall": f"{t.loc[label, '2000-2026: max drawdown']:.0%}",
                       "Dot-com bust": f"{bears.loc[label, dot]:+.0%}",
                       "London 2017-26 a year": f"{lon['CAGR']:+.1%}" if lon else "",
                       "London 2017-26 worst fall": f"{lon['Max drawdown']:.0%}" if lon else ""}
    return pd.DataFrame(rows).T


def _verdict(runs, bears, early, sp_early, t, london) -> str:
    dot = next(iter(BEARS))
    return (f"The rule worked before the years it was chosen on: {stats(early)['CAGR']:.1%} a year in 2000-2015 "
            f"against {stats(sp_early)['CAGR']:.1%} for the S&P 500, and the index's bounce after closes at the low was "
            f"as strong then as now. It came through the dot-com bust ({bears.loc[RULE, dot]:+.0%}) where holding a 3x "
            f"fund was wiped out and the 200-day rule in a 3x fund lost {-bears.loc[TREND, dot]:.0%}. But it fell "
            f"{-stats(runs[RULE][0])['Max drawdown']:.0%} at its worst and took years to recover, much deeper than the "
            "London test since 2016 showed. Buying half the usual amount while the index is below its 200-day average "
            f"keeps most of the return ({t.loc[HALF, '2000-2026: a year']:.1%} a year since 2000, "
            f"{london[HALF]['CAGR']:.1%} on the London funds since 2017) and cuts the worst fall to "
            f"{-t.loc[HALF, '2000-2026: max drawdown']:.0%} (London: {-london[HALF]['Max drawdown']:.0%}). It is a "
            "choice of risk, not a better rule: it was chosen after seeing the long history, so its numbers are not an "
            "independent test.")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    out = run_all()
    text = report(out)
    (REPO_ROOT / "backtest" / "LONG_HISTORY.md").write_text(text + "\n")
    out_dir = paths().root / "backtests" / "long_history"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: v[0] for k, v in out["runs"].items()}).to_csv(out_dir / "curves.csv")
    print(text)


if __name__ == "__main__":
    main()

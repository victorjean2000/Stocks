"""The momentum paper strategy with a market filter: hold the stocks only while SPY or QQQ is above a moving average.

Base: the live momentum strategy (BASE in backtest/momentum_refine.py with the top-30 buffer, as the paper account has
traded since 2026-11-13): at each month end, the 10 stocks among the 500 largest with the best 6-1 month return, kept
while they rank in the top 30, 10% each, traded at the next open, 10 bps a side, T-bill interest on idle cash.

Filters (momentum_trend Params: market_filter, market_index, market_sma, market_days, market_check):
- checked daily: sell everything at the next open after the index closes below its N-day average 3 days in a row,
  and buy the best-ranked stocks back at the open after 3 closes above it;
- checked monthly: on each monthly decision day, hold the portfolio for the month if the index closes above its
  average that day, otherwise T-bills until the next decision.
Indices SPY and QQQ (Alpaca), simple averages of 50, 100, 150 and 200 days, plus both / either index above the 200-day
average, a 1-close switch, and waiting for the next monthly decision to buy back.

As in backtest/momentum_refine.py, every variant runs with the monthly decision 0, 2, ..., 18 sessions after month end
(the result depends on the day) and the tables show the average of the ten; settings are judged on 2017-21 and
checked on 2022-26.

Run: python -m backtest.momentum_filter (about 2 minutes on 4 cores). Writes backtest/MOMENTUM_FILTER.md.
"""
from __future__ import annotations

import datetime as dt
import logging
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from stockdata.config import REPO_ROOT, paths

from .momentum_refine import BASE, FIRST, OFFSETS, SPLIT
from .momentum_trend import Market, Params, Result, load_market, md, simulate, spy, stats, yearly

log = logging.getLogger(__name__)

LIVE = BASE | dict(keep_rank=30)  # the paper account's rules since 2026-11-13
BASE_LABEL = "Live strategy, no filter"
SMAS = (50, 100, 150, 200)
FALLS = {"Late 2018 (20 Sep to 24 Dec 2018)": ("2018-09-20", "2018-12-24"),
         "Covid crash (19 Feb to 23 Mar 2020)": ("2020-02-19", "2020-03-23"),
         "2022 bear market (3 Jan to 12 Oct 2022)": ("2022-01-03", "2022-10-12"),
         "2025 tariff fall (19 Feb to 8 Apr 2025)": ("2025-02-19", "2025-04-08"),
         "Autumn 2026 (6 Oct to 20 Nov 2026)": ("2026-10-06", "2026-11-20")}


def variants() -> list[tuple[str, str, dict]]:
    """(group, label, Params changes from LIVE)."""
    v = [("Base", BASE_LABEL, {})]
    for idx in ("SPY", "QQQ"):
        v += [(f"{idx}, checked daily", f"{idx} above its {n}-day average (3 closes to switch), back in at once",
               dict(market_filter=True, market_index=idx, market_sma=n, refill="daily")) for n in SMAS]
    for idx in ("SPY", "QQQ"):
        v += [(f"{idx}, checked monthly", f"{idx} above its {n}-day average on the monthly decision day",
               dict(market_filter=True, market_index=idx, market_sma=n, market_check="monthly")) for n in SMAS]
    v += [("Other filters", "SPY and QQQ both above their 200-day averages (daily, 3 closes)",
           dict(market_filter=True, market_index="SPY+QQQ", refill="daily")),
          ("Other filters", "SPY or QQQ above its 200-day average (daily, 3 closes)",
           dict(market_filter=True, market_index="SPY|QQQ", refill="daily")),
          ("Other filters", "SPY above its 200-day average, 1 close to switch, back in at once",
           dict(market_filter=True, market_days=1, refill="daily")),
          ("Other filters", "SPY above its 200-day average (3 closes), back in at the next monthly decision",
           dict(market_filter=True, refill="monthly"))]
    return v


def fall(eq: pd.Series, a: str, z: str) -> float:
    """Return from the close before `a` to the last close on or before `z`."""
    before, upto = eq[eq.index < pd.Timestamp(a)], eq[eq.index <= pd.Timestamp(z)]
    return float(upto.iloc[-1] / before.iloc[-1] - 1) if len(before) and len(upto) else np.nan


def measure(r: Result) -> dict:
    eq = r.equity
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    s = stats(eq)
    exits = r.trades.loc[r.trades["reason"] == "market", "exit"].nunique()
    return {"CAGR": s["CAGR"], "2017-21": stats(eq[eq.index <= SPLIT])["CAGR"],
            "2022-26": stats(eq[eq.index >= SPLIT])["CAGR"], "Sharpe": s["Sharpe (rf=0)"],
            "Max drawdown": s["Max drawdown"], "Worst year": yearly(eq).min(), "In the market": r.invested.mean(),
            "Exits a year": exits / years, "Turnover": r.turnover} | {k: fall(eq, a, z) for k, (a, z) in FALLS.items()}


_M: Market | None = None  # shared with forked worker processes


def _one(task: tuple) -> tuple[dict, dict, pd.Series | None]:
    group, label, kw, off, start = task
    r = simulate(_M, Params(name=label, start=start, offset=off, **(LIVE | kw)))
    ys = {"variant": label, "offset": off} | {int(y): v for y, v in yearly(r.equity).items()}
    return {"group": group, "variant": label, "offset": off} | measure(r), ys, \
        ((r.equity, r.invested) if off == 0 else None)


def warm(m: Market) -> None:
    """Fill the indicator caches before forking, so the workers share them."""
    m.st(10, 3.0), m.sma(50), m.sma(200), m.score("r6_1")
    for n in SMAS:
        for idx in ("SPY", "QQQ"):
            m.risk_on(n, 3, idx), m.risk_on(n, 1, idx)
    m.risk_on(200, 3, "SPY+QQQ"), m.risk_on(200, 3, "SPY|QQQ")


def run_all(workers: int | None = None) -> dict:
    global _M
    t0 = time.time()
    m = load_market()
    warm(m)
    _M = m
    dates = m.close.index
    first = int(dates.searchsorted(pd.Timestamp(FIRST)))
    tasks = [(g, label, kw, off, str(dates[first + off].date())) for g, label, kw in variants() for off in OFFSETS]
    workers = workers or min(4, os.cpu_count() or 1)
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            done = pool.map(_one, tasks, chunksize=2)
    else:
        done = [_one(t) for t in tasks]
    runs = pd.DataFrame([row for row, _, _ in done])
    years = pd.DataFrame([ys for _, ys, _ in done])
    at0 = {row["variant"]: x for row, _, x in done if x is not None}
    curves = {k: eq for k, (eq, _) in at0.items()}
    invested = {k: inv for k, (_, inv) in at0.items()}
    base = runs[runs["group"] == "Base"].set_index("offset")
    for col in ("CAGR", "2017-21", "2022-26"):
        runs[f"{col} vs base"] = runs[col] - runs["offset"].map(base[col])
    sp = spy().reindex(curves[BASE_LABEL].index).ffill()
    out_dir = paths().root / "backtests" / "momentum_filter"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_dir / "runs.csv", index=False)
    pd.DataFrame(curves).to_csv(out_dir / "equity_curves_offset0.csv")
    off_share = {label: 1 - runs.loc[runs["variant"] == label, "In the market"].mean() for _, label, _ in variants()}
    log.info("%d runs in %.0fs", len(tasks), time.time() - t0)
    return {"runs": runs, "years": years, "spy": sp, "seconds": time.time() - t0, "off": off_share,
            "last": dates[-1], "curves": curves, "invested": invested}


# ------------------------------------------------------------------ report

NUM = ["CAGR", "2017-21", "2022-26", "CAGR vs base", "2017-21 vs base", "2022-26 vs base", "Sharpe", "Max drawdown",
       "Worst year", "In the market", "Exits a year", "Turnover", *FALLS]


def averages(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["group", "variant"], sort=False)
    avg = g[NUM].mean()
    avg["Timings better"] = g["CAGR vs base"].apply(lambda s: f"{int((s > 1e-9).sum())} of {len(OFFSETS)}")
    avg["Worst timing"] = g["CAGR"].min()
    return avg.reset_index()


def fmt(tbl: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = tbl[cols].copy().astype(object)
    for c in cols:
        if c == "Sharpe":
            out[c] = [f"{v:.2f}" for v in tbl[c]]
        elif c in ("Exits a year", "Turnover"):
            out[c] = [f"{v:.1f}" for v in tbl[c]]
        elif c.endswith("vs base") or c in FALLS:
            out[c] = [f"{v:+.1%}" for v in tbl[c]]
        elif c in ("CAGR", "2017-21", "2022-26", "Max drawdown", "Worst year", "In the market", "Worst timing"):
            out[c] = [f"{v:.1%}" for v in tbl[c]]
    return out


def _late(out: dict, label: str) -> str:
    """How far the base (offset 0) had already fallen in its deepest fall when the filter `label` first went to cash."""
    eq, f_eq = out["curves"][BASE_LABEL], out["curves"][label]
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq[:trough].idxmax()
    inv = out["invested"][label][peak:trough]
    cash = inv[inv < 0.05]
    text = (f"The base's deepest fall ({peak:%b %Y} to {trough:%b %Y}, {dd[trough]:.0%} at the paper account's "
            f"timing) came while the index fell far less (momentum stocks fell first), ")
    if cash.empty:
        return text + "and the best filter never went to cash in it."
    first = cash.index[0]
    return text + (f"so the best filter only went to cash on {first:%d %b %Y}, when the base was already "
                   f"{1 - eq[first] / eq[peak]:.0%} down; it fell {1 - f_eq[trough] / f_eq[peak]:.0%} over that fall.")


def _missed(out: dict, label: str) -> str:
    """What the base made (offset 0) on the days the filter `label` sat in cash, and on the other days."""
    r = out["curves"][BASE_LABEL].pct_change().dropna()
    cash = out["invested"][label].reindex(r.index).shift(1).fillna(1.0) < 0.05  # out of the market that whole day
    yearly_rate = lambda x: (1 + x).prod() ** (252 / max(len(x), 1)) - 1  # noqa: E731
    return (f"on the {cash.mean():.0%} of days the best filter sat in cash, the base strategy made "
            f"{yearly_rate(r[cash]):+.0%} a year (it was {yearly_rate(r[~cash]):+.0%} a year on the other days): the "
            "filter sells after a fall and buys back after the rebound has started, and momentum's best months "
            "often come in those rebounds (2020, 2023, 2025).")


def report(out: dict) -> str:
    runs, years, sp = out["runs"], out["years"], out["spy"]
    avg = averages(runs)
    base = avg[avg["group"] == "Base"].iloc[0]
    filt = avg[avg["group"] != "Base"]
    best = filt.loc[filt["2017-21"].idxmax()]  # picked on the tuning years
    calm = filt.loc[filt["Sharpe"].idxmax()]
    beat_both = filt[(filt["2017-21 vs base"] > 0) & (filt["2022-26 vs base"] > 0)]
    sp_cagr = stats(sp)["CAGR"]
    yrs = years.groupby("variant").mean(numeric_only=True).drop(columns="offset")
    ytab = pd.DataFrame({"Base (no filter)": yrs.loc[BASE_LABEL], f"Best on 2017-21: {best['variant']}":
                         yrs.loc[best["variant"]]})
    if calm["variant"] != best["variant"]:
        ytab[f"Best Sharpe: {calm['variant']}"] = yrs.loc[calm["variant"]]
    spy_y = yearly(sp)
    ytab["S&P 500 (SPY)"] = spy_y.reindex(ytab.index)
    ytab = ytab.map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
    cols = ["CAGR", "2017-21", "2022-26", "CAGR vs base", "Sharpe", "Max drawdown", "Worst year", "In the market",
            "Exits a year", "Timings better"]
    falls = fmt(avg.set_index("variant"), list(FALLS))
    lines = [
        f"# Momentum with a market filter ({FIRST} to {out['last'].date()})", "",
        f"Generated by `python -m backtest.momentum_filter` on {dt.date.today()} in {out['seconds']:.0f} s. The live "
        "momentum strategy (10 of the 500 largest by 6-1 month return, kept while in the top 30, 10% each, monthly, "
        "10 bps a side, T-bill interest on cash) with and without a market filter that holds the stocks only while "
        "SPY or QQQ is above a moving average. Every row is the average of 10 runs with the monthly decision 0, 2, "
        "..., 18 sessions after month end; settings are picked on 2017-21 and checked on 2022-26.", "",
        "## Summary", "",
        f"- **Base (no filter):** {base['CAGR']:.1%} a year ({base['2017-21']:.1%} in 2017-21, {base['2022-26']:.1%} "
        f"in 2022-26), Sharpe {base['Sharpe']:.2f}, max drawdown {base['Max drawdown']:.0%}. S&P 500: {sp_cagr:.1%}.",
        f"- **Best filter on 2017-21:** {best['variant']}: {best['CAGR']:.1%} a year ({best['2017-21 vs base']:+.1%} "
        f"against the base in 2017-21, {best['2022-26 vs base']:+.1%} in 2022-26), Sharpe {best['Sharpe']:.2f}, max "
        f"drawdown {best['Max drawdown']:.0%}, in the market {best['In the market']:.0%} of the time; it beat the base "
        f"at {best['Timings better']} timings.",
        f"- **Best Sharpe (return for the risk):** {calm['variant']}: {calm['CAGR']:.1%} a year, Sharpe "
        f"{calm['Sharpe']:.2f}, max drawdown {calm['Max drawdown']:.0%}.",
        f"- {len(beat_both)} of {len(filt)} filters beat the base in both 2017-21 and 2022-26"
        + (f" ({'; '.join(beat_both['variant'])})." if len(beat_both) else "."),
        f"- **Max drawdown:** the filters' worst falls ranged from {filt['Max drawdown'].max():.0%} to "
        f"{filt['Max drawdown'].min():.0%}, against {base['Max drawdown']:.0%} for the base. {_late(out, best['variant'])}",
        f"- **Why they cost return:** {_missed(out, best['variant'])}", "",
        "## All filters", "",
        md(fmt(avg.set_index("variant"), cols), "Filter"), "",
        "## The falls (return over each period, average of the timings)", "",
        md(falls, "Filter"), "",
        "## Year by year (average of the timings)", "",
        md(ytab.rename(index=str), "Year"), "",
        "## Notes", "",
        "- Only 2017-26 can be tested (the point-in-time stock data starts in 2016): one long bear market (2022) and "
        "several fast falls. A filter earns its keep in long bear markets like 2000-02 and 2008, which are not in the "
        "sample; in fast falls it sells near the bottom and buys back higher.",
        "- Daily filters trade the whole portfolio at each switch (costs included); in an ISA or a pension there is "
        "no tax on that, in a taxable account there would be.",
        "- Cash earns T-bill interest (BIL) while the filter is out."]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    out = run_all()
    text = report(out)
    (REPO_ROOT / "backtest" / "MOMENTUM_FILTER.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

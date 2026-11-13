"""Refining the paper momentum strategy, and testing whether taking profits helps.

Base: what the momentum paper account trades (trading/targets.py, backtest/strategies.py). At the last close of
each month, the 10 stocks among the 500 largest with the best 6-1 month return, 10% each, traded at the next open;
10 bps a side; idle cash earns T-bills (BIL). Each variant changes one thing (Params in backtest/momentum_trend.py).

Two guards against fooling ourselves:
- Timing luck: with 10 stocks the result depends on the day of the month the portfolio trades. Every variant runs
  with the monthly decision 0, 2, ..., 18 sessions after month end, and the tables show the average of the ten.
- Tuning versus checking: a rule's best level is picked on 2017-21 only, then checked on 2022-26.

Run: python -m backtest.momentum_refine (about 3 minutes on 4 cores). Writes backtest/MOMENTUM_REFINE.md, plus every
variant x timing and the offset-0 equity curves in data/backtests/momentum_refine/.
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

from .momentum_trend import Market, Params, Result, load_market, md, simulate, spy, stats, yearly

log = logging.getLogger(__name__)

FIRST = "2017-01-31"  # first decision at offset 0; trading starts the next morning, as in the strategy search
SPLIT = pd.Timestamp("2021-12-31")  # last day of the tuning years
OFFSETS = tuple(range(0, 20, 2))  # sessions after month end; 0 is the paper account's timing
BASE = dict(pool=500, momentum="r6_1", n_positions=10, sector_cap=10, golden_cross=False, trend_entry=False,
            trend_exit=False, monthly=True, refill="monthly", reweight=True, cash_yield=True)
RULE_EXITS = ("take_profit", "trail_stop", "stop_loss", "extension")
TAKE_PROFIT = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
TRAIL = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
NEXT = "next-best stock the next morning"
WAIT = "cash until the month end"
PROFIT_GROUPS = ["Take profit: sell all", "Take profit: sell half", "Trailing stop", "Sell when stretched",
                 "Stop loss", "Monthly trim"]
KEEP = (15, 20, 25, 30, 40, 50, 75, 100)
BUFFER = "Keep a holding while it ranks in the top 30"
REFINED = "Keep while in the top 30, no trim back to 10%"


def variants() -> list[tuple[str, str, dict]]:
    """(group, label, Params changes from BASE)."""
    v = [("Base", "Paper strategy: monthly top 10, back to 10% each at every rebalance", {})]
    v += [("Take profit: sell all", f"At +{x:.0%}, {WAIT}", dict(take_profit=x)) for x in TAKE_PROFIT]
    v += [("Take profit: sell all", f"At +{x:.0%}, {NEXT}", dict(take_profit=x, refill="daily")) for x in TAKE_PROFIT]
    v += [("Take profit: sell all", f"At +{x:.0%}, {NEXT}, sold stock sits out 3 months",
           dict(take_profit=x, refill="daily", rebuy_wait=3)) for x in (0.5, 1.0)]
    v += [("Take profit: sell half", f"Half at +{x:.0%} (once)", dict(take_profit=x, take_profit_part=0.5))
          for x in (0.25, 0.5, 1.0)]
    v += [("Take profit: sell half", f"Half at every further +{x:.0%}",
           dict(take_profit=x, take_profit_part=0.5, take_profit_ladder=True)) for x in (0.25, 0.5)]
    v += [("Take profit: sell half", "Half at +50% (once), no monthly trim back to 10%",
           dict(take_profit=0.5, take_profit_part=0.5, reweight=False))]
    v += [("Trailing stop", f"{x:.0%} below the highest close, {WAIT}", dict(trail_stop=x)) for x in TRAIL]
    v += [("Trailing stop", f"{x:.0%} below the highest close, {NEXT}", dict(trail_stop=x, refill="daily"))
          for x in TRAIL]
    v += [("Sell when stretched", f"{x:.0%} above its 50-day average (never bought above it), {NEXT}",
           dict(extension_exit=x, refill="daily")) for x in (0.2, 0.3, 0.4, 0.5)]
    v += [("Stop loss", f"{x:.0%} below the purchase price, {NEXT}", dict(stop_loss=x, refill="daily"))
          for x in (0.1, 0.15, 0.2, 0.3)]
    v += [("Monthly trim", "No trim back to 10% at rebalances (winners keep their weight)", dict(reweight=False))]
    # the strategy's other settings, one at a time
    v += [("Holdings", f"{n} stocks", dict(n_positions=n)) for n in (5, 8, 12, 15, 20)]
    v += [("Momentum measure", label, dict(momentum=k)) for k, label in (
        ("r12_1", "12-1 month return"), ("composite", "Average rank of 3, 6 and 12-1 month returns"),
        ("vol_adjusted", "Same, each return divided by volatility"))]
    v += [("Stock pool", f"Largest {n}", dict(pool=n)) for n in (200, 300, 750, 1000)]
    v += [("Stock pool", "All $3bn+ stocks", dict(pool=None))]
    v += [("Sell only below rank", f"Keep a holding while it ranks in the top {k}", dict(keep_rank=k))
          for k in KEEP]
    v += [("Sell only below rank, no trim", f"Keep while in the top {k}, no trim back to 10%",
           dict(keep_rank=k, reweight=False)) for k in (20, 30, 50)]
    v += [("SuperTrend exit", f"Sell after {d} bearish SuperTrend(10,{x:g}) closes, {NEXT}",
           dict(trend_exit=True, confirm_days=d, st_mult=x, refill="daily"))
          for d, x in ((3, 3.0), (10, 3.0), (10, 4.0))]
    v += [("Sector cap", f"At most {k} per sector", dict(sector_cap=k)) for k in (3, 4)]
    v += [("Market filter", "Cash while SPY is below its 200-day average (3 closes), back in at once",
           dict(market_filter=True, refill="daily"))]
    v += [("Volatility", "Never buy the 20% most volatile stocks", dict(max_vol_pct=0.8))]
    return v


def measure(r: Result) -> dict:
    eq = r.equity
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    s = stats(eq)
    closed = r.trades.dropna(subset=["exit"])
    return {"CAGR": s["CAGR"], "2017-21": stats(eq[eq.index <= SPLIT])["CAGR"],
            "2022-26": stats(eq[eq.index >= SPLIT])["CAGR"], "Sharpe": s["Sharpe (rf=0)"],
            "Max drawdown": s["Max drawdown"], "Worst year": yearly(eq).min(),
            "Rule sales per year": (closed["reason"].isin(RULE_EXITS).sum() + r.trims) / years,
            "Turnover": r.turnover, "Largest holding": r.top_weight.max()}


_M: Market | None = None  # the market, shared with forked worker processes


def _one(task: tuple) -> tuple[dict, dict, Result | None]:
    group, label, kw, off, start = task
    r = simulate(_M, Params(name=label, start=start, offset=off, **(BASE | kw)))
    ys = {"variant": label, "offset": off} | {int(y): v for y, v in yearly(r.equity).items()}
    return {"group": group, "variant": label, "offset": off} | measure(r), ys, (r if off == 0 else None)


def warm(m: Market) -> None:
    """Fill the market's indicator caches once, before forking, so the workers share them."""
    m.st(10, 3.0), m.st(10, 4.0), m.sma(50), m.sma(200), m.vol_rank(), m.risk_on(200, 3)
    for k in ("r6_1", "r12_1", "composite", "vol_adjusted"):
        m.score(k)


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
    rows = [row for row, _, _ in done]
    years = pd.DataFrame([ys for _, ys, _ in done])
    at0 = {row["variant"]: r for row, _, r in done if r is not None}
    curves = {k: r.equity for k, r in at0.items()}
    base0 = at0[variants()[0][1]]
    log.info("%d runs in %.0fs", len(tasks), time.time() - t0)
    runs = pd.DataFrame(rows)
    base = runs[runs["group"] == "Base"].set_index("offset")
    for col in ("CAGR", "2017-21", "2022-26"):
        runs[f"{col} vs base"] = runs[col] - runs["offset"].map(base[col])
    sp = spy().reindex(curves[variants()[0][1]].index).ffill()
    out_dir = paths().root / "backtests" / "momentum_refine"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_dir / "runs.csv", index=False)
    years.to_csv(out_dir / "calendar_years.csv", index=False)
    pd.DataFrame(curves).to_csv(out_dir / "equity_curves_offset0.csv")
    return {"market": m, "runs": runs, "years": years, "base0": base0, "spy": sp, "seconds": time.time() - t0}


def averages(runs: pd.DataFrame) -> pd.DataFrame:
    """Each variant averaged over the timings, plus its offset-0 result and how many timings beat the base."""
    num = ["CAGR", "2017-21", "2022-26", "CAGR vs base", "2017-21 vs base", "2022-26 vs base", "Sharpe",
           "Max drawdown", "Worst year", "Rule sales per year", "Turnover"]
    g = runs.groupby(["group", "variant"], sort=False)
    avg = g[num].mean()
    avg["Worst timing"] = g["CAGR"].min()
    avg["Best timing"] = g["CAGR"].max()
    avg["Largest holding"] = g["Largest holding"].max()
    avg["Offset 0"] = g.apply(lambda d: d.loc[d["offset"] == 0, "CAGR"].iloc[0], include_groups=False)
    avg["Timings n"] = g["CAGR vs base"].apply(lambda s: int((s > 1e-9).sum()))
    avg["Timings better"] = [f"{n} of {len(OFFSETS)}" for n in avg["Timings n"]]
    return avg.reset_index()


def after_level(m: Market, r: Result, levels=(0.2, 0.5, 1.0)) -> pd.DataFrame:
    """For base positions that closed at least x above their purchase price: what they did next, from the
    following open to the price the strategy actually sold at (or the last close if still held)."""
    cols, dates = m.close.columns, m.close.index
    O, C = m.open.to_numpy(), m.close.ffill().to_numpy()
    rows = {}
    for x in levels:
        further = []
        for t in r.trades.itertuples(index=False):
            j, a = cols.get_loc(t.ticker), dates.get_loc(t.entry)
            b = dates.get_loc(t.exit) if pd.notna(t.exit) else len(dates)
            hit = np.flatnonzero(C[a:b, j] >= t.entry_px * (1 + x))
            if hit.size and a + hit[0] + 1 < b:
                k = a + hit[0] + 1
                further.append(t.exit_px / O[k, j] - 1)
        f = np.array(further)
        rows[f"+{x:.0%}"] = {"Positions that got there": f"{len(f)} of {len(r.trades)}",
                             "Then rose further": f"{np.mean(f > 0):.0%}",
                             "Median further move": f"{np.median(f):+.0%}",
                             "Average further move": f"{np.mean(f):+.0%}"}
    return pd.DataFrame(rows).T


def fmt(tbl: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = tbl[cols].copy().astype(object)
    for c in cols:
        if c in ("Sharpe",):
            out[c] = [f"{v:.2f}" for v in tbl[c]]
        elif c in ("Rule sales per year", "Turnover"):
            out[c] = [f"{v:.1f}" for v in tbl[c]]
        elif c.endswith("vs base"):
            out[c] = [f"{v:+.1%}" for v in tbl[c]]
        elif c in ("CAGR", "2017-21", "2022-26", "Offset 0", "Max drawdown", "Worst year", "Worst timing",
                   "Best timing", "Largest holding"):
            out[c] = [f"{v:.1%}" for v in tbl[c]]
    return out


def picked_on_tuning_years(avg: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """For each group, the setting with the best 2017-21 average, and what it did afterwards."""
    rows = []
    for g in groups:
        d = avg[avg["group"] == g]
        best = d.loc[d["2017-21"].idxmax()]
        rows.append({"Rule": g, "Best setting on 2017-21": best["variant"],
                     "2017-21 vs base": f"{best['2017-21 vs base']:+.1%}",
                     "2022-26 vs base": f"{best['2022-26 vs base']:+.1%}",
                     "Whole period vs base": f"{best['CAGR vs base']:+.1%}",
                     "Timings better": best["Timings better"]})
    return pd.DataFrame(rows).set_index("Rule")


def spread(d: pd.DataFrame) -> str:
    lo, hi = d["CAGR vs base"].min(), d["CAGR vs base"].max()
    return f"{lo:+.1%} to {hi:+.1%}"


def report(out: dict) -> str:
    runs, m, base0, sp = out["runs"], out["market"], out["base0"], out["spy"]
    avg = averages(runs)
    kw = {label: k for _, label, k in variants()}
    of = avg["variant"].map(kw)
    row = avg.set_index("variant")
    b = avg[avg["group"] == "Base"].iloc[0]
    k30, r = row.loc[BUFFER], row.loc[REFINED]
    exits = avg[avg["group"].isin(PROFIT_GROUPS[:-1])]
    up = exits[exits["CAGR vs base"] > 0]
    robust = exits[(exits["CAGR vs base"] > 0) & (exits["2017-21 vs base"] > 0) & (exits["2022-26 vs base"] > 0) &
                   (exits["Timings n"] >= 0.8 * len(OFFSETS))]
    best = exits.loc[exits["CAGR vs base"].idxmax()]
    sell_low = avg[of.map(lambda k: k.get("take_profit", 9) <= 0.5 and k.get("take_profit_part", 1) == 1)]
    trail = avg[avg["group"] == "Trailing stop"]
    keep = avg[of.map(lambda k: "keep_rank" in k and 20 <= k["keep_rank"] <= 50 and k.get("reweight", True))]
    spy_s = {k: stats(v)["CAGR"] for k, v in (("all", sp), ("a", sp[sp.index <= SPLIT]), ("b", sp[sp.index >= SPLIT]))}
    timing = "the best of the ten" if b["Offset 0"] >= b["Best timing"] - 1e-12 else "one of the better ones"
    first, last = base0.equity.index[0].date(), base0.equity.index[-1].date()

    cols = ["variant", "CAGR", "CAGR vs base", "2017-21", "2022-26", "Worst timing", "Max drawdown", "Sharpe",
            "Rule sales per year", "Timings better"]
    cols_other = [c for c in cols if c != "Rule sales per year"] + ["Turnover", "Largest holding"]
    headline = pd.DataFrame({
        name: {"Average of 10 timings": f"{x['CAGR']:.1%}", "Worst timing": f"{x['Worst timing']:.1%}",
               "Month-end (paper) timing": f"{x['Offset 0']:.1%}", "2017-21": f"{x['2017-21']:.1%}",
               "2022-26": f"{x['2022-26']:.1%}", "Max drawdown": f"{x['Max drawdown']:.1%}",
               "Sharpe": f"{x['Sharpe']:.2f}", "Turnover": f"{x['Turnover']:.1f}",
               "Largest holding": f"{x['Largest holding']:.0%}"}
        for name, x in (("Current paper rules", b), ("Keep while in the top 30", k30), (REFINED, r))}).T
    yr = out["years"].set_index(["variant", "offset"])
    yb, yk, yn = (yr.loc[v] for v in (b["variant"], BUFFER, REFINED))
    gain = (yk - yb).mean()  # the buffer's yearly edge, averaged over the timings
    top2 = gain.sort_values(ascending=False).head(2)
    trailed = int((gain < 0).sum())
    years = pd.DataFrame({"Current paper rules": yb.mean().map(lambda v: f"{v:+.1%}"),
                          BUFFER: yk.mean().map(lambda v: f"{v:+.1%}"),
                          "Difference": gain.map(lambda v: f"{v:+.1%}"),
                          "Timings better": ((yk - yb) > 1e-9).sum().map(lambda n: f"{n} of {len(OFFSETS)}"),
                          REFINED: yn.mean().map(lambda v: f"{v:+.1%}"),
                          "SPY": yearly(sp).set_axis(yearly(sp).index.astype(int)).map(lambda v: f"{v:+.1%}")})
    example = (" It won those years mostly by holding on to big winners the current rules sold early: Tesla, "
               "DocuSign and Nvidia in 2020, AppLovin and Vertiv in 2024." if set(top2.index) == {2020, 2024} else "")

    lines = [
        f"# Refining the momentum strategy: does taking profits help? ({first} to {last})", "",
        f"Generated by `python -m backtest.momentum_refine` on {dt.date.today()}. See the module docstring for the "
        "method.", "",
        "Base: the strategy the momentum paper account trades. At the last close of each month it picks the 10 "
        "stocks among the 500 largest with the best 6-1 month return and holds them at 10% each from the next "
        "open (10 bps a side, idle cash earns T-bills). Every variant changes one thing. Each runs ten times, "
        "with the monthly trade 0, 2, ..., 18 sessions after month end, and the tables show the average of the "
        "ten: a 10-stock portfolio's result depends a lot on the trading day. 'Timings better' counts the "
        "trading days at which the variant beat the base over the whole period.", "",
        "## Answer", "",
        f"**Taking profits at a set level did not raise returns.** Of {len(exits)} take-profit, trailing-stop, "
        f"'stretched' and stop-loss settings, {len(up)} beat the current rules on average, by at most "
        f"{best['CAGR vs base']:+.1%} a year ('{best['variant']}', better at {best['Timings better']} trading "
        f"days), which is within the noise. "
        + ("None was better in both 2017-21 and 2022-26 and at 8 or more of the 10 trading days. " if robust.empty
           else f"Better in both periods and at 8+ trading days: {', '.join(robust['variant'])}. ")
        + f"Selling everything at +10% to +50% changed the yearly return by {spread(sell_low)}, and trailing "
        f"stops by {spread(trail)}. The levels that looked best in 2017-21 did not hold up in 2022-26 (table "
        "below). The reason: most of the profit comes from a few holdings that keep running long after they "
        "are up 50% or 100%, and a profit target sells exactly those.", "",
        f"**What helped was the opposite: holding winners longer.** Selling a holding only when it drops out of "
        f"the top 30 (instead of the top 10) returned {k30['CAGR']:.1%} a year on average against {b['CAGR']:.1%}, "
        f"better at {k30['Timings better']} trading days and in both 2017-21 ({k30['2017-21 vs base']:+.1%}) and "
        f"2022-26 ({k30['2022-26 vs base']:+.1%}), with a smaller drawdown and "
        f"{1 - k30['Turnover'] / b['Turnover']:.0%} less trading. Every cut-off from the top 20 to the top 50 "
        f"helped ({spread(keep)} a year). But the gain is lumpy: it came mostly from {top2.index[0]} "
        f"({top2.iloc[0]:+.0%}) and {top2.index[1]} ({top2.iloc[1]:+.0%}), and the buffer trailed the current rules "
        f"in {trailed} of the {len(gain)} calendar years (table below).{example} Leaving winners at their grown "
        f"weight instead of trimming them back to 10% at each rebalance raised the average to {r['CAGR']:.1%} "
        f"({r['CAGR vs base']:+.1%}; better at {r['Timings better']} trading days and in both periods), but then "
        f"one stock reached {r['Largest holding']:.0%} of the account (current rules: {b['Largest holding']:.0%}), "
        "so the return rests on a few stocks.", "",
        f"**Expect less than the headline.** The paper account trades at the month end, which returned "
        f"{b['Offset 0']:.1%} a year with the current rules, {timing}; the average over the ten trading days was "
        f"{b['CAGR']:.1%} (worst {b['Worst timing']:.1%}). SPY returned {spy_s['all']:.1%} ({spy_s['a']:.1%} in "
        f"2017-21, {spy_s['b']:.1%} in 2022-26). Survivorship bias and the choice of these rules from about 100 "
        "settings flatter every strategy number here.", "",
        md(headline, "Strategy"), "",
        "Turnover: one-way trading per year as a multiple of the portfolio. Largest holding: the highest share of "
        "the account one stock reached at any timing.", "",
        "Each rule's best setting picked on 2017-21, then checked on 2022-26:", "",
        md(picked_on_tuning_years(avg, [g for g in avg["group"].unique() if g != "Base"]), "Rule"), "",
        "## Calendar years", "",
        "Average of the ten trading days ('Timings better': how many of them the buffer beat the current rules "
        "that year); SPY for reference. The first year starts on each timing's first trade.", "",
        md(years, "Year"), "",
        "## Why selling winners early costs money", "",
        "Positions of the current rules (month-end timing) that closed at least this far above their purchase "
        "price, and what they did from the next open until the strategy actually sold them:", "",
        md(after_level(m, base0), "Gain since purchase"), "",
        "After the target is hit roughly half rise further and half fall back, but the average further move is "
        "large because a few keep running a long way. A profit target sells those too.", "",
        "## Every profit-taking and stop variant", "",
        "Averages over the ten trading days. 'Rule sales per year' counts sales made by the rule (half sales "
        "included).", "",
    ]
    for g in PROFIT_GROUPS:
        part = pd.concat([avg[avg["group"] == "Base"], avg[avg["group"] == g]])
        lines += [f"### {g}", "", md(fmt(part, cols).set_index("variant"), "Variant"), ""]
    other = avg[~avg["group"].isin(PROFIT_GROUPS + ["Base"])]
    tbl = fmt(pd.concat([avg[avg["group"] == "Base"], other]), cols_other).rename(columns={"variant": "Variant"})
    tbl.insert(0, "Setting", ["Base"] + list(other["group"]))
    lines += ["## The strategy's other settings, one at a time", "", md(tbl.set_index("Setting"), "Setting"), "",
              "## Caveats", "",
              "- Survivorship bias: the data holds only stocks listed today, which flatters momentum (by about 1% a "
              "year for the largest 500 at equal weight, more for a concentrated portfolio).",
              "- The base was itself picked from about 100 momentum settings in the strategy search "
              "(backtest/STRATEGIES.md), so its level is optimistic; comparisons between variants of it are "
              "fairer than its absolute return.",
              "- Ten years, mostly a bull market led by large US tech. Profit taking and stops would look better in "
              "a market that rises and then gives it all back (2000-02), which is not in the sample.",
              "- Signals use closing prices and trade at the next open; a stop resting at the exchange would fill "
              "during the day, sometimes better, sometimes worse after a gap."]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    out = run_all()
    text = report(out)
    (REPO_ROOT / "backtest" / "MOMENTUM_REFINE.md").write_text(text + "\n")
    print(text)
    log.info("done in %.0fs", out["seconds"])


if __name__ == "__main__":
    main()

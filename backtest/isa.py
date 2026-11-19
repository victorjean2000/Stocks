"""Daily strategies for a sterling ISA, with London-listed 3x funds.

Funds, real London prices in pence (Yahoo, approved by the user; see YAHOO_APPROVED in backtest/lqq3.py): LQQ3
(WisdomTree NASDAQ 100 3x Daily Leveraged) and 3LUS (WisdomTree S&P 500 3x Daily Leveraged). Their signals come from
QQQ and SPY at the US close (Alpaca), after London has shut, so every trade happens at the next London open. 10 bps a
side; idle cash earns sterling interest (FXB). Benchmark: the S&P 500 in pounds. Settings are compared on 2017-21 and
checked on 2022-26.

Rules (backtest/daily.py): buy a fund after a short dip in its index (RSI(2) < x, IBS < x, an N-day low close or three
lower closes; optionally only above the 200-day average) and sell on the bounce (close above the 5-day average,
RSI(2) > 70, close above the previous high, or an N-day high close). Portfolios: one fund alone; both funds with half
the account each; or one position at a time, all in, in the more oversold index.

Short funds: London prices of the 3x short funds (such as 3ULS, the S&P 500 3x short) are not approved yet, so whether
shorting helps is tested on the US 3x funds instead, in dollars at the next US open: TQQQ / UPRO long after dips while
the index is above its 200-day average, SQQQ / SPXU (3x short) after short rises (the mirror rules) while it is below.

Run: python -m backtest.isa (about 20 s). Writes backtest/ISA.md and data/backtests/isa/.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from stockdata import alpaca
from stockdata.config import REPO_ROOT, paths

from .daily import ENTRIES, ENTRY_LABEL, EXIT_LABEL, EXITS, dip_signals, run
from .indicators import regime, rsi
from .lqq3 import COST_BPS, dfmt, dip_measures, dip_state, fetch_yahoo, sterling, to_gbp
from .momentum_trend import md, stats
from .strategies import SPLIT, START, yearly_returns

log = logging.getLogger(__name__)

FUNDS = {"LQQ3.L": "QQQ", "3LUS.L": "SPY"}  # London 3x fund (pence) -> the US ETF whose close gives its signal
NAMES = {"LQQ3.L": "LQQ3", "3LUS.L": "3LUS"}
US = ["QQQ", "SPY", "TQQQ", "UPRO", "SQQQ", "SPXU", "BIL"]
US_COST = 5.0
WIN_RATE, EXCESS = 0.60, 0.15
PORTFOLIOS = {"LQQ3 alone": (["LQQ3.L"], 1), "3LUS alone": (["3LUS.L"], 1),
              "both funds, half each": (list(FUNDS), 2), "one at a time, all in, more oversold first": (list(FUNDS), 1)}
MIRROR = {"rsi": "rsi_hi", "ibs": "ibs_hi", "low": "high_n", "down": "up"}  # the short-side mirror of each dip rule
SHORT_EXIT = {"sma5": "below_sma5", "rsi70": "rsi30", "high": "low_prior", "nhigh": "nlow"}


# ------------------------------------------------------------------ data

def us_bars(max_age_days: int = 3) -> dict[str, pd.DataFrame]:
    """Daily bars (o, h, l, c) of the US ETFs from Alpaca, cached in data/benchmarks/ (completed sessions only)."""
    path = paths().root / "benchmarks" / "longshort_daily.parquet"
    df = pd.read_parquet(path) if path.exists() else None
    if df is None or not set(US) <= set(df["ticker"]) or \
            (pd.Timestamp.today().normalize() - df["date"].max()).days > max_age_days:
        df = alpaca.drop_unfinished(alpaca.fetch_bars(alpaca.session(), US, "2016-01-01", alpaca.safe_end(), "all", "sip"))
        df.to_parquet(path, index=False)
    return {t: g.set_index("date")[["o", "h", "l", "c"]].astype(float).sort_index() for t, g in df.groupby("ticker")}


def london_wanted(state: pd.Series, lse: pd.DatetimeIndex) -> np.ndarray:
    """For each London session, the state after the last US close before that day (US close is 9pm UK)."""
    pos = state.index.searchsorted(lse, side="left") - 1
    return np.where(pos >= 0, state.to_numpy(bool)[np.clip(pos, 0, None)], False)


def trades_frame(tr: list, dates: pd.DatetimeIndex, names: list[str]) -> pd.DataFrame:
    t = pd.DataFrame(tr, columns=["j", "i_in", "i_out", "px_in", "px_out", "ret"]).astype(
        {"j": int, "i_in": int, "i_out": int, "ret": float})
    return pd.DataFrame({"asset": [names[j] for j in t["j"]], "entry": dates[t["i_in"]], "exit": dates[t["i_out"]],
                         "ret": t["ret"], "days": t["i_out"] - t["i_in"]})


# ------------------------------------------------------------------ ISA: London funds

def isa_run(funds: dict[str, pd.DataFrame], states: dict[str, pd.Series], scores: dict[str, pd.Series], n_pos: int,
            cash_r: np.ndarray, lse: pd.DatetimeIndex, i0: int, cost_bps: float = COST_BPS):
    """Hold each fund on the London sessions its US-close state asks for, at most n_pos at 1/n_pos of the account
    (the higher score first), trading at London opens. Returns equity (GBP, 1 at i0) and trades."""
    cols = list(states)
    O = np.column_stack([funds[f]["open"].reindex(lse).to_numpy() for f in cols])
    C = np.column_stack([funds[f]["close"].reindex(lse).ffill().to_numpy() for f in cols])
    want = np.column_stack([london_wanted(states[f], lse) for f in cols])
    enter, leave = np.zeros_like(want), np.zeros_like(want)
    enter[:-1], leave[:-1] = want[1:], ~want[1:]  # decided before the next session's open
    score = np.column_stack([np.nan_to_num(london_scores(scores[f], lse), nan=-np.inf) for f in cols])
    score[:-1] = score[1:]
    eq, _, tr = run(O, C, enter, leave, score, n_pos, cost_bps, cash_r, i0)
    return pd.Series(eq, index=lse).dropna(), trades_frame(tr, lse, [NAMES[f] for f in cols])


def london_scores(score: pd.Series, lse: pd.DatetimeIndex) -> np.ndarray:
    pos = score.index.searchsorted(lse, side="left") - 1
    return np.where(pos >= 0, score.to_numpy(float)[np.clip(pos, 0, None)], np.nan)


def isa_variants() -> list[tuple[str, dict]]:
    v = []
    for entry in ENTRIES:
        kind, x = entry
        for leave in EXITS + (["nhigh"] if kind == "low" else []):
            for trend in (True, False):
                for port in PORTFOLIOS:
                    label = (f"{port}: {ENTRY_LABEL[kind].format(x=x)}{', above 200-day' if trend else ''}; "
                             f"sell {EXIT_LABEL[leave].format(x=x)}")
                    v.append((label, dict(entry=entry, leave=leave, trend=trend, port=port)))
    return v


# ------------------------------------------------------------------ shorts: US 3x funds as the test bed

def rip_signals(c: pd.Series, h: pd.Series, l: pd.Series, entry: tuple, leave: str):
    """Buy-the-short-fund and sell-it flags: the mirror image of the dip rules, only below the 200-day average."""
    kind, x = entry
    if kind == "rsi":
        e = rsi(c, 2) > 100 - x
    elif kind == "ibs":
        e = (c - l) / (h - l) > 1 - x
    elif kind == "low":
        e = c >= c.rolling(x).max()
    else:
        e = (c.diff() > 0).astype(int).rolling(x).sum() == x
    e &= c < c.rolling(200).mean()
    if leave == "sma5":
        out = c < c.rolling(5).mean()
    elif leave == "rsi70":
        out = rsi(c, 2) < 30
    elif leave == "high":
        out = c < l.shift(1)
    else:
        out = c <= c.rolling(x).min()
    return e.fillna(False).astype(bool), out.fillna(False).astype(bool)


def us_run(b: dict[str, pd.DataFrame], legs: list[tuple[str, pd.Series, pd.Series]], n_pos: int):
    """legs: (traded ETF, buy flags, sell flags) on US dates; next-open fills in dollars; cash earns T-bills."""
    dates = b["QQQ"].index
    O = np.column_stack([b[t]["o"].reindex(dates).to_numpy() for t, _, _ in legs])
    C = np.column_stack([b[t]["c"].reindex(dates).ffill().to_numpy() for t, _, _ in legs])
    E = np.column_stack([e.reindex(dates, fill_value=False).to_numpy(bool) for _, e, _ in legs])
    X = np.column_stack([x.reindex(dates, fill_value=False).to_numpy(bool) for _, _, x in legs])
    cash = b["BIL"]["c"].reindex(dates).ffill().pct_change().fillna(0.0).to_numpy()
    i0 = int(dates.searchsorted(pd.Timestamp(START))) - 1
    eq, _, tr = run(O, C, E, X, np.ones(E.shape), n_pos, US_COST, cash, i0)
    return pd.Series(eq, index=dates).dropna(), trades_frame(tr, dates, [t for t, _, _ in legs])


def short_tests(b: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For every dip setting (trend filter on): long only, long plus the mirror shorts, and the trend rules."""
    spy = b["SPY"]["c"]
    spy = spy[spy.index >= b["QQQ"].index[int(b["QQQ"].index.searchsorted(pd.Timestamp(START))) - 1]]
    rows = {}
    for entry in ENTRIES:
        kind, x = entry
        for leave in EXITS + (["nhigh"] if kind == "low" else []):
            long_legs, short_legs = [], []
            for idx, lev, inv in (("QQQ", "TQQQ", "SQQQ"), ("SPY", "UPRO", "SPXU")):
                c, h, l = (b[idx][k] for k in "chl")
                e, x_ = dip_signals(c, h, l, entry, leave, True)
                se, sx = rip_signals(c, h, l, entry, leave)
                long_legs.append((lev, e, x_))
                short_legs.append((inv, se, sx))
            label = f"{ENTRY_LABEL[kind].format(x=x)}, sell {EXIT_LABEL[leave].format(x=x)}"
            for name, legs in (("long dips only", long_legs), ("long dips + short rips", long_legs + short_legs),
                               ("short rips only", short_legs)):
                eq, t = us_run(b, legs, 2)
                rows[(label, name)] = dip_measures(eq, t, spy / spy.iloc[0])
    for idx, lev, inv in (("QQQ", "TQQQ", "SQQQ"), ("SPY", "UPRO", "SPXU")):
        on = regime(b[idx]["c"], 200, 1)
        for name, legs in ((f"trend: {lev} above 200-day, else cash", [(lev, on, ~on)]),
                           (f"trend: {lev} above 200-day, else {inv}", [(lev, on, ~on), (inv, ~on, on)])):
            eq, t = us_run(b, legs, 1)
            rows[("Trend rules", name)] = dip_measures(eq, t, spy / spy.iloc[0])
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------ report

def run_all() -> dict:
    b = us_bars()
    funds = {f: fetch_yahoo(f, max_age_days=4) for f in FUNDS}
    last = min(min(f.index[-1] for f in funds.values()), b["QQQ"].index[-1] + pd.Timedelta(days=1))
    lse = funds["LQQ3.L"].index.union(funds["3LUS.L"].index)
    lse = lse[lse <= last]
    gbpusd, cash = sterling()
    cash_r = cash.reindex(lse).fillna(0.0).to_numpy()
    i0 = int(lse.searchsorted(pd.Timestamp(START))) - 1
    spy = b["SPY"]["c"][b["SPY"].index >= lse[i0]]
    bench = to_gbp(spy / spy.iloc[0], gbpusd)
    oversold = {f: -rsi(b[FUNDS[f]]["c"], 2) for f in FUNDS}
    rows, curves, trades = {}, {}, {}
    for label, kw in isa_variants():
        cols, n = PORTFOLIOS[kw["port"]]
        states = {f: dip_state(b[FUNDS[f]].rename(columns=str), kw["entry"], kw["leave"], kw["trend"]) for f in cols}
        eq, t = isa_run(funds, states, oversold, n, cash_r, lse, i0)
        rows[label], curves[label], trades[label] = dip_measures(eq, t, bench), eq, t
    refs = {}
    for name, cols, n in (("Your current ISA rule: LQQ3 while QQQ is above its 200-day avg", ["LQQ3.L"], 1),
                          ("Both funds, half each, while their index is above its 200-day avg", list(FUNDS), 2)):
        states = {f: regime(b[FUNDS[f]]["c"], 200, 1) for f in cols}
        eq, t = isa_run(funds, states, oversold, n, cash_r, lse, i0)
        refs[name] = dip_measures(eq, t, bench)
        curves[name] = eq
    tbl = pd.DataFrame(rows).T
    shorts = short_tests(b)
    out_dir = paths().root / "backtests" / "isa"
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_dir / "isa_variants.csv")
    shorts.to_csv(out_dir / "short_tests.csv")
    return {"table": tbl, "refs": pd.DataFrame(refs).T, "curves": curves, "trades": trades, "bench": bench,
            "shorts": shorts, "lse": lse, "i0": i0}


def report(out: dict) -> str:
    tbl, refs, curves, trades, bench, shorts = (out[k] for k in ("table", "refs", "curves", "trades", "bench", "shorts"))
    split = pd.Timestamp(SPLIT)
    sp = {k: stats(v)["CAGR"] for k, v in (("all", bench), ("a", bench[bench.index <= split]),
                                           ("b", bench[bench.index >= split]))}
    port = pd.Series([s.split(":")[0] for s in tbl.index], index=tbl.index)
    ok = tbl[tbl["2017-21 win rate"] >= WIN_RATE]
    pick = ok["2017-21"].idxmax() if len(ok) else None
    hit = tbl[(tbl["Win rate"] >= WIN_RATE) & (tbl["vs S&P 500"] >= EXCESS)]
    both = hit[(hit["2017-21 vs S&P"] >= EXCESS) & (hit["2022-26 vs S&P"] >= EXCESS)]
    main = ["CAGR", "vs S&P 500", "2017-21 vs S&P", "2022-26 vs S&P", "Win rate", "Trades per year",
            "Avg hold (sessions)", "Max drawdown", "Worst year"]
    by_port = tbl.groupby(port).agg(**{
        "Settings": ("CAGR", "size"), "Median CAGR": ("CAGR", "median"), "Median vs S&P": ("vs S&P 500", "median"),
        "Median 2022-26 vs S&P": ("2022-26 vs S&P", "median"), "Median win rate": ("Win rate", "median"),
        "Median max drawdown": ("Max drawdown", "median"),
        f"Beat S&P by {EXCESS:.0%}+ in both halves": ("2022-26 vs S&P", lambda s: int(
            ((s >= EXCESS) & (tbl.loc[s.index, "2017-21 vs S&P"] >= EXCESS)).sum()))})
    by_port = by_port.reindex(list(PORTFOLIOS))
    for c in ("Median CAGR", "Median win rate", "Median max drawdown"):
        by_port[c] = by_port[c].map(lambda v: f"{v:.1%}")
    for c in ("Median vs S&P", "Median 2022-26 vs S&P"):
        by_port[c] = by_port[c].map(lambda v: f"{v:+.1%}")
    shown = ([pick] if pick else []) + list(both.sort_values("CAGR", ascending=False).index[:10])
    shown = list(dict.fromkeys(shown))
    first, last = bench.index[0].date(), bench.index[-1].date()
    lines = [
        f"# Daily strategies for a sterling ISA ({first} to {last})", "",
        f"Generated by `python -m backtest.isa` on {dt.date.today()}. See the module docstring for the rules. Real "
        "London prices of LQQ3 (3x Nasdaq-100) and 3LUS (3x S&P 500) in pence; signals at the US close, trades at "
        f"the next London open, {COST_BPS:g} bps a side, sterling interest on idle cash. Benchmark: the S&P 500 in "
        f"pounds, {sp['all']:.1%} a year ({sp['a']:.1%} in 2017-21, {sp['b']:.1%} in 2022-26). Targets: at least "
        f"{WIN_RATE:.0%} winning trades and {EXCESS:.0%} a year above the S&P 500.", "",
        "## Answer", "",
        f"{len(tbl)} settings tested; {len(hit)} reached both targets over the whole period and {len(both)} in both "
        "2017-21 and 2022-26.", "",
        "By portfolio:", "", md(by_port, "Portfolio"), "",
        "The pick chosen on 2017-21 only (best return among settings that won at least "
        f"{WIN_RATE:.0%} of their 2017-21 trades) first, then the settings that beat the S&P by {EXCESS:.0%} or more in "
        "both halves:", "", md(dfmt(tbl.loc[shown, main]), "Setting"), "",
        "For comparison:", "", md(dfmt(refs[main]), "Rule"), "",
        "## The steadiest family: buy when the index closes in the bottom 10% of its day's range (IBS < 0.1)", "",
        "Every IBS < 0.1 setting, best first. Picked out after seeing the results, so treat the 2022-26 column as the "
        "realistic expectation rather than the whole-period one:", "",
        md(dfmt(tbl[tbl.index.str.contains(": IBS < 0.1")].sort_values("CAGR", ascending=False)[main]), "Setting"), ""]
    if pick:
        years = pd.DataFrame({**{s: yearly_returns(curves[s]) for s in shown[:3]},
                              "S&P 500 in pounds": yearly_returns(bench)})
        worst = trades[pick].sort_values("ret").head(5)
        lines += ["## Calendar years", "", md(years.map(lambda v: f"{v:+.1%}"), "Year"), "",
                  "Worst trades of the 2017-21 pick: " + ", ".join(
                      f"{r.asset} {r.entry.date()} to {r.exit.date()} {r.ret:+.1%}" for r in worst.itertuples()) + ".",
                  ""]
    sh = shorts.copy()
    pairs = sh.drop(index="Trend rules", level=0, errors="ignore")
    lo = pairs.xs("long dips only", level=1)
    ls = pairs.xs("long dips + short rips", level=1)
    so = pairs.xs("short rips only", level=1)
    gain = pd.DataFrame({"CAGR": ls["CAGR"] - lo["CAGR"], "2022-26": ls["2022-26"] - lo["2022-26"],
                         "Max drawdown": ls["Max drawdown"] - lo["Max drawdown"]})
    lines += [
        "## Would 3x short funds help? (tested on the US funds)", "",
        "London prices of the short funds (e.g. 3ULS, the S&P 500 3x short) are not approved for this repository yet, "
        "so the question is answered on the US 3x funds, in dollars at the next US open: long TQQQ / UPRO after dips "
        "while the index is above its 200-day average, and short via SQQQ / SPXU after short rises (the mirror rules) "
        "while it is below. Two places, half the account each.", "",
        f"Across the {len(lo)} rule pairs, adding the short side changed the yearly return by "
        f"{gain['CAGR'].median():+.1%} (median; {(gain['CAGR'] > 0).mean():.0%} of pairs improved), the 2022-26 return "
        f"by {gain['2022-26'].median():+.1%}, and the max drawdown by {gain['Max drawdown'].median():+.1%}. The short "
        f"side alone: median {so['CAGR'].median():.1%} a year with {so['Win rate'].median():.0%} winning trades.", "",
        md(dfmt(pd.concat({"long dips only": lo, "long dips + short rips": ls}, axis=0).swaplevel().sort_index()
                .loc[:, ["CAGR", "2017-21", "2022-26", "Win rate", "Max drawdown"]]).reset_index()
           .rename(columns={"level_0": "Rule", "level_1": "Sides"}).set_index("Rule"), "Rule"), "",
        "Trend rules (hold the 3x fund above the 200-day average; below it hold cash, or the 3x short fund):", "",
        md(dfmt(sh.xs("Trend rules", level=0)[["CAGR", "2017-21", "2022-26", "Win rate", "Max drawdown"]]), "Rule"), "",
        "## Caveats", "",
        f"- {len(tbl)} settings were tried; whole-period winners are flattered by selection. The pick was chosen on "
        "2017-21 only, so its 2022-26 column is the honest test.",
        "- 3x funds can lose most of their value in a crash, and nothing can be sold before the next London open after "
        "a US-close signal; the London open of a 3x fund can also have a wide spread.",
        "- The short-side test uses US funds at the US open; London short funds trade at the London open, which the "
        "long-side results show can cost several points a year."]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    text = report(run_all())
    (REPO_ROOT / "backtest" / "ISA.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

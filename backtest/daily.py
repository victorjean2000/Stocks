"""Daily strategies with a high win rate: buy a short-term dip in an uptrend, sell on the bounce.

Every rule is checked at each close and traded at the next open (never at the price that produced the signal), with
5 bps a side for ETFs and 10 bps for stocks; idle cash earns T-bills (BIL). A trade wins if it made money after
costs. Settings are compared on 2017-21 and checked on 2022-26, as in backtest/strategies.py.

Families
  Index dips   signal on QQQ or SPY, traded in the index ETF or its 2x / 3x version (QLD, TQQQ; SSO, UPRO). Buy on
               RSI(2) < x, IBS < x (close near the day's low), a close at an N-day low, or three lower closes in a
               row; optionally only while the index closes above its 200-day average. Sell when the close is above
               the 5-day average, RSI(2) > 70, the close beats the previous day's high, or (N-day lows) at an N-day
               high close.
  Stock dips   the 500 largest stocks closing above their 200-day average: buy those with RSI(2) < x, biggest
               3-day fall first, up to n positions of 1/n of the account each; sell when the close is above the
               5-day average or RSI(2) > 70, optionally after at most 10 sessions; optionally only while SPY closes
               above its 200-day average.
  Fund dips    the same index-dip rules on nine 3x US equity funds at once, each signalled by its unleveraged
               ETF (FUNDS: Nasdaq-100, S&P 500, Dow, Russell 2000, tech, semiconductors, financials, biotech,
               homebuilders; chosen in advance as the 3x US index and sector funds with prices since 2016 and at
               least $50m traded a day), up to n at a time at 1/n of the account each, most oversold (lowest RSI(2))
               first; the same rules traded in the unleveraged ETFs for comparison.

Run: python -m backtest.daily (about a minute on 4 cores). Writes backtest/DAILY.md and data/backtests/daily/.
"""
from __future__ import annotations

import datetime as dt
import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stockdata import alpaca
from stockdata.config import REPO_ROOT, paths

from .indicators import rsi
from .momentum_trend import Market, load_market, md, stats, yearly
from .strategies import SPLIT, START, etf_bars

log = logging.getLogger(__name__)

ETF_COST, STOCK_COST = 5.0, 10.0  # bps a side
WIN_RATE = 0.60  # a "high probability" strategy: at least 60% of its trades make money after costs
EXCESS = 0.15  # the target: 15 points a year above SPY
TRADED = {"QQQ": ["QQQ", "QLD", "TQQQ"], "SPY": ["SPY", "SSO", "UPRO"]}
ENTRIES = [("rsi", 5), ("rsi", 10), ("rsi", 20), ("ibs", 0.1), ("ibs", 0.2), ("low", 5), ("low", 7), ("down", 3)]
EXITS = ["sma5", "rsi70", "high"]  # N-day lows also try an N-day high close
FUNDS = {"TQQQ": "QQQ", "UPRO": "SPY", "UDOW": "DIA", "TNA": "IWM", "TECL": "XLK", "SOXL": "SOXX", "FAS": "XLF",
         "LABU": "XBI", "NAIL": "ITB"}  # 3x fund -> the unleveraged ETF its signals come from
# The fund setting with the smallest drawdown among those beating SPY by EXCESS in both halves: chosen after
# seeing the results (every IBS < 0.1 fund setting passed), so the 2017-21 pick stays the honest test.
BALANCED = "9 3x funds, up to 9 at a time: IBS < 0.1; sell close > prior high"
FUND_ENTRIES = [("ibs", 0.1), ("ibs", 0.2), ("rsi", 10), ("rsi", 20), ("low", 5), ("low", 7)]
ENTRY_LABEL = {"rsi": "RSI(2) < {x:g}", "ibs": "IBS < {x:g}", "low": "{x}-day low close", "down": "{x} lower closes"}
EXIT_LABEL = {"sma5": "close > 5-day avg", "rsi70": "RSI(2) > 70", "high": "close > prior high",
              "nhigh": "{x}-day high close"}


@dataclass
class Data:
    m: Market
    dates: pd.DatetimeIndex
    etf: dict[str, pd.DataFrame]  # "o", "h", "l", "c": ETF bars on the stock calendar (completed sessions only)
    cash_r: np.ndarray
    i0: int  # the close before START: equity is 1 there and the first trade fills at START's open
    x: dict = field(default_factory=dict)  # stock signal frames, computed once
    fund: dict = field(default_factory=dict)  # "o", "h", "l", "c": the 3x funds and their ETFs (FUNDS)


def fund_bars(symbols=tuple(sorted(set(FUNDS) | set(FUNDS.values()))), start: str = "2016-01-01") -> pd.DataFrame:
    """Adjusted daily bars of the 3x funds and their ETFs from Alpaca, cached in data/benchmarks/ (completed
    sessions only)."""
    path = paths().root / "benchmarks" / "funds_daily.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_parquet(path)
        fresh = (pd.Timestamp.today().normalize() - df["date"].max()).days <= 3
        if fresh and set(symbols) <= set(df["ticker"]):
            return df
    df = alpaca.drop_unfinished(alpaca.fetch_bars(alpaca.session(), list(symbols), start, alpaca.safe_end(), "all",
                                                  "sip"))
    df.to_parquet(path, index=False)
    return df


def load() -> Data:
    m = load_market()
    dates = m.close.index
    bars = etf_bars()
    etf = {k: bars.pivot(index="date", columns="ticker", values=k).reindex(dates).astype(float) for k in "ohlc"}
    cash_r = etf["c"]["BIL"].ffill().pct_change().fillna(0.0).to_numpy()
    d = Data(m, dates, etf, cash_r, int(dates.searchsorted(pd.Timestamp(START))) - 1)
    fb = fund_bars()
    d.fund = {k: fb.pivot(index="date", columns="ticker", values=k).reindex(dates).astype(float) for k in "ohlc"}
    c = m.close
    spy = etf["c"]["SPY"]
    d.x.update({"rsi2": rsi(c, 2), "above200": c > m.sma(200), "above5": c > m.sma(5), "fall3": -(c / c.shift(3) - 1),
                "pool500": m.universe & (m.size_rank <= 500), "spy_on": spy > spy.rolling(200).mean()})
    return d


# ------------------------------------------------------------------ engine

def run(O: np.ndarray, C: np.ndarray, enter: np.ndarray, leave: np.ndarray, score: np.ndarray, n_pos: int,
        cost_bps: float, cash_r: np.ndarray, i0: int, max_hold: int | None = None, size: np.ndarray | None = None):
    """Up to n_pos positions of 1/n_pos of the account each. At each close, holdings with a `leave` signal (or held
    max_hold sessions) are sold at the next open, and free places go to the `enter` candidates with the highest
    score, bought at the next open. C must be forward-filled. size: share of a place to buy at each step (default
    all of it). Returns equity (1.0 at i0's close), the invested share of the account, and trades (column, entry
    index, exit index, entry price, exit price, return)."""
    n_t = C.shape[0]
    cost = cost_bps / 1e4
    cash = 1.0
    pos: dict[int, list] = {}  # column -> [shares, entry index, entry price]
    sell: set[int] = set()
    buy: list[int] = []
    eq, inv = np.full(n_t, np.nan), np.full(n_t, np.nan)
    trades = []
    for i in range(i0, n_t):
        if i > i0:
            for j in sorted(sell):
                px = O[i, j]
                if not np.isfinite(px):
                    continue  # no trade today: sell at the next open instead
                sh, ei, epx = pos.pop(j)
                cash += sh * px * (1 - cost)
                trades.append((j, ei, i, epx, px, (1 - cost) ** 2 * px / epx - 1))
                sell.discard(j)
            if buy:
                mark = cash + sum(sh * (O[i, j] if np.isfinite(O[i, j]) else C[i - 1, j]) for j, (sh, _, _) in pos.items())
                for j in buy:
                    px, amount = O[i, j], min(mark / n_pos * (1.0 if size is None else size[i, j]), cash)
                    if len(pos) >= n_pos or amount <= 0:
                        break
                    if np.isfinite(px) and j not in pos:
                        pos[j] = [amount * (1 - cost) / px, i, px]
                        cash -= amount
                buy = []
            cash *= 1 + cash_r[i]
        value = sum(sh * C[i, j] for j, (sh, _, _) in pos.items())
        eq[i] = cash + value
        inv[i] = value / eq[i]
        if i + 1 >= n_t:
            break
        sell |= {j for j, (_, ei, _) in pos.items() if leave[i, j] or (max_hold and i + 1 - ei >= max_hold)}
        free = n_pos - (len(pos) - len(sell))
        if free > 0:
            cand = np.flatnonzero(enter[i])
            if cand.size:
                cand = cand[np.argsort(-score[i, cand], kind="stable")]
                buy = [int(j) for j in cand if j not in pos][:free]
    return eq, inv, trades


# ------------------------------------------------------------------ strategies

def index_signals(d: Data, signal: str, entry: tuple, leave: str, trend: bool) -> tuple[np.ndarray, np.ndarray]:
    e, out = dip_signals(*(d.etf[k][signal] for k in ("c", "h", "l")), entry, leave, trend)
    return e.to_numpy(bool), out.to_numpy(bool)


def dip_signals(c, h, l, entry: tuple, leave: str, trend: bool):
    """Buy and sell flags at each close for one ETF (Series) or several (DataFrame columns)."""
    kind, x = entry
    if kind == "rsi":
        e = rsi(c, 2) < x
    elif kind == "ibs":
        e = (c - l) / (h - l) < x
    elif kind == "low":
        e = c <= c.rolling(x).min()
    else:  # x lower closes in a row
        e = (c.diff() < 0).astype(int).rolling(x).sum() == x
    if trend:
        e &= c > c.rolling(200).mean()
    if leave == "sma5":
        out = c > c.rolling(5).mean()
    elif leave == "rsi70":
        out = rsi(c, 2) > 70
    elif leave == "high":
        out = c > h.shift(1)
    else:  # an N-day high close
        out = c >= c.rolling(x).max()
    return e.fillna(False).astype(bool), out.fillna(False).astype(bool)


def index_label(signal: str, trade: str, entry: tuple, leave: str, trend: bool) -> str:
    kind, x = entry
    return (f"{signal} {ENTRY_LABEL[kind].format(x=x)}{', above 200-day' if trend else ''}; "
            f"sell {EXIT_LABEL[leave].format(x=x)}; trade {trade}")


def variants() -> list[tuple[str, str, dict]]:
    """(family, label, keyword arguments of one_run)."""
    v = []
    for signal, trades in TRADED.items():
        for trade in trades:
            for entry in ENTRIES:
                for leave in EXITS + (["nhigh"] if entry[0] == "low" else []):
                    for trend in (True, False):
                        v.append(("Index dips", index_label(signal, trade, entry, leave, trend),
                                  dict(kind="index", signal=signal, trade=trade, entry=entry, leave=leave,
                                       trend=trend)))
    for x in (5, 10):
        for n in (5, 10, 20):
            for leave in ("sma5", "rsi70"):
                for market in (False, True):
                    for hold in (None, 10):
                        label = (f"Largest 500, RSI(2) < {x}, {n} positions; sell {EXIT_LABEL[leave]}"
                                 f"{', max 10 sessions' if hold else ''}{'; SPY above 200-day' if market else ''}")
                        v.append(("Stock dips", label, dict(kind="stocks", x=x, n=n, leave=leave, market=market,
                                                             max_hold=hold)))
    for lever in (True, False):
        for entry in FUND_ENTRIES:
            for leave in EXITS:
                for trend in (True, False):
                    for n in ((3, 5, 9) if lever else (9,)):
                        kind, x = entry
                        label = (f"{'9 3x funds' if lever else '9 unleveraged ETFs'}, up to {n} at a time: "
                                 f"{ENTRY_LABEL[kind].format(x=x)}{', above 200-day' if trend else ''}; "
                                 f"sell {EXIT_LABEL[leave].format(x=x)}")
                        v.append(("Fund dips" if lever else "Fund dips, unleveraged", label,
                                  dict(kind="funds", entry=entry, leave=leave, trend=trend, n=n, lever=lever)))
    # references: buy and hold, and the LQQ3 trend rule (daily check, rare trades)
    v += [("Reference", f"Buy and hold {t}", dict(kind="hold", trade=t)) for t in ("QQQ", "TQQQ")]
    v.append(("Reference", "TQQQ while QQQ closes above its 200-day avg (the LQQ3 account's rule)",
              dict(kind="trend", signal="QQQ", trade="TQQQ")))
    return v


def one_run(d: Data, kind: str, **kw):
    ones = None
    if kind in ("index", "hold", "trend"):
        trade = kw["trade"]
        O, C = d.etf["o"][[trade]].to_numpy(), d.etf["c"][[trade]].ffill().to_numpy()
        if kind == "index":
            e, x = index_signals(d, kw["signal"], kw["entry"], kw["leave"], kw["trend"])
        elif kind == "hold":
            e, x = np.ones(len(d.dates), bool), np.zeros(len(d.dates), bool)
        else:
            c = d.etf["c"][kw["signal"]]
            on = (c > c.rolling(200).mean()).to_numpy(bool)
            e, x = on, ~on
        ones = np.ones((len(d.dates), 1))
        return run(O, C, e[:, None], x[:, None], ones, 1, ETF_COST, d.cash_r, d.i0)
    if kind == "funds":
        traded = list(FUNDS) if kw["lever"] else list(FUNDS.values())
        sig = {k: d.fund[k][list(FUNDS.values())].set_axis(traded, axis=1) for k in "chl"}
        e, x = dip_signals(sig["c"], sig["h"], sig["l"], kw["entry"], kw["leave"], kw["trend"])
        score = -rsi(sig["c"], 2)
        return run(d.fund["o"][traded].to_numpy(), d.fund["c"][traded].ffill().to_numpy(), e.to_numpy(bool),
                   x.to_numpy(bool), score.fillna(-np.inf).to_numpy(), kw["n"], kw.get("cost", ETF_COST), d.cash_r,
                   d.i0)
    m, s = d.m, d.x
    ok = s["pool500"] & s["above200"] & (s["rsi2"] < kw["x"])
    if kw["market"]:
        ok = ok & s["spy_on"].to_numpy(bool)[:, None]
    out = s["above5"] if kw["leave"] == "sma5" else s["rsi2"] > 70
    return run(m.open.to_numpy(), m.close.ffill().to_numpy(), ok.fillna(False).to_numpy(bool),
               out.fillna(False).to_numpy(bool), s["fall3"].fillna(-np.inf).to_numpy(), kw["n"], STOCK_COST,
               d.cash_r, d.i0, kw["max_hold"])


# ------------------------------------------------------------------ evaluation

def evaluate(d: Data, eq: np.ndarray, inv: np.ndarray, trades: list, names: list[str], spy: pd.Series) -> tuple:
    dates = d.dates
    e = pd.Series(eq, index=dates).dropna()
    t = pd.DataFrame(trades, columns=["j", "i_in", "i_out", "px_in", "px_out", "ret"]).astype(
        {"j": int, "i_in": int, "i_out": int, "px_in": float, "px_out": float, "ret": float})
    t["asset"] = [names[j] for j in t["j"]]
    t["entry"], t["exit"] = dates[t["i_in"]], dates[t["i_out"]]
    t["days"] = t["i_out"] - t["i_in"]
    years = (e.index[-1] - e.index[0]).days / 365.25
    split = pd.Timestamp(SPLIT)
    a, b = e[e.index <= split], e[e.index >= split]
    sa, sb = spy[spy.index <= split], spy[spy.index >= split]
    s = stats(e)
    wins, losses = t.loc[t["ret"] > 0, "ret"], t.loc[t["ret"] <= 0, "ret"]
    early, late = t[t["exit"] <= split], t[t["exit"] > split]
    row = {"CAGR": s["CAGR"], "vs SPY": s["CAGR"] - stats(spy)["CAGR"],
           "2017-21": stats(a)["CAGR"], "2017-21 vs SPY": stats(a)["CAGR"] - stats(sa)["CAGR"],
           "2022-26": stats(b)["CAGR"], "2022-26 vs SPY": stats(b)["CAGR"] - stats(sb)["CAGR"],
           "Win rate": (t["ret"] > 0).mean() if len(t) else np.nan,
           "2017-21 win rate": (early["ret"] > 0).mean() if len(early) else np.nan,
           "2022-26 win rate": (late["ret"] > 0).mean() if len(late) else np.nan,
           "Trades per year": len(t) / years, "Avg hold (sessions)": t["days"].mean() if len(t) else np.nan,
           "Avg trade": t["ret"].mean() if len(t) else np.nan,
           "Avg win": wins.mean() if len(wins) else np.nan, "Avg loss": losses.mean() if len(losses) else np.nan,
           "Worst trade": t["ret"].min() if len(t) else np.nan,
           "Profit factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.nan,
           "Time invested": float(np.nanmean(inv[d.i0:])), "Max drawdown": s["Max drawdown"],
           "Sharpe": s["Sharpe (rf=0)"], "Worst year": yearly(e).min()}
    return row, e, t


_D: Data | None = None  # shared with forked worker processes


def _one(task: tuple):
    family, label, kw = task
    eq, inv, trades = one_run(_D, **kw)
    if "trade" in kw:
        names = [kw["trade"]]
    elif kw["kind"] == "funds":
        names = list(FUNDS) if kw["lever"] else list(FUNDS.values())
    else:
        names = [str(c) for c in _D.m.close.columns]
    spy = _D.etf["c"]["SPY"].iloc[_D.i0:]
    row, e, t = evaluate(_D, eq, inv, trades, names, spy)
    return {"Family": family, "Strategy": label} | row, e, t


def run_all(workers: int | None = None) -> dict:
    global _D
    t0 = time.time()
    _D = load()
    tasks = variants()
    workers = workers or min(4, os.cpu_count() or 1)
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            done = pool.map(_one, tasks, chunksize=4)
    else:
        done = [_one(t) for t in tasks]
    tbl = pd.DataFrame([r for r, _, _ in done]).set_index("Strategy")
    curves = {label: e for (_, label, _), (_, e, _) in zip(tasks, done)}
    trades = {label: t for (_, label, _), (_, _, t) in zip(tasks, done)}
    spy = _D.etf["c"]["SPY"].iloc[_D.i0:]
    out_dir = paths().root / "backtests" / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_dir / "all_variants.csv")
    # cost sensitivity of the fund settings shown in detail
    kws = {label: kw for _, label, kw in tasks}
    costs = {}
    for label in {pick(tbl, "Fund dips"), BALANCED} - {None}:
        for bps in (10.0, 20.0):
            eq, inv, tr = one_run(_D, **(kws[label] | {"cost": bps}))
            costs[(label, bps)] = evaluate(_D, eq, inv, tr, list(FUNDS), spy)[0]
    log.info("%d variants in %.0fs", len(tasks), time.time() - t0)
    return {"data": _D, "table": tbl, "curves": curves, "trades": trades, "spy": spy / spy.iloc[0], "costs": costs,
            "seconds": time.time() - t0}


def pick(tbl: pd.DataFrame, family: str) -> str | None:
    """The family's best 2017-21 CAGR among settings that won at least WIN_RATE of their 2017-21 trades."""
    f = tbl[(tbl["Family"] == family) & (tbl["2017-21 win rate"] >= WIN_RATE)]
    return f["2017-21"].idxmax() if len(f) else None


# ------------------------------------------------------------------ report

PCT = ["CAGR", "vs SPY", "2017-21", "2017-21 vs SPY", "2022-26", "2022-26 vs SPY", "Win rate", "2017-21 win rate",
       "2022-26 win rate", "Avg trade", "Avg win", "Avg loss", "Worst trade", "Time invested", "Max drawdown",
       "Worst year"]


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    for c in df.columns:
        if c in PCT:
            out[c] = [("" if pd.isna(v) else f"{v:+.1%}" if "vs SPY" in c else f"{v:.1%}") for v in df[c]]
        elif df[c].dtype.kind == "f":
            out[c] = [("" if pd.isna(v) else f"{v:.2f}" if c in ("Sharpe", "Profit factor") else f"{v:.0f}")
                      for v in df[c]]
    return out


def robustness(f: pd.DataFrame, what: str) -> str:
    return (f"{what.format(n=len(f))}: median win rate {f['Win rate'].median():.0%}, median "
            f"{f['vs SPY'].median():+.1%} a year against SPY ({f['2017-21 vs SPY'].median():+.1%} in 2017-21, "
            f"{f['2022-26 vs SPY'].median():+.1%} in 2022-26), median max drawdown {f['Max drawdown'].median():.0%}; "
            f"{(f['vs SPY'] >= EXCESS).mean():.0%} beat SPY by {EXCESS:.0%} or more over the whole period and "
            f"{((f['2017-21 vs SPY'] >= EXCESS) & (f['2022-26 vs SPY'] >= EXCESS)).mean():.0%} in both halves.")


def report(out: dict) -> str:
    tbl, curves, trades, spy, d = out["table"], out["curves"], out["trades"], out["spy"], out["data"]
    spy_s = {k: stats(v)["CAGR"] for k, v in (("all", spy), ("a", spy[spy.index <= pd.Timestamp(SPLIT)]),
                                              ("b", spy[spy.index >= pd.Timestamp(SPLIT)]))}
    tested = tbl[tbl["Family"] != "Reference"]
    hit = tested[(tested["Win rate"] >= WIN_RATE) & (tested["vs SPY"] >= EXCESS)]
    hit_both = hit[(hit["2017-21 vs SPY"] >= EXCESS) & (hit["2022-26 vs SPY"] >= EXCESS)]
    picks = {f: pick(tbl, f) for f in ("Index dips", "Fund dips", "Stock dips")}
    main = ["CAGR", "vs SPY", "2017-21", "2017-21 vs SPY", "2022-26", "2022-26 vs SPY", "Win rate", "Trades per year",
            "Avg hold (sessions)", "Max drawdown"]
    detail = ["Win rate", "2017-21 win rate", "2022-26 win rate", "Trades per year", "Avg hold (sessions)", "Avg trade",
              "Avg win", "Avg loss", "Worst trade", "Profit factor", "Time invested", "Max drawdown", "Sharpe",
              "Worst year"]
    first, last = spy.index[0].date(), spy.index[-1].date()
    lines = [
        f"# Daily strategies with a high win rate ({first} to {last})", "",
        f"Generated by `python -m backtest.daily` on {dt.date.today()}. See the module docstring for the rules.", "",
        f"Daily: every rule is checked at each close and trades at the next open; positions last a few sessions. "
        f"High win rate: at least {WIN_RATE:.0%} of trades make money after costs (5 bps a side for ETFs, 10 bps "
        f"for stocks). Target: {EXCESS:.0%} a year above SPY, which returned {spy_s['all']:.1%} a year "
        f"({spy_s['a']:.1%} in 2017-21, {spy_s['b']:.1%} in 2022-26). Idle cash earns T-bills.", "",
        "## Answer", "",
        f"{len(tested)} settings tested. {len(hit)} reached both targets over the whole period (win rate at least "
        f"{WIN_RATE:.0%} and at least {EXCESS:.0%} a year above SPY); {len(hit_both)} of them also beat SPY by "
        f"{EXCESS:.0%} or more in both 2017-21 and 2022-26.", "",
        "Each family's pick, chosen on 2017-21 only (best return among settings that won at least "
        f"{WIN_RATE:.0%} of their 2017-21 trades), then checked on 2022-26:", "",
        md(fmt(tbl.loc[[p for p in picks.values() if p], main]), "Strategy"), "",
    ]
    lines += [robustness(tested[tested["Family"] == "Fund dips"],
                         "Across all {n} settings that buy dips in the nine 3x funds at once"), ""]
    lev = tested[tested.index.str.endswith(("trade TQQQ", "trade UPRO"))]
    lines += [f"Across all {len(lev)} single-index dip settings traded in 3x funds (TQQQ, UPRO): median win rate "
              f"{lev['Win rate'].median():.0%}, median {lev['vs SPY'].median():+.1%} a year against SPY "
              f"({lev['2017-21 vs SPY'].median():+.1%} in 2017-21, {lev['2022-26 vs SPY'].median():+.1%} in 2022-26), "
              f"median max drawdown {lev['Max drawdown'].median():.0%}; "
              f"{(lev['2022-26 vs SPY'] >= EXCESS).mean():.0%} of them beat SPY by {EXCESS:.0%} or more in 2022-26. "
              f"The high win rate holds almost everywhere; beating SPY by {EXCESS:.0%} needs 3x leverage and depends "
              "on the exact setting.", ""]
    if len(hit_both):
        lines += [f"Settings that beat SPY by {EXCESS:.0%} or more in both 2017-21 and 2022-26 with a win rate of at "
                  f"least {WIN_RATE:.0%}:", "", md(fmt(hit_both[main]), "Strategy"), ""]
    if len(hit):
        lines += ["Settings that reached both targets over the whole period, best first:", "",
                  md(fmt(hit.sort_values("CAGR", ascending=False).head(15)[main]), "Strategy"), ""]
    shown = [p for p in picks.values() if p] + ([BALANCED] if BALANCED in tbl.index else [])
    lines += ["## The picks in detail", "",
              f"With '{BALANCED}': the fund setting with the smallest drawdown among those that beat SPY by "
              f"{EXCESS:.0%} or more in both halves. It was chosen after seeing the results (every IBS < 0.1 fund "
              "setting passed both halves), so the fund pick chosen on 2017-21 remains the honest test.", "",
              md(fmt(tbl.loc[shown, detail]).T, "Measure"), ""]
    if out.get("costs"):
        rows = {}
        for (label, bps), r in sorted(out["costs"].items()):
            rows[f"{label}, {bps:g} bps a side"] = r
        for label in {k for k, _ in out["costs"]}:
            rows[f"{label}, 5 bps a side"] = tbl.loc[label].to_dict()
        ct = pd.DataFrame(rows).T.sort_index()
        lines += ["Trading costs (5 bps a side is the base case; opening auctions in the smaller funds may cost more):",
                  "", md(fmt(ct[["CAGR", "vs SPY", "2017-21 vs SPY", "2022-26 vs SPY", "Win rate"]]), "Setting"), ""]
    for label in [p for p in (picks["Fund dips"], BALANCED) if p in trades]:
        t = trades[label]
        g = t.groupby("asset")["ret"]
        share = (g.sum() / t["ret"].sum()).sort_values(ascending=False)
        lines += [f"Where '{label}' made its money (share of summed trade returns, trades): " +
                  ", ".join(f"{a} {v:.0%} ({g.size()[a]})" for a, v in share.items()) + "."]
    lines += [""]
    years = pd.DataFrame({**{p: yearly(curves[p]) for p in shown}, "SPY": yearly(spy)})
    lines += ["## Calendar years", "", md(years.map(lambda v: f"{v:+.1%}"), "Year"), ""]
    for f, p in picks.items():
        if not p:
            continue
        t = trades[p].sort_values("ret").head(5)
        lines += [f"Worst trades, {f.lower()} pick: " + ", ".join(
            f"{r.asset} {r.entry.date()} to {r.exit.date()} {r.ret:+.1%}" for r in t.itertuples()) + "."]
    fam = tested.groupby("Family").agg(**{"Settings": ("CAGR", "size"),
                                          f"Win rate >= {WIN_RATE:.0%}": ("Win rate", lambda s: int((s >= WIN_RATE).sum())),
                                          "Median CAGR": ("CAGR", "median"), "Best CAGR": ("CAGR", "max"),
                                          "Median win rate": ("Win rate", "median")})
    fam["Median CAGR"] = fam["Median CAGR"].map(lambda v: f"{v:.1%}")
    fam["Best CAGR"] = fam["Best CAGR"].map(lambda v: f"{v:.1%}")
    fam["Median win rate"] = fam["Median win rate"].map(lambda v: f"{v:.0%}")
    ref = tbl[tbl["Family"] == "Reference"]
    lines += ["", "## Families", "", md(fam, "Family"), "",
              "For comparison (the trend rule trades rarely and wins less than half its trades, but its winners are "
              "large):", "", md(fmt(ref[main]), "Strategy"), "",
              "## Caveats", "",
              "- A high win rate is not low risk: the average loss is larger than the average win, and the worst trades "
              "come in fast sell-offs. With 3x funds a dip that keeps falling costs three times as much; nothing "
              "can be sold before the next open.",
              f"- {len(tested)} settings were tried, so the best whole-period numbers are flattered by selection. The "
              "picks were chosen on 2017-21 only; their 2022-26 columns are the honest test.",
              "- The nine 3x funds were chosen by today's trading volume among funds that exist today; 3x funds that "
              "closed are not in the test. Each of them can lose most of its value in a crash (they fell 70-90% in "
              "March 2020), and 2020's crash and rebound account for a large share of the fund results.",
              "- Survivorship bias: the stock data holds only companies listed today, so stocks that collapsed and "
              "were delisted (e.g. SVB in March 2023) are missing. Buying dips gains more from that bias than momentum "
              "does; the ETF results have no such bias.",
              "- Fills at the next open with fixed costs; no slippage model, no taxes. Ten years, mostly a rising market "
              "led by US tech, which suits buying dips in the Nasdaq."]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    out = run_all()
    text = report(out)
    (REPO_ROOT / "backtest" / "DAILY.md").write_text(text + "\n")
    print(text)
    log.info("done in %.0fs", out["seconds"])


if __name__ == "__main__":
    main()

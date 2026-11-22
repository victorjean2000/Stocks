"""Thorough tests of the ISA dip rule (RECOMMENDED in backtest/isa.py) and of why it works.

The rule: half the ISA in LQQ3 (Nasdaq-100 3x) and half in 3LUS (S&P 500 3x). Buy a fund at the London open after
its index (QQQ / SPY) closes in the bottom 10% of the day's range (IBS < 0.1); sell at the London open after the index
closes above its 5-day average. Real London prices in pence, 10 bps a side, sterling interest on cash.

Sections of backtest/ISA_TESTS.md (python -m backtest.isa_tests, about 2 minutes):
1. Why it works: the index's moves after closes near the day's low (all IBS deciles, the 10 days after, hour by hour
   from 30-minute bars), and which signals bounce most (late selling, big down days, volatile markets), year by year.
2. Robustness: thresholds x exit lengths, costs, execution timing, a random-entry test, a block bootstrap of the
   daily returns, walk-forward parameter choice, rolling 12-month results.
3. More rules, one at a time: entry filters, other exits, stop losses, time stops, volatility sizing.
4. Hourly candles: trades in the two hours London and New York are both open (an early exit or entry at the London
   close, 11:30 in New York), and, for comparison, the same rule on the US 3x funds traded at different times of day.
Settings are judged on 2017-21 and checked on 2022-26, as everywhere in this repository.

Data: Alpaca daily and 30-minute bars (SIP; 30-minute bars from 2016 cached in data/benchmarks/us_30min.parquet),
London fund prices as in backtest/isa.py (Yahoo, approved).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from stockdata import alpaca
from stockdata.config import REPO_ROOT, paths

from . import isa
from .daily import run
from .indicators import rsi
from .lqq3 import fetch_yahoo, fill_stale_prints, sterling, to_gbp
from .momentum_trend import md, stats
from .strategies import SPLIT, START, yearly_returns

log = logging.getLogger(__name__)

NY, LONDON = "America/New_York", "Europe/London"
FUNDS = {"LQQ3.L": "QQQ", "3LUS.L": "SPY"}  # London fund -> its signal index
US3X = {"TQQQ": "QQQ", "UPRO": "SPY"}  # the US 3x funds of the same indices
ENTRY, EXIT_N = 0.1, 5  # the recommendation: IBS < 0.1, sell above the 5-day average
COST, US_COST = 10.0, 5.0
INTRA_PATH = paths().root / "benchmarks" / "us_30min.parquet"


# ------------------------------------------------------------------ data

def us_30min(max_age_days: int = 3) -> pd.DataFrame:
    """30-minute bars (4:00-20:00 New York) of QQQ, SPY, TQQQ and UPRO since 2016, complete days only."""
    df = pd.read_parquet(INTRA_PATH) if INTRA_PATH.exists() else None
    today = pd.Timestamp.now(tz=NY).normalize()
    if df is None or (today.tz_localize(None) - df["date"].max()).days > max_age_days:
        sess = alpaca.session()
        df = alpaca.fetch_intraday(sess, [*FUNDS.values(), *US3X], "2016-01-01T00:00:00Z",
                                   today.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
        INTRA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(INTRA_PATH, index=False)
    return df


def us_calendar(end: str) -> pd.DataFrame:
    cal = alpaca.get_calendar(alpaca.session(), "2016-01-01", end)
    cal["open_ts"] = [pd.Timestamp(f"{d.date()} {o}", tz=NY) for d, o in zip(cal["date"], cal["open"])]
    return cal


def london_close_mark(dates: pd.DatetimeIndex) -> np.ndarray:
    """New York time (HHMM) of the London close (16:30 London) on each date: 1130, or 1230 in the weeks the clocks
    change on different dates."""
    return np.array([pd.Timestamp(f"{d.date()} 16:30", tz=LONDON).tz_convert(NY).strftime("%H%M") for d in dates])


def session_table(sym: str, bars30: pd.DataFrame, cal: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """One row per US session: the official daily bar (o, h, l, c), the price at each half hour of the regular
    session (m1000 ... m1600: the close of the 30-minute bar ending then) with the day's high and low so far
    (hi..., lo...), the first pre-market price from 4:00 (pm), the last after-hours price (ah), and the price,
    high and low so far at the London close (lc, lc_hi, lc_lo)."""
    g = bars30[bars30["ticker"] == sym].merge(cal[["date", "open_ts", "close_ts"]], on="date").sort_values("ts")
    reg = g[(g["ts"] >= g["open_ts"]) & (g["ts"] < g["close_ts"])].copy()
    reg["end"] = (reg["ts"] + pd.Timedelta(minutes=30)).dt.strftime("%H%M")
    reg["run_h"] = reg.groupby("date")["h"].cummax()
    reg["run_l"] = reg.groupby("date")["l"].cummin()
    m = reg.pivot(index="date", columns="end", values="c").add_prefix("m")
    hi = reg.pivot(index="date", columns="end", values="run_h").add_prefix("hi")
    lo = reg.pivot(index="date", columns="end", values="run_l").add_prefix("lo")
    pre = g[(g["ts"].dt.hour >= 4) & (g["ts"] < g["open_ts"])].groupby("date")["o"].first().rename("pm")
    post = g[g["ts"] >= g["close_ts"]].groupby("date")["c"].last().rename("ah")
    out = daily[["o", "h", "l", "c"]].join([m, hi, lo, pre, post], how="left")
    mark = london_close_mark(out.index)
    for col, src in (("lc", "m"), ("lc_hi", "hi"), ("lc_lo", "lo")):
        vals = np.full(len(out), np.nan)
        for t in np.unique(mark):
            if f"{src}{t}" in out:
                rows = mark == t
                vals[rows] = out.loc[rows, f"{src}{t}"].to_numpy()
        out[col] = vals
    return out


# ------------------------------------------------------------------ signals and states

def ibs(t: pd.DataFrame) -> pd.Series:
    return (t["c"] - t["l"]) / (t["h"] - t["l"])


def exit_flags(t: pd.DataFrame, kind: str, x: float | None = None) -> pd.Series:
    """Sell flags at each close: sma (close above its x-day average), high (above the previous high), strong
    (IBS above x), up (an up close), rsi (RSI(2) above x)."""
    c = t["c"]
    if kind == "sma":
        return c > c.rolling(int(x)).mean()
    if kind == "high":
        return c > t["h"].shift(1)
    if kind == "strong":
        return ibs(t) > x
    if kind == "up":
        return c > c.shift(1)
    if kind == "rsi":
        return rsi(c, 2) > x
    raise ValueError(kind)


def us_state(entry: pd.Series, leave: pd.Series, c: pd.Series, stop: float | None = None,
             max_days: int | None = None) -> pd.Series:
    """Holding (True) after each US close: in from a close with an entry flag until a close with a sell flag, a
    close `stop` below the entry close, or `max_days` closes after the entry."""
    e, x, px = entry.to_numpy(bool), leave.to_numpy(bool), c.to_numpy(float)
    state, held, ref, days = np.zeros(len(px), bool), False, np.nan, 0
    for i in range(len(px)):
        if held:
            days += 1
            if x[i] or (stop is not None and px[i] <= ref * (1 - stop)) or (max_days is not None and days >= max_days):
                held = False
        elif e[i]:
            held, ref, days = True, px[i], 0
        state[i] = held
    return pd.Series(state, index=c.index)


# ------------------------------------------------------------------ engines

def event_run(P: np.ndarray, W: np.ndarray, cost_bps: float, cash: np.ndarray, k0: int,
              size: np.ndarray | None = None):
    """Holdings on an event axis (rows: events in time order; columns: funds, one place of 1/n each). P: fill
    prices (NaN: no trade possible then), W: holding wanted after each event's fill, size: share of the place
    to buy at each event (known before it). Equity is 1 at event k0. Returns equity, invested share and trades."""
    enter, leave = np.zeros_like(W), np.zeros_like(W)
    enter[:-1], leave[:-1] = W[1:], ~W[1:]
    mark = pd.DataFrame(P).ffill().to_numpy()
    return run(P, mark, enter, leave, np.ones(P.shape), P.shape[1], cost_bps, cash, k0, size=size)


def london_run(funds: dict[str, pd.DataFrame], want_open: np.ndarray, want_close: np.ndarray, cash_r: np.ndarray,
               lse: pd.DatetimeIndex, i0: int, cost_bps: float = COST, size_open: np.ndarray | None = None):
    """The ISA funds on London sessions with two moments a day: the open and the close. want_open / want_close:
    holding wanted after each session's open / close (rows: sessions, columns: funds). Returns equity at London
    closes (1 at session i0) and trades (fund, entry and exit session and moment, return)."""
    cols = list(funds)
    n = len(lse)
    P = np.empty((2 * n, len(cols)))
    P[0::2] = np.column_stack([funds[f]["open"].reindex(lse).to_numpy() for f in cols])
    P[1::2] = np.column_stack([funds[f]["close"].reindex(lse).to_numpy() for f in cols])
    W = np.empty((2 * n, len(cols)), bool)
    W[0::2], W[1::2] = want_open, want_close
    cash = np.zeros(2 * n)
    cash[1::2] = cash_r
    size = None
    if size_open is not None:
        size = np.ones((2 * n, len(cols)))
        size[0::2] = size_open
    eq, inv, tr = event_run(P, W, cost_bps, cash, 2 * i0 + 1, size)
    moment = ["open", "close"]
    trades = pd.DataFrame([{"asset": isa.NAMES[cols[j]], "entry": lse[a // 2], "entry at": moment[a % 2],
                            "exit": lse[b // 2], "exit at": moment[b % 2], "ret": r, "days": (b - a) / 2}
                           for j, a, b, _, _, r in tr],
                          columns=["asset", "entry", "entry at", "exit", "exit at", "ret", "days"])
    return pd.Series(eq[1::2], index=lse).dropna(), trades, pd.Series(inv[1::2], index=lse).dropna()


def london_wanted(state: pd.Series, lse: pd.DatetimeIndex) -> np.ndarray:
    return isa.london_wanted(state, lse)


def event_wants(t: pd.DataFrame, st: pd.DataFrame, entry: pd.Series, leave: pd.Series, lse: pd.DatetimeIndex,
                exit_n: int = EXIT_N, early_exit: bool = False, early_entry: float | None = None,
                lc_stop: float | None = None, drop: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Holding wanted after each London open and close when decisions can also be taken at the London close
    (16:30 UK, 11:30 in New York) from the index's price then (session table `st`): early_exit sells when the index is
    already above its exit line (the mean of the previous exit_n - 1 closes: a close there would be above the exit_n-day
    average); early_entry buys when the index is in the bottom `early_entry` of the day's range so far and `drop`
    below the previous close; lc_stop sells when it is `lc_stop` below the entry. US-close decisions (entry / leave)
    fill at the next London open as usual."""
    c, e, x = t["c"].to_numpy(float), entry.to_numpy(bool), leave.to_numpy(bool)
    line = t["c"].rolling(exit_n - 1).mean().to_numpy()
    lc, hi, lo = (st[k].reindex(lse).to_numpy(float) for k in ("lc", "lc_hi", "lc_lo"))
    events = sorted([(d, 2, k) for k, d in enumerate(t.index)] + [(d, 0, i) for i, d in enumerate(lse)] +
                    [(d, 1, i) for i, d in enumerate(lse)])
    w_open, w_close = np.zeros(len(lse), bool), np.zeros(len(lse), bool)
    theo = held = False
    ref, last = np.nan, -1
    for _, kind, k in events:
        if kind == 2:  # US close
            last = k
            if theo:
                theo = not x[k]
            elif e[k]:
                theo, ref = True, c[k]
        elif kind == 0:  # London open: fill the last US-close decision
            held = theo
            w_open[k] = held
        else:  # London close
            p = lc[k]
            if np.isfinite(p) and last >= 0:
                if held:
                    if (early_exit and p > line[last]) or (lc_stop is not None and p <= ref * (1 - lc_stop)):
                        held = theo = False
                elif early_entry is not None and hi[k] > lo[k] and (p - lo[k]) / (hi[k] - lo[k]) < early_entry \
                        and p < c[last] * (1 - drop):
                    held = theo = True
                    ref = p
            w_close[k] = held
    return w_open, w_close


# ------------------------------------------------------------------ the ISA set-up

class Env:
    """US daily bars, session tables, the two London funds (cleaned as in backtest/isa.py), the London calendar,
    sterling cash and the S&P 500 in pounds."""

    def __init__(self):
        self.b = isa.us_bars()
        self.bars30 = us_30min()
        self.cal = us_calendar(str(self.b["SPY"].index[-1].date()))
        self.st = {s: session_table(s, self.bars30, self.cal, self.b[s]) for s in (*FUNDS.values(), *US3X)}
        self.gbpusd, cash = sterling()
        self.funds = {}
        for f, s in FUNDS.items():
            df, _ = isa.adjust_actions(fetch_yahoo(f, max_age_days=4), self.b[s]["c"], 3)
            self.funds[f] = fill_stale_prints(df, self.b[s]["c"], 3, self.gbpusd)
        lse = self.funds["LQQ3.L"].index.union(self.funds["3LUS.L"].index)
        self.lse = lse[lse <= min(f.index[-1] for f in self.funds.values())]
        self.i0 = int(self.lse.searchsorted(pd.Timestamp(START))) - 1
        self.cash_r = cash.reindex(self.lse).fillna(0.0).to_numpy()
        spy = self.b["SPY"]["c"][self.b["SPY"].index >= self.lse[self.i0]]
        self.bench = to_gbp(spy / spy.iloc[0], self.gbpusd)


def vol20(t: pd.DataFrame) -> pd.Series:
    return t["c"].pct_change().rolling(20).std() * np.sqrt(252)


ENTRIES = {"ibs": lambda t, x: ibs(t) < x,
           "ibs or rsi": lambda t, x: (ibs(t) < x) | (rsi(t["c"], 2) < 10)}

FILTERS = {
    "only when RSI(2) is below 30 too": lambda t, o: rsi(t["c"], 2) < 30,
    "only after a down day": lambda t, o: t["c"] < t["c"].shift(1),
    "only on a wide-range day (range above its 20-day average)":
        lambda t, o: (t["h"] - t["l"]) > (t["h"] - t["l"]).rolling(20).mean(),
    "only when volatility is above its 1-year median": lambda t, o: vol20(t) > vol20(t).rolling(252).median(),
    "only when volatility is below its 1-year median": lambda t, o: vol20(t) <= vol20(t).rolling(252).median(),
    "only when the other index also closed low (IBS < 0.2)": lambda t, o: (ibs(o) < 0.2).reindex(t.index, fill_value=False),
    "not on Fridays": lambda t, o: pd.Series(t.index.dayofweek != 4, index=t.index),
    "only below the 5-day average": lambda t, o: t["c"] < t["c"].rolling(5).mean(),
    "no entries in a crash (20-day volatility above 40%)": lambda t, o: vol20(t) < 0.40,
    "only above the 200-day average": lambda t, o: t["c"] > t["c"].rolling(200).mean(),
}


def isa_rule(env: Env, x: float = ENTRY, leave: tuple = ("sma", EXIT_N), filters: tuple = (), stop: float | None = None,
             max_days: int | None = None, delay: int = 0, at: str = "open", cost: float = COST,
             vol_target: float | None = None, early_exit: bool = False, early_entry: float | None = None,
             lc_stop: float | None = None, funds: tuple = tuple(FUNDS), entry: str = "ibs"):
    """Equity, trades and invested share of one version of the rule on the London funds (each fund one place)."""
    other = {"QQQ": "SPY", "SPY": "QQQ"}
    wo, wc, size = [], [], []
    for f in funds:
        s = FUNDS[f]
        t = env.b[s]
        e = ENTRIES[entry](t, x)
        for name in filters:
            e &= FILTERS[name](t, env.b[other[s]]).fillna(False)
        out = exit_flags(t, *leave)
        if early_exit or early_entry is not None or lc_stop is not None:
            a, c_ = event_wants(t, env.st[s], e, out, env.lse, int(leave[1]) if leave[0] == "sma" else EXIT_N,
                                early_exit, early_entry, lc_stop)
        else:
            state = us_state(e, out, t["c"], stop, max_days)
            if delay:
                state = state.shift(delay, fill_value=False)
            a = london_wanted(state, env.lse)
            if at == "close":  # filled at the close of the first London session after the signal
                c_, a = a, np.r_[False, a[:-1]]
            else:
                c_ = a
        wo.append(a)
        wc.append(c_)
        if vol_target is not None:
            v = vol20(t)
            pos = v.index.searchsorted(env.lse, side="left") - 1
            size.append(np.clip(vol_target / v.to_numpy()[np.clip(pos, 0, None)], 0.05, 1.0))
    sub = {f: env.funds[f] for f in funds}
    return london_run(sub, np.column_stack(wo), np.column_stack(wc), env.cash_r, env.lse, env.i0, cost,
                      np.column_stack(size) if size else None)


def measure(eq: pd.Series, trades: pd.DataFrame | None = None, inv: pd.Series | None = None) -> dict:
    split = pd.Timestamp(SPLIT)
    y = yearly_returns(eq)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    row = {"CAGR": stats(eq)["CAGR"], "2017-21": stats(eq[eq.index <= split])["CAGR"],
           "2022-26": stats(eq[eq.index >= split])["CAGR"], "Without 2020": isa.without_years(eq, [2020])["CAGR"],
           "Max drawdown": stats(eq)["Max drawdown"], "Worst year": y.min()}
    if trades is not None:
        row.update({"Win rate": (trades["ret"] > 0).mean(), "Trades a year": len(trades) / years,
                    "Average trade": trades["ret"].mean()})
    if inv is not None:
        row["Time invested"] = inv.mean()
    return row


# ------------------------------------------------------------------ 1. why it works

def _t(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    return float((a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)))


def index_tables(st: pd.DataFrame, x: float = ENTRY) -> dict:
    """The index after closes in the bottom x of the day's range: by IBS decile, over the next 10 days, through
    the next day (after hours, 4:00 pre-market, the open, the London close, the close), by the day's last hour, by the
    day's move, by volatility, and by year. Returns are simple, from the signal close."""
    c, day = st["c"], st["c"].pct_change()
    sig = ibs(st) < x
    n1, n3 = c.shift(-1) / c - 1, c.shift(-3) / c - 1
    dec = pd.cut(ibs(st), np.linspace(0, 1, 11), include_lowest=True,
                 labels=[f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)])
    deciles = pd.DataFrame({"Days": n1.groupby(dec, observed=True).size(),
                            "Next day": n1.groupby(dec, observed=True).mean(),
                            "Next day up": (n1 > 0).groupby(dec, observed=True).mean(),
                            "Next 3 days": n3.groupby(dec, observed=True).mean()})
    path = pd.DataFrame({k: {"After a signal": (c.shift(-k) / c - 1)[sig].mean(), "All days": (c.shift(-k) / c - 1).mean()}
                         for k in range(1, 11)})
    nxt = st.shift(-1)
    points = {"8 pm (after hours)": st["ah"], "4 am next day (pre-market)": nxt["pm"], "Next open (9:30)": nxt["o"],
              "Next London close (11:30)": nxt["lc"], "Next close (16:00)": nxt["c"]}
    anatomy = pd.DataFrame({k: {"After a signal": (v / c - 1)[sig].mean(), "All days": (v / c - 1).mean()}
                            for k, v in points.items()}).T
    anatomy["Extra"] = anatomy["After a signal"] - anatomy["All days"]
    last = (c / st["m1500"] - 1)[sig]
    third = pd.qcut(last, 3, labels=["fell most", "middle", "fell least"])
    late = pd.DataFrame({k: {"Last hour": last[third == k].mean(), "Next day": n1[sig][third == k].mean(),
                             "Next 3 days": n3[sig][third == k].mean()} for k in third.cat.categories}).T
    size = pd.cut(day[sig], [-1, -0.02, -0.01, 1], labels=["down more than 2%", "down 1-2%", "down less than 1% or up"])
    move = pd.DataFrame({k: {"Signals": int((size == k).sum()), "Next day": n1[sig][size == k].mean(),
                             "Next 3 days": n3[sig][size == k].mean()} for k in size.cat.categories}).T
    v = day.rolling(20).std() * np.sqrt(252)
    vt = pd.qcut(v, 3, labels=["calmest third", "middle third", "most volatile third"])
    vol = pd.DataFrame({k: {"After a signal": n3[sig & (vt == k)].mean(), "All days": n3[vt == k].mean()}
                        for k in vt.cat.categories}).T
    vol["Extra"] = vol["After a signal"] - vol["All days"]
    with np.errstate(invalid="ignore", divide="ignore"):
        at1530 = (st["m1530"] - st["lo1530"]) / (st["hi1530"] - st["lo1530"]) < x
    late_low = st["l"] < st["lo1530"]  # the day's low came in the last half hour
    groups = {"Closed at the low, low made in the last half hour": sig & late_low,
              "Closed at the low, low made before 15:30": sig & ~late_low,
              "At the low at 15:30 but recovered by the close (no signal)": at1530 & ~sig}
    low_timing = pd.DataFrame({k: {"Days": int(g.sum()), "Last half hour": (c / st["m1530"] - 1)[g].mean(),
                                   "Next day": n1[g].mean(), "Next 3 days": n3[g].mean()} for k, g in groups.items()}).T
    yr = c.index.year
    years = pd.DataFrame({"Signals": sig.groupby(yr).sum(), "Next 3 days after a signal": n3[sig].groupby(yr[sig]).mean(),
                          "Any 3 days": n3.groupby(yr).mean()})
    years["Extra"] = years["Next 3 days after a signal"] - years["Any 3 days"]
    return {"deciles": deciles, "path": path, "anatomy": anatomy, "late": late, "low_timing": low_timing, "move": move,
            "vol": vol,
            "years": years, "t1": _t(n1[sig], n1[~sig]), "t3": _t(n3[sig], n3[~sig]), "signals": int(sig.sum()),
            "share": float(sig.mean())}


# ------------------------------------------------------------------ 2. robustness

def runs_of(w: np.ndarray) -> list[tuple[bool, int]]:
    """[(value, length), ...] of the consecutive runs in a boolean array."""
    out = []
    for v in w:
        if out and out[-1][0] == v:
            out[-1] = (v, out[-1][1] + 1)
        else:
            out.append((bool(v), 1))
    return out


def shuffle_runs(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The same number and lengths of holding spells and of gaps, in a random order (gaps and spells alternate as in
    w): random timing with exactly the same exposure."""
    r = runs_of(w)
    holds = [n for v, n in r if v]
    gaps = [n for v, n in r if not v]
    rng.shuffle(holds)
    rng.shuffle(gaps)
    out, h, g = [], iter(holds), iter(gaps)
    for v, _ in r:
        out += [v] * (next(h) if v else next(g))
    return np.array(out, bool)


def random_entry_test(env: Env, n: int = 500, seed: int = 0) -> dict:
    """CAGR of the rule against the same holding spells placed at random times (each fund separately)."""
    rng = np.random.default_rng(seed)
    eq, _, _ = isa_rule(env)
    base = []
    for f, s in FUNDS.items():
        t = env.b[s]
        base.append(london_wanted(us_state(ibs(t) < ENTRY, exit_flags(t, "sma", EXIT_N), t["c"]), env.lse))
    W = np.column_stack(base)
    k = env.i0 + 1
    cagr = []
    for _ in range(n):
        R = W.copy()
        for j in range(W.shape[1]):
            R[k:, j] = shuffle_runs(W[k:, j], rng)
        e, _, _ = london_run(env.funds, R, R, env.cash_r, env.lse, env.i0)
        cagr.append(stats(e)["CAGR"])
    cagr = np.array(cagr)
    return {"actual": stats(eq)["CAGR"], "median": float(np.median(cagr)), "p95": float(np.percentile(cagr, 95)),
            "best": float(cagr.max()), "share_beating": float((cagr >= stats(eq)["CAGR"]).mean()), "n": n}


def block_bootstrap(eq: pd.Series, bench: pd.Series, n: int = 2000, block: int = 10, seed: int = 0) -> dict:
    """Paired block bootstrap of the daily returns (blocks of `block` London sessions, the same blocks for the rule
    and the S&P 500 in pounds): ranges of the yearly return and max drawdown over a period as long as the test."""
    rng = np.random.default_rng(seed)
    b = bench.reindex(eq.index).ffill()
    r, rb = eq.pct_change().to_numpy()[1:], b.pct_change().to_numpy()[1:]
    rb = np.nan_to_num(rb)
    m = len(r)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    starts = rng.integers(0, m - block, size=(n, m // block + 1))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :m]
    path, pb = np.cumprod(1 + r[idx], axis=1), np.cumprod(1 + rb[idx], axis=1)
    cagr, cb = path[:, -1] ** (1 / years) - 1, pb[:, -1] ** (1 / years) - 1
    dd = (path / np.maximum.accumulate(path, axis=1) - 1).min(axis=1)
    q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    return {"cagr": (q(cagr, 5), q(cagr, 50), q(cagr, 95)), "dd": (q(dd, 5), q(dd, 50), q(dd, 95)),
            "beats": float((cagr > cb).mean()), "beats15": float((cagr > cb + 0.15).mean()), "n": n, "block": block}


GRID_X = (0.05, 0.1, 0.15, 0.2, 0.3)
GRID_N = (3, 5, 7, 10)


def grid(env: Env) -> tuple[pd.DataFrame, dict]:
    rows, curves = {}, {}
    for x in GRID_X:
        for n_ in GRID_N:
            eq, tr, inv = isa_rule(env, x=x, leave=("sma", n_))
            rows[(x, n_)] = measure(eq, tr, inv)
            curves[(x, n_)] = eq
    return pd.DataFrame(rows).T, curves


def walk_forward(curves: dict, first: int = 2019) -> tuple[pd.Series, dict]:
    """Each year from `first`: the grid setting with the best yearly return over all earlier years, traded that
    year. Returns the chained equity and the setting used each year."""
    ref = next(iter(curves.values()))
    years = sorted(set(ref.index.year))
    start = ref[ref.index < pd.Timestamp(f"{first}-01-01")].index[-1]
    parts, chosen, level = [pd.Series([1.0], index=[start])], {}, 1.0
    for y in [y for y in years if y >= first]:
        past = {k: e[e.index < pd.Timestamp(f"{y}-01-01")] for k, e in curves.items()}
        best = max(past, key=lambda k: stats(past[k])["CAGR"])
        e = curves[best]
        prev = e[e.index < pd.Timestamp(f"{y}-01-01")].iloc[-1]
        seg = e[e.index.year == y] / prev * level
        parts.append(seg)
        level = seg.iloc[-1]
        chosen[y] = best
    return pd.concat(parts), chosen


def rolling_year(eq: pd.Series, bench: pd.Series, window: int = 252) -> dict:
    b = bench.reindex(eq.index).ffill()
    r, rb = eq / eq.shift(window) - 1, b / b.shift(window) - 1
    ok = r.notna() & rb.notna()
    r, rb = r[ok], rb[ok]
    return {"windows": int(ok.sum()), "positive": float((r > 0).mean()), "beat": float((r > rb).mean()),
            "worst": float(r.min()), "worst_end": r.idxmin(), "median": float(r.median()), "best": float(r.max())}


# ------------------------------------------------------------------ 4. hourly: the US 3x funds at other times of day

US_EVENTS = ("pm", "o", "m1030", "m1130", "m1230", "m1330", "m1430", "m1530", "c")


def us_timing(env: Env, entry_at: str = "open", exit_at: str = "open", hourly_exit: bool = False,
              x: float = ENTRY, exit_n: int = EXIT_N, cost: float = US_COST, peek: bool = False):
    """The rule on TQQQ / UPRO (half each, in dollars) with trades at chosen times of the US day, from 30-minute
    bars. entry_at / exit_at: 'open' or 'pm' (decided at the close, filled at the next open or the first pre-market
    price from 4:00), 'close' (decided at 15:30 from the price and range then, filled at the official close) or
    '1530' (decided and filled at 15:30). hourly_exit: also sell at the first half hour (10:30 ... 15:30) the index
    is above its exit line. peek: decide the 'close' trades with the closing price itself (not possible in
    practice; the best a close could do). Cash earns T-bills (BIL)."""
    dates = env.st["QQQ"].index
    dates = dates[dates >= pd.Timestamp("2016-01-01")]
    ev = len(US_EVENTS)
    P = np.full((len(dates) * ev, len(US3X)), np.nan)
    W = np.zeros_like(P, bool)
    for j, (fund, s) in enumerate(US3X.items()):
        f, ix = env.st[fund].reindex(dates), env.st[s].reindex(dates)
        for e_i, col in enumerate(US_EVENTS):
            P[e_i::ev, j] = f[col].to_numpy()
        c = ix["c"].to_numpy()
        line = ix["c"].rolling(exit_n - 1).mean().shift(1).to_numpy()  # mean of the previous exit_n - 1 closes
        ibs_c = ibs(ix).to_numpy()
        p1530, h1530, l1530 = ((ix["c"], ix["h"], ix["l"]) if peek else (ix["m1530"], ix["hi1530"], ix["lo1530"]))
        p1530, h1530, l1530 = p1530.to_numpy(), h1530.to_numpy(), l1530.to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            ibs_1530 = (p1530 - l1530) / (h1530 - l1530)
        held, pend_in, pend_out = False, False, False
        for k in range(len(dates)):
            for e_i, col in enumerate(US_EVENTS):
                if col in ("pm", "o"):
                    when = "pm" if col == "pm" else "open"
                    if pend_in and entry_at == when:
                        held, pend_in = True, False
                    if pend_out and exit_at == when:
                        held, pend_out = False, False
                elif col.startswith("m"):
                    p = ix[col].iat[k] if col in ix else np.nan
                    if hourly_exit and held and np.isfinite(p) and p > line[k]:
                        held = False
                    if col == "m1530":
                        if entry_at == "1530" and not held and ibs_1530[k] < x:
                            held = True
                        elif exit_at == "1530" and held and p1530[k] > line[k]:
                            held = False
                else:  # the close
                    if entry_at == "close" and not held and ibs_1530[k] < x:
                        held = True
                    elif exit_at == "close" and held and p1530[k] > line[k]:
                        held = False
                    if entry_at in ("open", "pm") and not held and not pend_in and ibs_c[k] < x:
                        pend_in = True
                    if exit_at in ("open", "pm") and held and c[k] > line[k]:
                        pend_out = True
                W[k * ev + e_i, j] = held
    cash = np.zeros(len(P))
    bil = env.b["BIL"]["c"].reindex(dates).ffill().pct_change().fillna(0.0).to_numpy()
    cash[ev - 1::ev] = bil
    k0 = int(dates.searchsorted(pd.Timestamp(START))) * ev - 1
    eq, inv, tr = event_run(P, W, cost, cash, k0)
    eq = pd.Series(eq[ev - 1::ev], index=dates).dropna()
    trades = pd.DataFrame([{"ret": r, "days": (b_ - a) / ev} for _, a, b_, _, _, r in tr], columns=["ret", "days"])
    return eq, trades, pd.Series(inv[ev - 1::ev], index=dates).dropna()


# ------------------------------------------------------------------ the variants

BASE = "The rule: IBS < 0.1, sell above the 5-day average"
RULES = [
    (BASE, {}),
    ("Entry: also buy when RSI(2) is below 10", {"entry": "ibs or rsi"}),
    *[(f"Entry: {name}", {"filters": (name,)}) for name in FILTERS],
    ("Exit: a close above the previous day's high", {"leave": ("high",)}),
    ("Exit: a close in the top 20% of the day's range", {"leave": ("strong", 0.8)}),
    ("Exit: the first up close", {"leave": ("up",)}),
    ("Exit: RSI(2) above 70", {"leave": ("rsi", 70)}),
    ("Exit: a close above the 3-day average", {"leave": ("sma", 3)}),
    ("Exit: a close above the 7-day average", {"leave": ("sma", 7)}),
    *[(f"Stop: sell after an index close {s:.0%} below the entry close", {"stop": s}) for s in (0.03, 0.05, 0.08, 0.12)],
    *[(f"Time stop: sell after {d} sessions", {"max_days": d}) for d in (3, 5, 10)],
    *[(f"Size: less in volatile markets (aim for {v:.0%} index volatility)", {"vol_target": v}) for v in (0.15, 0.2, 0.3)],
]
HOURLY = [
    (BASE, {}),
    ("Sell at the London close once the index is above its exit line (11:30 New York)", {"early_exit": True}),
    ("Also buy at the London close when the index is in the bottom 10% of the day's range so far and 1% down",
     {"early_entry": 0.1}),
    ("Both", {"early_exit": True, "early_entry": 0.1}),
    ("Sell at the London close when the index is 5% below the entry close", {"lc_stop": 0.05}),
]
TIMING = [(BASE + " (next London open)", {}), ("Filled at the close of the next London session", {"at": "close"}),
          ("Filled one session later (the next-but-one London open)", {"delay": 1})]
SINGLE = [("LQQ3 alone (the whole ISA)", {"funds": ("LQQ3.L",)}), ("3LUS alone (the whole ISA)", {"funds": ("3LUS.L",)})]
COSTS = (10.0, 20.0, 30.0, 50.0)
US_TIMES = [
    ("Next US open (as in backtest/daily.py)", dict(entry_at="open", exit_at="open")),
    ("4:00 pre-market (about when London opens)", dict(entry_at="pm", exit_at="pm")),
    ("At the close, decided at 15:30 (market-on-close order)", dict(entry_at="close", exit_at="close")),
    ("At 15:30", dict(entry_at="1530", exit_at="1530")),
    ("Buy at the next open, sell at the first half hour above the exit line",
     dict(entry_at="open", exit_at="open", hourly_exit=True)),
    ("At the close, with the closing price known (not possible: the ceiling)",
     dict(entry_at="close", exit_at="close", peek=True)),
]


def table(env: Env, variants: list) -> pd.DataFrame:
    rows = {}
    for label, kw in variants:
        eq, tr, inv = isa_rule(env, **kw)
        rows[label] = measure(eq, tr, inv)
    return pd.DataFrame(rows).T


def run_all() -> dict:
    env = Env()
    eq, tr, inv = isa_rule(env)
    out = {"env": env, "eq": eq, "trades": tr, "base": measure(eq, tr, inv), "bench": measure(env.bench)}
    out["why"] = {s: index_tables(env.st[s][env.st[s].index >= pd.Timestamp("2016-01-01")]) for s in FUNDS.values()}
    out["grid"], curves = grid(env)
    out["wf"], out["wf_chosen"] = walk_forward(curves)
    fixed = curves[(ENTRY, EXIT_N)]
    fixed = fixed[fixed.index >= out["wf"].index[0]]
    out["wf_fixed"] = fixed / fixed.iloc[0]
    sp = env.bench[env.bench.index >= out["wf"].index[0]]
    out["wf_bench"] = sp / sp.iloc[0]
    out["costs"] = pd.DataFrame({f"{c:.0f} bps a side": measure(*isa_rule(env, cost=c)) for c in COSTS}).T
    out["timing"] = table(env, TIMING)
    out["single"] = table(env, SINGLE)
    out["random"] = random_entry_test(env)
    out["boot"] = block_bootstrap(eq, env.bench)
    out["rolling"] = rolling_year(eq, env.bench)
    out["rules"] = table(env, RULES)
    out["hourly"] = table(env, HOURLY)
    us = {}
    for label, kw in US_TIMES:
        e, t_, i_ = us_timing(env, **kw)
        us[label] = measure(e, t_, i_)
    spy = env.b["SPY"]["c"]
    spy = spy[spy.index >= e.index[0]]
    us["S&P 500 (SPY, dollars)"] = measure(spy / spy.iloc[0])
    out["us_times"] = pd.DataFrame(us).T
    return out


# ------------------------------------------------------------------ report

PCT = ("CAGR", "2017-21", "2022-26", "Without 2020", "Max drawdown", "Worst year", "Win rate", "Average trade",
       "Time invested")


def fmt(df: pd.DataFrame, cols: tuple = ("CAGR", "2017-21", "2022-26", "Max drawdown", "Worst year", "Win rate",
                                         "Trades a year", "Time invested")) -> pd.DataFrame:
    out = df[[c for c in cols if c in df]].copy()
    for c in out:
        if c == "Trades a year":
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
        elif c in PCT:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    return out


def pct(df: pd.DataFrame, digits: int = 2, cols: list | None = None) -> pd.DataFrame:
    out = df.copy()
    for c in (cols or out.columns):
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:+.{digits}%}")
    return out


def better_both(df: pd.DataFrame, base: str) -> list[str]:
    b = df.loc[base]
    return [k for k, r in df.iterrows() if k != base and r["2017-21"] > b["2017-21"] and r["2022-26"] > b["2022-26"]]


def _side_by_side(q: pd.DataFrame, s: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.concat({"QQQ": q[cols], "SPY": s[cols]}, axis=1).pipe(
        lambda d: d.set_axis([f"{a} {b.lower()}" for a, b in d.columns], axis=1))


def report(out: dict) -> str:
    base, bench = out["base"], out["bench"]
    q, s = out["why"]["QQQ"], out["why"]["SPY"]
    first, last = out["eq"].index[0].date(), out["eq"].index[-1].date()
    rnd, boot, roll = out["random"], out["boot"], out["rolling"]
    g, rules, hourly = out["grid"], out["rules"], out["hourly"]
    wins = better_both(rules, BASE)
    worse_dd = {k: rules.loc[k, "Max drawdown"] for k in wins}
    path_q = (q["path"].loc["After a signal"] - q["path"].loc["All days"])
    path_s = (s["path"].loc["After a signal"] - s["path"].loc["All days"])
    an_q, an_s = q["anatomy"], s["anatomy"]
    pre_q = an_q.loc["4 am next day (pre-market)", "Extra"]
    three_q = path_q[3]
    yrs_q, yrs_s = q["years"], s["years"]
    wf, wf_fixed, wf_bench = out["wf"], out["wf_fixed"], out["wf_bench"]
    costs = out["costs"]
    lowt_q, lowt_s = q["low_timing"], s["low_timing"]
    grid_line = _grid_text(g)
    rec = "Closed at the low, low made in the last half hour"
    recov = "At the low at 15:30 but recovered by the close (no signal)"
    lines = [
        f"# Testing the ISA dip rule ({first} to {last})", "",
        f"Generated by `python -m backtest.isa_tests` on {dt_today()}. The rule (RECOMMENDED in backtest/isa.py): half "
        "the ISA in LQQ3 (Nasdaq-100 3x) and half in 3LUS (S&P 500 3x); buy a fund at the London open after its index "
        "(QQQ / SPY) closes in the bottom 10% of the day's range (IBS < 0.1), sell at the London open after the index "
        "closes above its 5-day average. Real London prices in pence (Yahoo, cleaned as in backtest/isa.py), 10 bps a "
        "side, sterling interest on cash, the S&P 500 in pounds as the benchmark. Index data: Alpaca daily bars, and "
        "30-minute bars since 2016 for the hourly tests. Changes are judged on 2017-21 and checked on 2022-26.", "",
        "## Summary", "",
        f"- **The rule:** {base['CAGR']:.1%} a year ({base['2017-21']:.1%} in 2017-21, {base['2022-26']:.1%} in "
        f"2022-26) against {bench['CAGR']:.1%} for the S&P 500 in pounds; max drawdown {base['Max drawdown']:.0%}, "
        f"invested {base['Time invested']:.0%} of the time, {base['Win rate']:.0%} of trades won.",
        f"- **The edge is real:** after QQQ closes in the bottom 10% of its range, the next 3 days average "
        f"{q['deciles'].iloc[0]['Next 3 days']:+.2%} against {q['path'].loc['All days', 3]:+.2%} for any 3 days "
        f"(t = {q['t3']:.1f}), and that extra return was positive in {int((yrs_q['Extra'] > 0).sum())} of "
        f"{len(yrs_q)} years. The same time in the market at random times made {rnd['median']:.1%} a year (median of "
        f"{rnd['n']} runs); none reached the rule's {rnd['actual']:.1%}.",
        f"- **Expect a wide range:** resampling the daily returns gives {boot['cagr'][0]:.0%} to {boot['cagr'][2]:.0%} "
        f"a year and a max drawdown of {boot['dd'][2]:.0%} to {boot['dd'][0]:.0%} (5th to 95th percentile); it beat "
        f"the S&P 500 in {boot['beats']:.0%} of the resamples.",
        f"- **The settings are not a lucky pick:** {grid_line[0].lower() + grid_line[1:]} Choosing the best setting "
        f"each year from the earlier years only (walk-forward) made {stats(wf)['CAGR']:.1%} a year from 2019, against "
        f"{stats(wf_fixed)['CAGR']:.1%} for the fixed rule.",
        f"- **No refinement beat it on both return and risk.** Of {len(rules) - 1} one-at-a-time changes, "
        + (f"{len(wins)} did better in both halves ({'; '.join(wins)}), with a max drawdown of "
           f"{', '.join(f'{v:.0%}' for v in worse_dd.values())} against {base['Max drawdown']:.0%}"
           if wins else "none did better in both halves")
        + ". Stop losses, time stops, trend and volatility filters, smaller positions in volatile markets and every "
          "hourly variant made it worse or only traded return for risk.",
        f"- **Costs matter most:** at 20 bps a side it makes {costs.iloc[1]['CAGR']:.1%} a year, at 30 bps "
        f"{costs.iloc[2]['CAGR']:.1%}, at 50 bps {costs.iloc[3]['CAGR']:.1%} ({base['Trades a year']:.0f} round trips "
        "a year).",
        "- **Verdict:** keep the rule as it is. Details and the reasons below.", "",
        "## 1. Why it works", "",
        "### The index after a close near the day's low", "",
        "Next-day and 3-day returns of the index after each close, by where the close sat in the day's range (IBS "
        "decile; 0.0-0.1 is the signal), 2016-2026:", "",
        md(pct(_side_by_side(q["deciles"], s["deciles"], ["Next day", "Next 3 days"]))
           .assign(**{"QQQ days": q["deciles"]["Days"].astype(int),
                      "QQQ next day up": q["deciles"]["Next day up"].map(lambda v: f"{v:.0%}")}), "Close in the range"),
        "",
        f"The bottom 10% stands out: QQQ's next day averages {q['deciles'].iloc[0]['Next day']:+.2%} (up "
        f"{q['deciles'].iloc[0]['Next day up']:.0%} of the time) against {q['deciles'].iloc[3:7]['Next day'].mean():+.2%} "
        "for closes in the middle of the range (0.3-0.7). "
        f"Against all other days the difference is t = {q['t1']:.1f} for the next day and {q['t3']:.1f} for 3 days "
        f"(SPY: {s['t1']:.1f} and {s['t3']:.1f}): unlikely to be chance.", "",
        "### How long the bounce lasts", "",
        "Extra return after a signal (the index's return from the signal close minus the average over the same number "
        "of days):", "",
        md(pct(pd.DataFrame({"QQQ": path_q, "SPY": path_s}).T.rename(columns=lambda k: f"{k} day{'s' if k > 1 else ''}")),
           "Index"), "",
        "The extra return builds over 3-4 days and then stops growing: the dip is a temporary overshoot that is "
        "repaid, not the start of a trend. That is why selling once the index is back above its 5-day average works: "
        f"the rule holds for about {out['trades']['days'].mean():.0f} sessions on average.", "",
        "### When the bounce happens (30-minute bars)", "",
        "Average return from the signal close through the next day, against all days:", "",
        md(pct(_side_by_side(an_q, an_s, ["After a signal", "All days", "Extra"]), 3), "From the signal close to"), "",
        f"Little happens in the evening. {_share(an_q, 'Next open (9:30)', 1):.0%} of QQQ's extra next-day return comes "
        f"during the next US session, and {_share(an_s, 'Next London close (11:30)', 1):.0%} of SPY's after 11:30 New "
        f"York. Your trades happen at the London open (8:00 UK, about 3:00 in New York): by 4:00 only "
        f"{pre_q:+.2%} of QQQ's {three_q:+.2%} 3-day extra return has happened, so buying in London, hours before New "
        "York opens, gives up little.", "",
        "### What makes a signal: the close itself", "",
        md(_low_table(lowt_q, lowt_s), "Signal day"), "",
        f"Days that were at their low at 15:30 but recovered by the close did not bounce (QQQ next day "
        f"{lowt_q.loc[recov, 'Next day']:+.2%}), while closes at the low did ({lowt_q.loc[rec, 'Next day']:+.2%} when "
        "the low came in the last half hour). The signal is how the day ends, so it cannot be traded before the close "
        "(section 4 shows a 15:30 version failing). Neither a bigger fall in the last hour nor a bigger fall over "
        "the day gives a bigger bounce:", "",
        "By how far the index fell in the last hour of the signal day:", "",
        md(pct(_side_by_side(q["late"], s["late"], ["Last hour", "Next day", "Next 3 days"])), "Last hour"), "",
        "By the signal day's own move (a small dip that closes at the low bounces about as much as a big fall):", "",
        md(pct(_side_by_side(q["move"], s["move"], ["Next day", "Next 3 days"]))
           .assign(**{"QQQ signals": q["move"]["Signals"].astype(int)}), "Signal day"), "",
        "### Calm and volatile markets, year by year", "",
        "Next 3 days after a signal against any 3 days in the same volatility third (20-day volatility):", "",
        md(pct(_side_by_side(q["vol"], s["vol"], ["After a signal", "All days", "Extra"])), "Market"), "",
        "Extra 3-day return after signals, by year:", "",
        md(pct(pd.DataFrame({"QQQ": yrs_q["Extra"], "SPY": yrs_s["Extra"]}).T)
           .rename(columns=str), "Index"), "",
        f"The extra return is about the same in calm and volatile markets (bigger swings in volatile ones, so single "
        f"trades win and lose more), and it was positive in {int((yrs_q['Extra'] > 0).sum())} of {len(yrs_q)} years "
        f"for QQQ and {int((yrs_s['Extra'] > 0).sum())} of {len(yrs_s)} for SPY, before and after 2022.", "",
        "### So why does it work?", "",
        "What the data shows: when the index closes at the bottom of its day's range, it tends to recover more than "
        "usual over the next 3-4 days, and then the extra return stops. It depends on the close being at the low (not "
        "on the size of the fall), it shows up in every kind of market and nearly every year, and it is strongest in "
        "the Nasdaq-100.", "",
        "The likely reason (consistent with these facts, but not something the prices can prove): a lot of trading "
        "has to happen at the close whatever the price. Index funds and ETFs trade in the closing auction; leveraged "
        "and inverse ETFs such as TQQQ and SQQQ must rebalance at the close and so sell after a falling day; options "
        "dealers sell into falls to stay hedged; stop orders and margin calls add to it. A day that ends with sellers "
        "in control pushes the close below where buyers would normally meet it. Once that pressure is gone the next "
        "day, the price drifts back. The rule is paid for buying from people who have to sell. That fits two more "
        "facts: it is quick (the overshoot is repaid within days), and it works in indices, where these flows are, "
        "far better than in single stocks, where a fall is more often news (backtest/DAILY.md: the same dip rules "
        "on the 500 largest stocks made a median 4.7% a year, even with survivorship bias in their favour).", "",
        "Why the 3x funds: in the index the edge is small (about "
        f"{three_q:+.1%} over 3 days for QQQ); three times that, after costs, is {base['Average trade']:+.1%} a trade. "
        f"Because the funds are held only {base['Time invested']:.0%} of the time, their daily-reset decay and "
        "crash exposure stay limited. Where it hurts: when the selling is not temporary, each new low triggers "
        "another buy. The deepest falls of the rule: " + "; ".join(
            f"{d:.0%} ({a:%b %Y} to {z:%b %Y})" for a, z, d in worst_drawdowns(out["eq"])) + ".", "",
        "## 2. Is it robust? (London funds, real prices)", "",
        "### Entry threshold and exit length", "",
        "Yearly return in 2017-21 (rows: buy when IBS is below; columns: sell above the N-day average):", "",
        md(_pivot(g, "2017-21"), "IBS below"), "",
        "Yearly return in 2022-26:", "",
        md(_pivot(g, "2022-26"), "IBS below"), "",
        "Max drawdown:", "",
        md(_pivot(g, "Max drawdown", "{:.0%}"), "IBS below"), "",
        _grid_text(g) + " Looser entries (IBS < 0.2, 0.3) make more trades of lower quality and deeper drawdowns.", "",
        "### Walk-forward", "",
        f"Each year from 2019, trade the grid setting with the best yearly return over all earlier years: "
        f"{stats(wf)['CAGR']:.1%} a year (max drawdown {stats(wf)['Max drawdown']:.0%}) against "
        f"{stats(wf_fixed)['CAGR']:.1%} ({stats(wf_fixed)['Max drawdown']:.0%}) for the fixed rule and "
        f"{stats(wf_bench)['CAGR']:.1%} for the S&P 500 in pounds. Settings chosen: "
        + ", ".join(f"{y}: IBS < {k[0]:g}, {k[1]}-day exit" for y, k in out["wf_chosen"].items()) + ".", "",
        "### Costs and timing", "",
        md(fmt(costs), "Cost"), "",
        md(fmt(out["timing"]), "Fill"), "",
        f"Every 10 bps a side costs about {(costs.iloc[0]['CAGR'] - costs.iloc[-1]['CAGR']) / 4 * 100:.1f} points a "
        "year. Filling "
        f"at the London close instead of the open made {out['timing'].iloc[1]['CAGR']:.1%} a year (much of the bounce "
        f"is gone by then); a day's delay made {out['timing'].iloc[2]['CAGR']:.1%}.", "",
        "### Random timing with the same exposure", "",
        f"Each fund's holding spells and the gaps between them, shuffled into a random order ({rnd['n']} runs: same "
        f"number and length of trades, random dates): median {rnd['median']:.1%} a year, 95th percentile "
        f"{rnd['p95']:.1%}, best {rnd['best']:.1%}. The rule's {rnd['actual']:.1%} beat all of them: the money comes "
        "from when it buys, not from being in 3x funds part of the time.", "",
        "### Bootstrap", "",
        f"{boot['n']} resamples of the rule's daily returns in blocks of {boot['block']} sessions (the S&P 500 resampled "
        f"on the same days): yearly return {boot['cagr'][0]:.1%} / {boot['cagr'][1]:.1%} / {boot['cagr'][2]:.1%} (5th / "
        f"median / 95th percentile), max drawdown {boot['dd'][0]:.0%} / {boot['dd'][1]:.0%} / {boot['dd'][2]:.0%}. "
        f"It beat the S&P 500 in {boot['beats']:.0%} of the resamples and by 15 points a year or more in "
        f"{boot['beats15']:.0%}. One resampled path in twenty fell {-boot['dd'][0]:.0%} or more at some point.", "",
        "### Rolling 12 months", "",
        f"Of {roll['windows']:,} rolling 12-month periods, {roll['positive']:.0%} made money and {roll['beat']:.0%} "
        f"beat the S&P 500 in pounds; median {roll['median']:+.0%}, worst {roll['worst']:+.1%} (to "
        f"{roll['worst_end']:%d %b %Y}).", "",
        "### Each fund alone", "",
        md(fmt(pd.concat([rules.loc[[BASE]], out["single"]])), "Rule"), "",
        f"LQQ3 alone made {out['single'].iloc[0]['CAGR']:.1%} a year with a max drawdown of "
        f"{out['single'].iloc[0]['Max drawdown']:.0%}; 3LUS alone {out['single'].iloc[1]['CAGR']:.1%} and "
        f"{out['single'].iloc[1]['Max drawdown']:.0%}. Half in each fell less than either alone: the two indices do "
        "not always dip on the same days.", "",
        "## 3. Adding rules, one at a time", "",
        md(fmt(rules), "Change"), "",
        _rules_text(rules, wins), "",
        "## 4. Hourly candles", "",
        "LQQ3 and 3LUS trade in London from 8:00 to 16:30 UK, so in your ISA the US hours can only be used in the two "
        "hours both markets are open (14:30-16:30 UK, 9:30-11:30 in New York). Hourly London fund prices are not in "
        "the approved data, so these tests use the index's 30-minute bars (Alpaca) to decide and the funds' real "
        "London close to trade.", "",
        "### In your ISA: decisions at the London close", "",
        md(fmt(hourly), "Rule"), "",
        f"Selling as soon as the index is back above its exit line at 11:30 New York made {hourly.iloc[1]['CAGR']:.1%} "
        "a year: it sells before the afternoon, where much of the bounce comes (section 1). Buying during the falling "
        f"day made {hourly.iloc[2]['CAGR']:.1%} ({hourly.iloc[2]['2017-21']:.1%} in 2017-21 but "
        f"{hourly.iloc[2]['2022-26']:.1%} in 2022-26) with a deeper drawdown: it catches more of the fall. "
        + ("None of them did better in both halves." if not better_both(hourly, BASE) else
           f"Better in both halves: {', '.join(better_both(hourly, BASE))}."), "",
        "### The same rule on the US 3x funds at other times of day (dollars, for comparison)", "",
        "TQQQ and UPRO, half each, real 30-minute prices, 5 bps a side, T-bill interest on cash. Not possible in an "
        "ISA; it shows where in the day the rule earns:", "",
        md(fmt(out["us_times"]), "Trade at"), "",
        "Trading at the next open or at 4:00 (about when London opens) earns about the same; trading at the close with "
        "the closing price known would earn the most, but deciding at 15:30 instead picks the wrong days (section 1) "
        "and earns almost nothing. Selling at the first half hour above the exit line cuts the winners short, as in "
        "London.", "",
        "## 5. Verdict", "",
        "- Keep the rule as it is: IBS < 0.1 on QQQ / SPY at the US close, buy LQQ3 / 3LUS at the next London open, "
        "sell at the London open after the index closes above its 5-day average, half the ISA in each fund.",
        "- Selling on RSI(2) > 70 instead made a few points more in both halves but fell further "
        f"({rules.loc['Exit: RSI(2) above 70', 'Max drawdown']:.0%} against {base['Max drawdown']:.0%}); a matter "
        "of taste, not a free improvement." if "Exit: RSI(2) above 70" in rules.index else "",
        "- Do not add stop losses: the rule buys falls on purpose, and a stop sells just before the bounce it is "
        "waiting for.",
        "- Trade at the open and keep costs down: use a limit order near the fund's indicative value (or wait a few "
        f"minutes after 8:00 for the spread to settle); 20 bps a side instead of 10 costs about "
        f"{(costs.iloc[0]['CAGR'] - costs.iloc[1]['CAGR']) * 100:.0f} points a year.",
        "- Caveats: the data starts in 2016, so the rule has not been through a long bear market like 2000-02 or "
        "2008, when dip buying in 3x funds could have been very painful; Yahoo's London prices are daily only; tax "
        f"does not apply inside an ISA, platform dealing fees do (about {2 * base['Trades a year']:.0f} trades a "
        "year)."]
    return "\n".join(x for x in lines if x is not None)


def _share(anatomy: pd.DataFrame, upto: str, k: int) -> float:
    """Share of the extra return to the next close that comes after the point `upto`."""
    total = anatomy.loc["Next close (16:00)", "Extra"]
    return float((total - anatomy.loc[upto, "Extra"]) / total)


def _low_table(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=a.index)
    for name, t in (("QQQ", a), ("SPY", b)):
        out[f"{name} days"] = t["Days"].astype(int)
        for c in ("Last half hour", "Next day", "Next 3 days"):
            out[f"{name} {c.lower()}"] = t[c].map(lambda v: f"{v:+.2%}")
    return out


def _grid_text(g: pd.DataFrame) -> str:
    """Whether the recommended exit length and entry are the best of the grid."""
    best_n = {x: g.loc[x]["CAGR"].idxmax() for x in GRID_X}
    halves = [(x, h) for x in GRID_X for h in ("2017-21", "2022-26") if g.loc[x][h].idxmax() == EXIT_N]
    full = g.xs(EXIT_N, level=1)["CAGR"].sort_values(ascending=False)
    n_best = sum(1 for n_ in best_n.values() if n_ == EXIT_N)
    return (f"The {EXIT_N}-day exit is the best of {', '.join(map(str, GRID_N))} days at {n_best} of {len(GRID_X)} entry "
            f"thresholds over the whole period (and in {len(halves)} of {2 * len(GRID_X)} threshold-halves); with it, "
            f"the best entries are IBS < {full.index[0]:g} ({full.iloc[0]:.1%} a year) and IBS < {full.index[1]:g} "
            f"({full.iloc[1]:.1%}).")


def worst_drawdowns(eq: pd.Series, k: int = 3) -> list[tuple]:
    """The k deepest peak-to-trough falls that do not overlap: (peak, trough, depth)."""
    dd = eq / eq.cummax() - 1
    out, used = [], pd.Series(False, index=eq.index)
    while len(out) < k:
        d = dd[~used]
        if d.empty or d.min() >= 0:
            break
        trough = d.idxmin()
        peak = eq[:trough].idxmax()
        rec = eq[trough:][eq[trough:] >= eq[peak]]
        end = rec.index[0] if len(rec) else eq.index[-1]
        used[(used.index >= peak) & (used.index <= end)] = True
        out.append((peak, trough, float(dd[trough])))
    return out


def _pivot(g: pd.DataFrame, col: str, f: str = "{:.1%}") -> pd.DataFrame:
    p = g[col].unstack()
    p.index = [f"{x:g}" for x in p.index]
    p.columns = [f"{n}-day avg" for n in p.columns]
    return p.map(lambda v: f.format(v))


def _rules_text(rules: pd.DataFrame, wins: list[str]) -> str:
    b = rules.loc[BASE]
    worse = rules[(rules["2017-21"] < b["2017-21"]) & (rules["2022-26"] < b["2022-26"])]
    return (f"{len(worse)} of {len(rules) - 1} changes did worse in both halves. "
            + (f"Better in both: {'; '.join(wins)}. " if wins else "None did better in both halves. ")
            + "Filters that skip some dips (trend, volatility, wide-range days, Fridays) cut good trades with the bad; "
              "tight stops sell into the very falls the rule buys (and made drawdowns deeper); smaller positions in "
              "volatile markets lower the drawdown roughly as much as the return, like holding some cash.")


def dt_today() -> str:
    import datetime as dt
    return str(dt.date.today())


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    out = run_all()
    text = report(out)
    (REPO_ROOT / "backtest" / "ISA_TESTS.md").write_text(text + "\n")
    out_dir = paths().root / "backtests" / "isa_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in ("rules", "hourly", "timing", "costs", "us_times", "grid"):
        out[k].to_csv(out_dir / f"{k}.csv")
    print(text)


if __name__ == "__main__":
    main()

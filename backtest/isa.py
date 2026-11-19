"""Daily strategies for a sterling ISA, with London-listed 3x funds from WisdomTree and Leverage Shares.

UNIVERSE: the 3x funds in the user's ISA with usable London prices (Yahoo, approved; splits and consolidations undone
by adjust_actions; untraded days with a stale or false price valued from the index by fill_stale_prints): WisdomTree LQQ3 (Nasdaq-100), 3LUS (S&P 500), 3UKL (FTSE 100), 3LDE (DAX), 3LEU (EURO STOXX 50),
3UKS (FTSE 100 -3x); Leverage Shares 3SMH (semiconductors) and 3GDX (gold miners), from April 2024 (Yahoo has no
prices from December 2022). Left out with reasons: EXCLUDED. The dip rules run on fund sets (US indices; plus sectors;
plus European indices; plus the FTSE short) with one place per fund or up to 2 / 3 at a time. RECOMMENDED is the
answer; PREREGISTERED is the US nine-fund rule, fixed before any London price was seen.

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
from .lqq3 import COST_BPS, dfmt, dip_measures, dip_state, fetch_yahoo, fill_stale_prints, sterling, to_gbp
from .momentum_trend import md, stats
from .strategies import SPLIT, START, yearly_returns

log = logging.getLogger(__name__)

FUNDS = {"LQQ3.L": "QQQ", "3LUS.L": "SPY"}  # London 3x fund (pence) -> the US ETF whose close gives its signal
NAMES = {"LQQ3.L": "LQQ3", "3LUS.L": "3LUS", "3UKL.L": "3UKL", "3LDE.L": "3LDE", "3LEU.L": "3LEU",
         "3SMH.L": "3SMH", "3XLF.L": "3XLF", "3XLE.L": "3XLE", "3GDX.L": "3GDX", "3ULS.L": "3ULS", "3UKS.L": "3UKS"}
US = ["QQQ", "SPY", "TQQQ", "UPRO", "SQQQ", "SPXU", "BIL", "SMH", "XLF", "XLE", "GDX", "EWU", "EWG", "FEZ"]
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
            cash_r: np.ndarray, lse: pd.DatetimeIndex, i0: int, cost_bps: float = COST_BPS,
            block: dict[str, np.ndarray] | None = None):
    """Hold each fund on the London sessions its US-close state asks for, at most n_pos at 1/n_pos of the account
    (the higher score first), trading at London opens. block: per fund, London sessions where holding is not
    allowed (to take given trades out). Returns equity (GBP, 1 at i0) and trades."""
    cols = list(states)
    O = np.column_stack([funds[f]["open"].reindex(lse).to_numpy() for f in cols])
    C = np.column_stack([funds[f]["close"].reindex(lse).ffill().to_numpy() for f in cols])
    want = np.column_stack([london_wanted(states[f], lse) for f in cols])
    for j, f in enumerate(cols):
        if block and f in block:
            want[:, j] &= ~block[f]
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


# ------------------------------------------------------------------ ISA fund universe (WisdomTree + Leverage Shares)

# London 3x funds in the user's ISA. signal: the US ETF whose close gives the signal, or "own" (the fund's own London
# prices: European and UK indices trade in London hours). proxy: what the fund is 3x (or -3x) of, to measure
# consolidations and splits that Yahoo leaves in the prices.
UNIVERSE = {
    "LQQ3.L": dict(name="LQQ3", tracks="Nasdaq-100 3x", issuer="WisdomTree", side=1, signal="QQQ", proxy="QQQ"),
    "3LUS.L": dict(name="3LUS", tracks="S&P 500 3x", issuer="WisdomTree", side=1, signal="SPY", proxy="SPY"),
    "3UKL.L": dict(name="3UKL", tracks="FTSE 100 3x", issuer="WisdomTree", side=1, signal="own", proxy="EWU"),
    "3LDE.L": dict(name="3LDE", tracks="DAX 3x", issuer="WisdomTree", side=1, signal="own", proxy="EWG"),
    "3LEU.L": dict(name="3LEU", tracks="EURO STOXX 50 3x", issuer="WisdomTree", side=1, signal="own", proxy="FEZ"),
    # Leverage Shares sector funds: Yahoo has no prices from 2022-12-09 to 2024-04-16 (consolidated on return),
    # so they are used from 2024-04-16 only
    "3SMH.L": dict(name="3SMH", tracks="Semiconductors (SMH) 3x", issuer="Leverage Shares", side=1, signal="SMH",
                   proxy="SMH", start="2024-04-16"),
    "3GDX.L": dict(name="3GDX", tracks="Gold miners (GDX) 3x", issuer="Leverage Shares", side=1, signal="GDX",
                   proxy="GDX", start="2024-04-16"),
    "3UKS.L": dict(name="3UKS", tracks="FTSE 100 -3x", issuer="WisdomTree", side=-1, signal="own:3UKL.L", proxy="EWU"),
}
# Approved and fetched but left out: 3XLF and 3XLE barely trade (about GBP 300 and GBP 8,000 a day, no price range on
# a fifth of days); 3ULS's 2022 consolidation ratio cannot be pinned down; the silver (3LSI) and oil (3LOI) pence
# lines have too little or broken history.
EXCLUDED = {"3XLF": "barely trades (about GBP 300 a day; no price range on 26% of days)",
            "3XLE": "barely trades (about GBP 8,000 a day; no price range on 17% of days)",
            "3ULS": "S&P 500 -3x: its December 2022 consolidation ratio cannot be pinned down from the prices",
            "3LSI / 3LOI": "silver and oil: one day of pence prices, or broken history from late 2020"}
SETS = {"US indices (2)": ["LQQ3.L", "3LUS.L"],
        "US indices + semis and gold miners (4)": ["LQQ3.L", "3LUS.L", "3SMH.L", "3GDX.L"],
        "WisdomTree indices (5)": ["LQQ3.L", "3LUS.L", "3UKL.L", "3LDE.L", "3LEU.L"],
        "indices + semis and gold miners (7)": ["LQQ3.L", "3LUS.L", "3UKL.L", "3LDE.L", "3LEU.L", "3SMH.L",
                                                "3GDX.L"],
        "7 longs + FTSE 100 short (8)": ["LQQ3.L", "3LUS.L", "3UKL.L", "3LDE.L", "3LEU.L", "3SMH.L", "3GDX.L",
                                         "3UKS.L"]}
UNI_ENTRIES = [("ibs", 0.1), ("ibs", 0.2), ("rsi", 10), ("rsi", 20), ("low", 5)]
# The recommendation: the steadiest setting on the ISA funds (chosen after seeing the results, from the IBS < 0.1
# family, which beat the S&P in both halves with every US-index fund set)
RECOMMENDED = "US indices (2), one slot each: IBS < 0.1; sell close > 5-day avg"
# Decided before seeing any London price: the steadiest rule of the US nine-fund test (backtest/DAILY.md)
PREREGISTERED = "indices + semis and gold miners (7), one slot each: IBS < 0.1; sell close > prior high"
ROUND_RATIOS = np.array([2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 80, 100, 125, 150, 200, 250, 300, 400,
                         500, 1000], dtype=float)


def adjust_actions(df: pd.DataFrame, under: pd.Series, lev: float) -> tuple[pd.DataFrame, list[tuple]]:
    """Undo consolidations and splits left in a London fund's daily prices. A close-to-close move above +100% or
    below -50% is not a 3x daily move: its ratio is measured against lev x the underlying (median of fund / model
    over 15 sessions on each side), rounded to a round ratio when within 5%, and all earlier prices are scaled by it."""
    out = df.copy()
    model = (1 + lev * under.pct_change().fillna(0.0)).cumprod().reindex(out.index, method="ffill")
    r = out["close"].pct_change()
    fixes = []
    for d in sorted(r[(r > 1.0) | (r < -0.5)].index, reverse=True):
        k = out.index.get_loc(d)
        rel = out["close"] / model
        ratio = rel.iloc[k:k + 15].median() / rel.iloc[max(0, k - 15):k].median()
        cands = np.concatenate([ROUND_RATIOS, 1 / ROUND_RATIOS])
        best = cands[np.argmin(np.abs(np.log(cands / ratio)))]
        used = best if abs(np.log(best / ratio)) < np.log(1.05) else ratio
        out.loc[out.index < d, ["open", "high", "low", "close"]] *= used
        fixes.append((str(d.date()), round(float(ratio), 2), float(used)))
    return out, fixes


def load_universe(b: dict[str, pd.DataFrame], gbpusd: pd.Series) -> tuple[dict, dict, dict]:
    """London prices of the UNIVERSE funds: consolidations and splits undone, untraded false prices replaced by the
    value implied by the proxy (fill_stale_prints). Also returns per fund the action fixes and the replaced days."""
    funds, fixes, filled = {}, {}, {}
    for t, f in UNIVERSE.items():
        df = fetch_yahoo(t, max_age_days=4)
        if f.get("start"):
            df = df[df.index >= pd.Timestamp(f["start"])]
        df, fixes[t] = adjust_actions(df, b[f["proxy"]]["c"], 3 * f["side"])
        funds[t] = fill_stale_prints(df, b[f["proxy"]]["c"], 3 * f["side"], gbpusd)
        filled[t] = df.index[funds[t]["close"].to_numpy() != df["close"].to_numpy()]
    return funds, fixes, filled


def signal_bars(t: str, funds: dict[str, pd.DataFrame], b: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sig = UNIVERSE[t]["signal"]
    if sig.startswith("own"):
        src = funds[sig.split(":")[1]] if ":" in sig else funds[t]
        return src.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c"})[["o", "h", "l", "c"]]
    return b[sig]


def universe_state(t: str, bars: pd.DataFrame, entry: tuple, leave: str, trend: bool) -> pd.Series:
    """Holding state after each signal close: dips for a long fund; for a short fund the mirror rule (a short
    rise while the index is below its 200-day average)."""
    if UNIVERSE[t]["side"] > 0:
        return dip_state(bars, entry, leave, trend)
    e, x = rip_signals(bars["c"], bars["h"], bars["l"], entry, leave)
    state, held = [], False
    for buy, sell in zip(e.to_numpy(), x.to_numpy()):
        held = (not sell) if held else bool(buy)
        state.append(held)
    return pd.Series(state, index=bars.index)


def universe_variants() -> list[tuple[str, dict]]:
    v = []
    for set_name, cols in SETS.items():
        for slots in ("one slot each", "up to 2 at a time", "up to 3 at a time"):
            if slots != "one slot each" and int(slots.split()[2]) >= len(cols):
                continue
            for entry in UNI_ENTRIES:
                kind, x = entry
                for leave in EXITS:
                    for trend in (False, True):
                        label = (f"{set_name}, {slots}: {ENTRY_LABEL[kind].format(x=x)}"
                                 f"{', above 200-day' if trend else ''}; sell {EXIT_LABEL[leave].format(x=x)}")
                        v.append((label, dict(set=set_name, slots=slots, entry=entry, leave=leave, trend=trend)))
    return v


def run_setting(b, funds, kw: dict, cash_r, lse, i0, block: dict[str, np.ndarray] | None = None):
    """Equity and trades of one universe setting (see universe_variants)."""
    cols = SETS[kw["set"]]
    bars = {t: signal_bars(t, funds, b) for t in cols}
    states = {t: universe_state(t, bars[t], kw["entry"], kw["leave"], kw["trend"]) for t in cols}
    oversold = {t: -rsi(bars[t]["c"], 2) for t in cols}
    n = len(cols) if kw["slots"] == "one slot each" else int(kw["slots"].split()[2])
    return isa_run({t: funds[t] for t in cols}, states, oversold, n, cash_r, lse, i0, block=block)


def run_universe(b, funds, cash_r, lse, i0, bench) -> tuple[pd.DataFrame, dict, dict]:
    rows, curves, trades = {}, {}, {}
    for label, kw in universe_variants():
        eq, t = run_setting(b, funds, kw, cash_r, lse, i0)
        rows[label], curves[label], trades[label] = dip_measures(eq, t, bench), eq, t
    return pd.DataFrame(rows).T, curves, trades


def without_years(eq: pd.Series, years=()) -> dict:
    """Yearly return and max drawdown with the given calendar years' daily returns left out (the rest chained)."""
    r = eq.pct_change().dropna()
    r = r[~r.index.year.isin(list(years))]
    curve = (1 + r).cumprod()
    span = (r.index[-1] - r.index[0]).days / 365.25 - sum(1.0 for y in years if r.index[0].year < y < r.index[-1].year)
    return {"CAGR": curve.iloc[-1] ** (1 / span) - 1, "Max drawdown": float((curve / curve.cummax() - 1).min())}


def best_trades_removed(out: dict, label: str, k: int = 10) -> pd.Series:
    """The setting's equity with its k best trades taken out (the money stays in cash for those sessions)."""
    lse, rev = out["ulse"], {v: key for key, v in NAMES.items()}
    block: dict[str, np.ndarray] = {}
    for r in out["utrades"][label].sort_values("ret", ascending=False).head(k).itertuples():
        arr = block.setdefault(rev[r.asset], np.zeros(len(lse), bool))
        arr[lse.get_loc(r.entry):lse.get_loc(r.exit)] = True
    eq, _ = run_setting(out["b"], out["ufunds"], dict(universe_variants())[label], out["ucash"], lse, out["ui0"], block)
    return eq


# ------------------------------------------------------------------ report

def run_all() -> dict:
    b = us_bars()
    gbpusd, cash = sterling()
    ufunds, fixes, filled = load_universe(b, gbpusd)
    funds = {f: ufunds[f] for f in FUNDS}
    last = min(min(f.index[-1] for f in funds.values()), b["QQQ"].index[-1] + pd.Timedelta(days=1))
    lse = funds["LQQ3.L"].index.union(funds["3LUS.L"].index)
    lse = lse[lse <= last]
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
    ulse = ufunds["LQQ3.L"].index
    for df in ufunds.values():
        ulse = ulse.union(df.index)
    ulse = ulse[ulse <= min(df.index[-1] for df in ufunds.values())]
    ui0 = int(ulse.searchsorted(pd.Timestamp(START))) - 1
    ucash = cash.reindex(ulse).fillna(0.0).to_numpy()
    ubench = to_gbp(spy[spy.index >= ulse[ui0]] / spy[spy.index >= ulse[ui0]].iloc[0], gbpusd)
    utbl, ucurves, utrades = run_universe(b, ufunds, ucash, ulse, ui0, ubench)
    out_dir = paths().root / "backtests" / "isa"
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_dir / "isa_variants.csv")
    shorts.to_csv(out_dir / "short_tests.csv")
    utbl.to_csv(out_dir / "universe_variants.csv")
    return {"table": tbl, "refs": pd.DataFrame(refs).T, "curves": curves, "trades": trades, "bench": bench,
            "shorts": shorts, "lse": lse, "i0": i0, "utable": utbl, "ucurves": ucurves, "utrades": utrades,
            "ubench": ubench, "ufunds": ufunds, "fixes": fixes, "filled": filled, "b": b, "ulse": ulse, "ui0": ui0, "ucash": ucash}


def without_winners_section(out: dict) -> list[str]:
    """How much of the recommendation's lead comes from 2020 and from its best trades."""
    rec, bench = out["ucurves"][RECOMMENDED], out["ubench"]
    lq = out["ufunds"]["LQQ3.L"]["close"]
    lq = lq[lq.index >= rec.index[0]]
    lines_ = {"Recommended": rec, "S&P 500 in pounds": bench,
              "Your current rule (LQQ3 above its 200-day avg)":
                  out["curves"]["Your current ISA rule: LQQ3 while QQQ is above its 200-day avg"],
              "LQQ3 buy and hold": lq / lq.iloc[0]}
    rows = {}
    for name, eq in lines_.items():
        best = int(yearly_returns(eq).idxmax())
        a, b, c = without_years(eq), without_years(eq, [2020]), without_years(eq, [best])
        rows[name] = {"All years": f"{a['CAGR']:.1%}", "Without 2020": f"{b['CAGR']:.1%}",
                      "Without its own best year": f"{c['CAGR']:.1%} (no {best})",
                      "Max drawdown": f"{a['Max drawdown']:.0%}", "Max drawdown without 2020": f"{b['Max drawdown']:.0%}"}
    t = out["utrades"][RECOMMENDED]
    split = pd.Timestamp(SPLIT)
    early = lambda e: without_years(e[e.index <= split], [2020])["CAGR"]  # noqa: E731
    sp_all = without_years(bench)["CAGR"]
    cut = {k: without_years(best_trades_removed(out, RECOMMENDED, k)) for k in (5, 10, 20)}
    top = t.sort_values("ret", ascending=False).head(10)
    return [
        "## Without the biggest winners", "",
        "2020 (the crash and the rebound) was the recommendation's best year. Its daily returns taken out, the rest "
        "chained together (all lines over the same years):", "",
        md(pd.DataFrame(rows).T, "Rule"), "",
        f"2017-21 without 2020: {early(rec):.1%} a year against {early(bench):.1%} for the S&P 500 in pounds (2022-26 "
        f"unchanged). Trades in 2020: {int((t['entry'].dt.year == 2020).sum())} of {len(t)}; win rate without them "
        f"{(t.loc[t['entry'].dt.year != 2020, 'ret'] > 0).mean():.0%}.", "",
        "With its best trades taken out (the money stays in cash for those sessions): " + "; ".join(
            f"{k} best: {v['CAGR']:.1%} a year ({v['CAGR'] - sp_all:+.1%} against the S&P)" for k, v in cut.items())
        + ". The ten best: " + ", ".join(f"{r.asset} {r.entry.date()} {r.ret:+.0%}" for r in top.itertuples()) + ".",
        ""]


def blend(e1: pd.Series, e2: pd.Series, w: float = 0.5) -> pd.Series:
    """Equity of an account with w in strategy e1 and 1 - w in e2, set back to w / 1 - w every January (both
    curves start at 1 on the same dates)."""
    e2 = e2.reindex(e1.index)
    year = e1.index.year.to_numpy()
    parts, level = [], 1.0
    for y in np.unique(year):
        rows = np.flatnonzero(year == y)
        base = max(rows[0] - 1, 0)
        seg = level * (w * e1.iloc[rows] / e1.iloc[base] + (1 - w) * e2.iloc[rows] / e2.iloc[base])
        parts.append(seg)
        level = seg.iloc[-1]
    return pd.concat(parts)


def index_edge(bars: pd.DataFrame, x: float = 0.1, start: str = "2017-01-01") -> dict:
    """The index's own moves after a close in the bottom x of its day's range, against all days."""
    bars = bars[bars.index >= pd.Timestamp(start)]
    hit = ((bars["c"] - bars["l"]) / (bars["h"] - bars["l"]) < x).to_numpy()
    nxt1, nxt3 = (bars["c"].shift(-1) / bars["c"] - 1).to_numpy(), (bars["c"].shift(-3) / bars["c"] - 1).to_numpy()
    return {"days": int(hit.sum()), "share": hit.mean(), "next day": np.nanmean(nxt1[hit]),
            "next day up": np.nanmean(nxt1[hit] > 0), "next 3 days": np.nanmean(nxt3[hit]),
            "all next day": np.nanmean(nxt1), "all next day up": np.nanmean(nxt1 > 0), "all next 3 days": np.nanmean(nxt3)}


def _combo_row(eq: pd.Series, invested: float) -> dict:
    split = pd.Timestamp(SPLIT)
    y = yearly_returns(eq)
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq[:trough].idxmax()
    return {"CAGR": f"{stats(eq)['CAGR']:.1%}", "Without 2020": f"{without_years(eq, [2020])['CAGR']:.1%}",
            "2017-21": f"{stats(eq[eq.index <= split])['CAGR']:.1%}",
            "2022-26": f"{stats(eq[eq.index >= split])['CAGR']:.1%}",
            "Max drawdown": f"{dd.min():.0%} ({peak:%b %Y} to {trough:%b %Y})",
            "Worst year": f"{y.min():+.0%} ({y.idxmin()})", "Time invested": f"{invested:.0%}"}


def combination_section(out: dict) -> list[str]:
    """What the recommendation earns from, and the recommendation combined with the trend rule (LQQ3 while QQQ is
    above its 200-day simple average)."""
    b, funds, lse, i0, cash, bench = (out[k] for k in ("b", "ufunds", "ulse", "ui0", "ucash", "ubench"))
    kws = dict(universe_variants())
    rec_kw = kws[RECOMMENDED]
    filtered = next(label for label, kw in kws.items() if kw == {**rec_kw, "trend": True})
    cols = SETS[rec_kw["set"]]
    sig = {f: UNIVERSE[f]["signal"] for f in cols}
    trend = {f: regime(b[sig[f]]["c"], 200, 1) for f in cols}
    dip = {f: universe_state(f, b[sig[f]], rec_kw["entry"], rec_kw["leave"], False) for f in cols}
    score = {f: -rsi(b[sig[f]]["c"], 2) for f in cols}

    def go(states: dict) -> tuple[pd.Series, float]:
        eq, t = isa_run({f: funds[f] for f in states}, states, {f: score[f] for f in states}, len(states), cash,
                        lse, i0)
        return eq, t["days"].sum() / (len(eq) * len(states))

    rec, t_rec = out["ucurves"][RECOMMENDED], out["utrades"][RECOMMENDED]
    inv_rec = t_rec["days"].sum() / (len(rec) * len(cols))
    t_f = out["utrades"][filtered]
    trend_eq, inv_trend = go({"LQQ3.L": trend["LQQ3.L"]})
    lines_ = {"Recommended dip rule": (rec, inv_rec),
              "Dip rule, buying only above the 200-day avg": (out["ucurves"][filtered],
                                                              t_f["days"].sum() / (len(rec) * len(cols))),
              "Your current rule: LQQ3 above the 200-day avg": (trend_eq, inv_trend),
              "Your rule, plus dip trades in LQQ3 below the 200-day avg": go(
                  {"LQQ3.L": trend["LQQ3.L"] | dip["LQQ3.L"]}),
              "Both funds: hold above the 200-day avg, dip trades below it": go(
                  {f: trend[f] | dip[f] for f in cols}),
              "Half the ISA in your rule, half in the dip rule (reset each January)": (
                  blend(trend_eq, rec), (inv_trend + inv_rec) / 2),
              "S&P 500 in pounds": (bench, 1.0)}
    rows = {name: _combo_row(eq, inv) for name, (eq, inv) in lines_.items()}

    # the recommendation's trades by the index's trend and volatility at the signal close
    info = []
    for r in t_rec.itertuples():
        c = b[sig[next(f for f in cols if NAMES[f] == r.asset)]]["c"]
        k = c.index.searchsorted(r.entry, side="left") - 1
        info.append({"ret": r.ret, "above": bool(c.iloc[k] > c.rolling(200).mean().iloc[k]),
                     "vol": float(c.pct_change().rolling(20).std().iloc[k] * np.sqrt(252))})
    info = pd.DataFrame(info)
    info["band"] = pd.qcut(info["vol"], 3, labels=["calmest third", "middle third", "most volatile third"])
    total = info["ret"].sum()
    by = {}
    for name, g in [("Index above its 200-day avg", info[info["above"]]),
                    ("Index below its 200-day avg", info[~info["above"]])] + \
            [(f"{band} (20-day volatility {g['vol'].min():.0%} to {g['vol'].max():.0%})", g)
             for band, g in info.groupby("band", observed=True)]:
        by[name] = {"Trades": len(g), "Win rate": f"{(g['ret'] > 0).mean():.0%}",
                    "Average trade": f"{g['ret'].mean():+.1%}", "Share of profit": f"{g['ret'].sum() / total:.0%}"}
    edge = {s: index_edge(b[s]) for s in ("QQQ", "SPY")}
    below = info[~info["above"]]
    c = b["QQQ"]["c"]
    ema_on = (c > c.ewm(span=200, adjust=False).mean()) & (np.arange(len(c)) >= 199)
    ema_eq, _ = go({"LQQ3.L": pd.Series(ema_on, index=c.index)})
    return [
        "## What the recommendation captures", "",
        "Short-term overreaction in the index: a close in the bottom 10% of the day's range (QQQ / SPY, the 9:30-16:00 "
        "New York session, Alpaca daily bars) tends to be followed by a bounce over the next few days. Since 2017:", "",
        md(pd.DataFrame({s: {"Closes in the bottom 10% of the range": f"{e['days']} ({e['share']:.0%} of days)",
                             "Next day after one": f"{e['next day']:+.2%} (up {e['next day up']:.0%})",
                             "Next day, all days": f"{e['all next day']:+.2%} (up {e['all next day up']:.0%})",
                             "Next 3 days after one": f"{e['next 3 days']:+.2%}",
                             "Next 3 days, all days": f"{e['all next 3 days']:+.2%}"} for s, e in edge.items()}),
           "Index"), "",
        f"The rule holds each fund only for that bounce (about {t_rec['days'].mean():.0f} sessions; invested "
        f"{inv_rec:.0%} of the time) and the 3x funds turn it into {t_rec['ret'].mean():+.1%} a trade on average. The "
        "bounces are bigger when the market is volatile, and dips below the 200-day average pay more per trade:", "",
        md(pd.DataFrame(by).T, "Index at the signal"), "",
        "## Combined with your 200-day rule", "",
        f"The averages are simple moving averages (the plain mean of the last 200, or 5, daily closes; the live paper "
        f"account uses the same). An exponential 200-day average would have given your rule {stats(ema_eq)['CAGR']:.1%} "
        f"a year against {stats(trend_eq)['CAGR']:.1%}, with a max drawdown of {stats(ema_eq)['Max drawdown']:.0%} "
        f"against {stats(trend_eq)['Max drawdown']:.0%}.", "",
        md(pd.DataFrame(rows).T, "Rule"), "",
        f"Buying dips only above the 200-day average skips the {len(below)} trades taken below it "
        f"({len(below) / len(info):.0%} of trades, {below['ret'].sum() / total:.0%} of the profit): a smaller drawdown "
        "for a lower return. Holding LQQ3 whenever the index is above its average and adding dip trades below it "
        "gives the highest return but the deepest fall, because the trend rule is still fully invested when a crash "
        "starts and the dip trades then buy into it. Splitting the ISA between the two rules sits in between.", ""]


def _years(days: pd.DatetimeIndex) -> str:
    """'n (2017: a, 2018: b, ...)' for the days from START."""
    days = days[days >= pd.Timestamp(START)]
    by = pd.Series(1, index=days).groupby(days.year).size()
    return f"{len(days)} ({', '.join(f'{y}: {k}' for y, k in by.items())})" if len(days) else "0"


def universe_section(out: dict) -> list[str]:
    """The ISA fund universe: what was checked, the rule fixed in advance, the 2017-21 pick and the sets."""
    utbl, ucurves, utrades, bench, funds, fixes = (out[k] for k in ("utable", "ucurves", "utrades", "ubench",
                                                                    "ufunds", "fixes"))
    split = pd.Timestamp(SPLIT)
    sp = {k: stats(v)["CAGR"] for k, v in (("all", bench), ("a", bench[bench.index <= split]),
                                           ("b", bench[bench.index >= split]))}
    main = ["CAGR", "vs S&P 500", "2017-21 vs S&P", "2022-26 vs S&P", "Win rate", "Trades per year",
            "Avg hold (sessions)", "Max drawdown", "Worst year"]
    ok = utbl[utbl["2017-21 win rate"] >= WIN_RATE]
    pick = ok["2017-21"].idxmax() if len(ok) else None
    both = utbl[(utbl["Win rate"] >= WIN_RATE) & (utbl["2017-21 vs S&P"] >= EXCESS) & (utbl["2022-26 vs S&P"] >= EXCESS)]
    rows = {}
    for t, f in UNIVERSE.items():
        df = funds[t]
        last = df[df.index >= df.index[-1] - pd.Timedelta(days=365)]
        rows[f["name"]] = {"Tracks": f["tracks"], "Issuer": f["issuer"], "Used from": str(df.index[0].date())
                           if df.index[0] > pd.Timestamp("2016-01-01") else "before 2017",
                           "Median GBP traded a day (last year)": f"{(last['close'] / 100 * last['volume']).median():,.0f}",
                           "Split / consolidation fixed": ", ".join(f"{d} x{u:g}" for d, _, u in fixes[t]) or "-",
                           "Untraded days repriced (backtest years)": _years(out["filled"][t])}
    funds_tbl = pd.DataFrame(rows).T
    group = pd.Series([s.split(":")[0] for s in utbl.index], index=utbl.index)
    by = utbl.groupby(group, sort=False).agg(**{
        "Settings": ("CAGR", "size"), "Median CAGR": ("CAGR", "median"), "Median vs S&P": ("vs S&P 500", "median"),
        "Median 2017-21 vs S&P": ("2017-21 vs S&P", "median"), "Median 2022-26 vs S&P": ("2022-26 vs S&P", "median"),
        "Median win rate": ("Win rate", "median"), "Median max drawdown": ("Max drawdown", "median")})
    for c in ("Median CAGR", "Median win rate", "Median max drawdown"):
        by[c] = by[c].map(lambda v: f"{v:.1%}")
    for c in ("Median vs S&P", "Median 2017-21 vs S&P", "Median 2022-26 vs S&P"):
        by[c] = by[c].map(lambda v: f"{v:+.1%}")
    # the FTSE short: each long-only setting against the same setting with the short fund added
    pairs = [(x, x.replace("indices + semis and gold miners (7)", "7 longs + FTSE 100 short (8)"))
             for x in utbl.index if x.startswith("indices + semis and gold miners (7)")]
    d_cagr = pd.Series([utbl.loc[b_, "CAGR"] - utbl.loc[a, "CAGR"] for a, b_ in pairs])
    d_dd = pd.Series([utbl.loc[b_, "Max drawdown"] - utbl.loc[a, "Max drawdown"] for a, b_ in pairs])
    fam = utbl[utbl.index.str.startswith(RECOMMENDED.split(":")[0] + ": IBS < 0.1")]
    halves = fam[["2017-21 vs S&P", "2022-26 vs S&P"]]
    beat, worst = int((halves > 0).all(axis=1).sum()), float(halves.min().min())
    family_text = f"{beat} of the {len(fam)} IBS < 0.1 settings on these two funds beat the S&P in both halves"
    if beat < len(fam):
        family_text += (" and the rest were level with it (within 0.5%)" if worst > -0.005 else
                        f" and none trailed it by more than {-worst:.1%} in either half")
    shown = [x for x in [RECOMMENDED, PREREGISTERED, pick] if x in utbl.index]
    shown += [x for x in both.sort_values("CAGR", ascending=False).index if x not in shown][:8]
    lines = [
        "## The funds checked (WisdomTree and Leverage Shares, London-listed, in pence)", "",
        "WisdomTree's 3x index funds trade every day since 2012. Leverage Shares' 3x sector funds list in London from "
        "December 2021, but Yahoo has no prices for them from December 2022 to April 2024, so they enter in April "
        "2024. Signals: the US ETF at the US close for US markets (QQQ, SPY, SMH, GDX), the fund's own London prices "
        "for the FTSE 100, DAX and EURO STOXX 50; every trade at the next London open.", "",
        md(funds_tbl, "Fund"), "",
        "Left out: " + "; ".join(f"{k}: {v}" for k, v in EXCLUDED.items()) + ".", "",
        "## Answer", "",
        f"Benchmark: the S&P 500 in pounds, {sp['all']:.1%} a year ({sp['a']:.1%} in 2017-21, {sp['b']:.1%} in "
        f"2022-26). {len(utbl)} settings on these funds; {len(both)} beat it by {EXCESS:.0%} or more in both 2017-21 "
        f"and 2022-26 with at least {WIN_RATE:.0%} winning trades.", "",
        f"**Recommended: {RECOMMENDED}.** Half the ISA in LQQ3 (Nasdaq-100 3x) and half in 3LUS (S&P 500 3x): buy a "
        "fund at the London open after its index (QQQ for LQQ3, SPY for 3LUS) closes in the bottom 10% of its day's "
        "range, sell at the London open after the index closes above its 5-day average, cash in between. It was "
        f"picked after seeing the results, but {family_text}. "
        "The European index funds and the short funds made results worse; Leverage Shares' semiconductor and gold "
        "miner funds (prices only from April 2024) did not improve it.", "",
        "Rows: the recommendation; the rule fixed in advance from the US nine-fund test (backtest/DAILY.md), before "
        "any London price was seen (the honest out-of-sample test of that idea); the setting with the best 2017-21 "
        f"return among those that won at least {WIN_RATE:.0%} of their 2017-21 trades; then the settings that beat "
        f"the S&P by {EXCESS:.0%} or more in both halves:", "",
        md(dfmt(utbl.loc[shown, main]), "Setting"), "",
        "By fund set and number of places:", "", md(by, "Funds and places"), "",
        f"Adding the FTSE 100 3x short fund (bought after short rises while the FTSE is below its 200-day average) "
        f"changed the yearly return by {d_cagr.median():+.1%} (median over {len(pairs)} settings; "
        f"{(d_cagr > 0).mean():.0%} improved) and the max drawdown by {d_dd.median():+.1%}.", ""]
    if RECOMMENDED in ucurves and PREREGISTERED in ucurves:
        years = pd.DataFrame({"Recommended": yearly_returns(ucurves[RECOMMENDED]),
                              "Rule fixed in advance (7 funds)": yearly_returns(ucurves[PREREGISTERED]),
                              "S&P 500 in pounds": yearly_returns(bench)})
        lines += ["Calendar years:", "", md(years.map(lambda v: f"{v:+.1%}"), "Year"), ""]
        for label, name in ((RECOMMENDED, "the recommendation"), (PREREGISTERED, "the rule fixed in advance")):
            t = utrades[label]
            share = (t.groupby("asset")["ret"].sum() / t["ret"].sum()).sort_values(ascending=False)
            lines += [f"Where {name} made its money (share of summed trade returns, trades): " + ", ".join(
                f"{a} {v:.0%} ({(t['asset'] == a).sum()})" for a, v in share.items()) + "."]
        worst = utrades[RECOMMENDED].sort_values("ret").head(5)
        lines += ["", "Worst trades of the recommendation: " + ", ".join(
            f"{r.asset} {r.entry.date()} to {r.exit.date()} {r.ret:+.1%}" for r in worst.itertuples()) + ".", ""]
        lines += without_winners_section(out)
        lines += combination_section(out)
    return lines


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
        *universe_section(out), "",
        "## Two funds only: LQQ3 and 3LUS (the earlier test)", "",
        f"Generated by `python -m backtest.isa` on {dt.date.today()}. See the module docstring for the rules. Real "
        "London prices of LQQ3 (3x Nasdaq-100) and 3LUS (3x S&P 500) in pence; signals at the US close, trades at "
        f"the next London open, {COST_BPS:g} bps a side, sterling interest on idle cash. Benchmark: the S&P 500 in "
        f"pounds, {sp['all']:.1%} a year ({sp['a']:.1%} in 2017-21, {sp['b']:.1%} in 2022-26). Targets: at least "
        f"{WIN_RATE:.0%} winning trades and {EXCESS:.0%} a year above the S&P 500.", "",
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
        "long-side results show can cost several points a year.",
        "- Prices on days nothing traded: Yahoo often repeats the last price (about a third of 2018 for LQQ3 and 3LUS) "
        "and sometimes shows a false one (3LUS, 3LDE and 3LEU +15-27% on 26-27 June 2017, back the next day). Those "
        "days are valued at the price implied by the index since the last real price (fill_stale_prints; within "
        "about 2% a day of real prices where both exist), so trades happen on schedule as with a market maker's "
        "quote. Trading at the stale or false prices instead would move single years by more than ten points, so "
        "2017-18 are the least certain years."]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    text = report(run_all())
    (REPO_ROOT / "backtest" / "ISA.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

"""LQQ3 (WisdomTree NASDAQ 100 3x Daily Leveraged, London Stock Exchange, priced in pence) in a sterling ISA.

The trend rule: hold LQQ3 while QQQ closes above its 200-day average, otherwise cash. The signal comes at
the US close, after London has shut, so each switch trades at LQQ3's open in the next London session.

Sources: LQQ3.L daily prices from Yahoo Finance, the one source outside Alpaca and SEC EDGAR, approved by
the user on 2026-11-05 for LQQ3 only (cached in data/benchmarks/, never committed). Untraded days whose price is
stale or false are valued at the price implied by 3x QQQ in pounds (fill_stale_prints). Everything else comes
from Alpaca: QQQ for the signal, SPY for the S&P 500, and FXB (Invesco CurrencyShares British Pound) for the
pound/dollar rate and for sterling cash interest (what FXB's pound deposits paid after its 0.40% fee, about
0% until 2022: conservative next to a money-market fund).

Daily dip trading (python -m backtest.lqq3 dips, writes backtest/LQQ3_DAILY.md): the short-dip rules of
backtest/daily.py on QQQ (RSI(2), IBS, N-day lows, down streaks; sell on the bounce), traded in LQQ3 at the next
London open (or the next London close) in pounds, the only 3x fund with approved price data. Compared with the
S&P 500 in pounds; picked on 2017-21, checked on 2022-26.

Run: python -m backtest.lqq3   (writes backtest/LQQ3.md)
"""
from __future__ import annotations

import datetime as dt
import logging
import time

import numpy as np
import pandas as pd
import requests

from stockdata import alpaca
from stockdata.config import REPO_ROOT, paths

from .indicators import regime
from .momentum_trend import md, stats
from .strategies import SPLIT, START, Book, run, yearly_returns

log = logging.getLogger(__name__)

SYMBOL = "LQQ3.L"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/"
COST_BPS = 10.0  # LQQ3 bid/ask spread, per side
FXB_FEE = 0.004  # FXB's yearly fee: while it paid no interest the fee came out of its price
SHADOW_START = "2026-11-05"  # first trading day of the lqq3 paper account
LONDON = "Europe/London"


def drop_bad_prints(df: pd.DataFrame, jump: float = 0.15) -> pd.DataFrame:
    """Remove isolated bad prints: a flat bar (open = high = low = close) more than `jump` away from both
    neighbouring closes in the same direction (seen: 24 Oct 2025, +39% then -22%, no volume)."""
    c = df["close"]
    up, down = c / c.shift(1) - 1, c / c.shift(-1) - 1
    flat = (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    spike = (up.abs() > jump) & (down.abs() > jump) & (np.sign(up) == np.sign(down)) & flat
    return df[~spike.fillna(False)]


def fill_stale_prints(df: pd.DataFrame, under: pd.Series, lev: float, gbpusd: pd.Series | None = None,
                      tol: float = 0.10) -> pd.DataFrame:
    """Replace London prices that no trade backs (volume 0) and that are not real prices with the fund's value
    implied by its index, so a backtest trades on the day at about what a market maker would quote, as a real
    investor could: a repeat of the previous close (Yahoo carries the last price forward on quiet days: about a
    third of 2018 for LQQ3 and 3LUS, through index moves of up to 4% a day), or a close more than `tol` off the
    fund's path (against the local median of fund / path over 21 sessions; seen: 3LUS +27% on 26 and 27 June 2017
    with no volume, back the next day). The path is lev x the daily moves of `under` (its US underlying), in
    pounds when gbpusd is given, taken midway between the US closes either side of the London close (London shuts
    at 11:30 in New York). A replaced day is a flat bar at the last real close times the path since then, like
    the other untraded days. Days with trades are kept as they are."""
    out = df.copy()
    path = (1 + lev * under.dropna().pct_change().fillna(0.0)).cumprod()
    if gbpusd is not None:
        path = path / gbpusd.reindex(path.index).ffill().bfill()
    us, p = path.index, path.to_numpy()
    before = us.searchsorted(out.index, side="left") - 1  # the last US close before the London day
    upto = us.searchsorted(out.index, side="right") - 1  # that day's US close (after London shuts), if any
    mid = np.where(before >= 0, np.sqrt(p[np.clip(before, 0, None)] * p[np.clip(upto, 0, None)]), np.nan)
    c = out["close"].to_numpy(float)
    vol0 = (out["volume"].fillna(0) <= 0).to_numpy()
    repeat = vol0 & np.r_[False, c[1:] == c[:-1]]
    rel = pd.Series(np.where(repeat, np.nan, c / mid), index=out.index)
    off = vol0 & ((rel / rel.rolling(21, center=True, min_periods=5).median() - 1).abs() > tol).to_numpy()
    bad = repeat | off
    last = pd.Series(np.where(bad, np.nan, np.arange(len(c)))).ffill().to_numpy()
    fix = bad & np.isfinite(last) & np.isfinite(mid)
    j = last[fix].astype(int)
    value = c[j] / mid[j] * mid[fix]
    for col in ("open", "high", "low", "close"):
        out.loc[out.index[fix], col] = value
    return out


# London-listed funds whose Yahoo prices the user approved (CLAUDE.md), and when
YAHOO_APPROVED = {"LQQ3.L": "2026-11-05", "3LUS.L": "2026-11-19",
                  # WisdomTree and Leverage Shares funds in the user's ISA (approved 2026-11-19: "check the funds
                  # available in ISA from those two companies ... dont just only use the funds that I mention")
                  **{f"{t}.L": "2026-11-19" for t in ("3UKL", "3LDE", "3LEU", "3LSI", "3LOI", "3ULS", "3UKS",
                                                       "3SMH", "3XLF", "3XLE", "3GDX")}}


def fetch_lqq3(max_age_days: int | None = None) -> pd.DataFrame:
    """Daily LQQ3 prices in pence (open, high, low, close, volume), bad prints removed."""
    return fetch_yahoo(SYMBOL, max_age_days)


def fetch_yahoo(symbol: str, max_age_days: int | None = None) -> pd.DataFrame:
    """Daily prices in pence of an approved London fund (YAHOO_APPROVED), bad prints removed.

    max_age_days: reuse the cache if its last bar is at most this old (None: always download).
    Today's bar is dropped until London has closed (16:30 plus a margin)."""
    if symbol not in YAHOO_APPROVED:
        raise ValueError(f"{symbol}: Yahoo prices are approved only for {sorted(YAHOO_APPROVED)} (see CLAUDE.md)")
    path = paths().root / "benchmarks" / f"{symbol}.parquet"
    if max_age_days is not None and path.exists():
        df = pd.read_parquet(path)
        if (pd.Timestamp.now(tz=LONDON).tz_localize(None).normalize() - df.index.max()).days <= max_age_days:
            return df
    r = requests.get(YAHOO_CHART + symbol, params={"period1": 0, "period2": int(time.time()), "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    df = parse_chart(r.json())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def parse_chart(payload: dict, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Yahoo chart JSON -> daily bars in pence. A last bar whose close is still blank takes the quoted
    market price once London has closed that day (whenever its last trade was); a bar of today before
    the close is left out."""
    res = payload["chart"]["result"][0]
    meta = res["meta"]
    if meta.get("currency") != "GBp":
        raise ValueError(f"{meta.get('symbol', SYMBOL)} is quoted in {meta.get('currency')}, expected GBp (pence)")
    q = res["indicators"]["quote"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(LONDON).tz_localize(None).normalize()
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")}, index=idx).astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    now = (now or pd.Timestamp.now(tz=LONDON)).tz_convert(LONDON)
    mkt = pd.Timestamp(meta.get("regularMarketTime", 0), unit="s", tz="UTC").tz_convert(LONDON)
    last = df.index[-1]
    # London has closed for the last bar's day: its last quoted price is the close, even when the last trade
    # came before the 16:30 auction (LQQ3 trades thinly)
    shut = last < now.tz_localize(None).normalize() or now.hour * 60 + now.minute >= 16 * 60 + 45
    if pd.isna(df["close"].iloc[-1]) and mkt.tz_localize(None).normalize() == last and shut and \
            meta.get("regularMarketPrice"):
        df.loc[last, "close"] = float(meta["regularMarketPrice"])
    if now.hour * 60 + now.minute < 16 * 60 + 45:
        df = df[df.index < now.tz_localize(None).normalize()]
    df = df.dropna(subset=["open", "close"])
    return drop_bad_prints(df)


def alpaca_daily(symbols: list[str], adjustment: str = "all", start: str = "2016-01-01") -> pd.DataFrame:
    bars = alpaca.drop_unfinished(alpaca.fetch_bars(alpaca.session(), symbols, start, alpaca.safe_end(), adjustment,
                                                    "sip"))
    return bars.pivot(index="date", columns="ticker", values="c").sort_index()


def sterling(start: str = "2016-01-01") -> tuple[pd.Series, pd.Series]:
    """(GBP/USD rate, daily sterling cash return) from FXB: its price is about 100 x GBP/USD (the fee is
    added back for the years it paid no interest); its distributions are the sterling deposit interest."""
    raw, adj = alpaca_daily(["FXB"], "raw", start)["FXB"], alpaca_daily(["FXB"], "all", start)["FXB"]
    interest = ((adj / adj.shift(1)) / (raw / raw.shift(1)) - 1).fillna(0.0)
    interest = interest.where(interest.abs() > 1e-5, 0.0)  # rounding noise
    paying = interest.rolling(63, min_periods=1).sum() > 0
    years = pd.Series((raw.index - raw.index[0]).days / 365.25, index=raw.index)
    fee_back = np.exp(FXB_FEE * (years.where(~paying).ffill().fillna(0)))  # stops growing once it pays
    return (raw / 100 * fee_back).rename("GBPUSD"), interest.rename("gbp_cash")


def schedule(on: pd.Series, lse: pd.DatetimeIndex, i0: int) -> dict[int, dict[int, float]]:
    """Targets on the London calendar for run(): the state at each US close is traded at the open of the
    first London session after that day. The first entry sets the position for London session i0."""
    out: dict[int, dict[int, float]] = {}
    first = on[on.index < lse[i0]]
    if len(first):
        out[i0 - 1] = {0: 1.0} if first.iloc[-1] else {}
    changes = on[on != on.shift(1)]
    for t, state in changes[changes.index >= lse[i0 - 1]].items():
        k = int(lse.searchsorted(t, side="right"))  # first London session strictly after the US close
        if i0 <= k < len(lse):
            out[k - 1] = {0: 1.0} if state else {}  # a later US close mapping to the same session wins
    return out


def _plan(lq: pd.DataFrame, on: pd.Series, start: str, at_close: bool):
    """LQQ3 fill prices and London-calendar targets for a US-close state (True = hold LQQ3). A target decided at
    London session k is filled at o[k + 1]: the next open, or with at_close the close of the first London
    session after the signal."""
    lse = lq.index
    o, c = lq[["open"]].rename(columns={"open": SYMBOL}), lq[["close"]].rename(columns={"close": SYMBOL})
    if at_close:  # fill at the close of the first London session after the signal instead of its open
        o = c.shift(1).fillna(o)
    i0 = int(lse.searchsorted(pd.Timestamp(start)))
    targets = schedule(on, lse, i0)
    if at_close:
        targets = {k + 1: v for k, v in targets.items() if k + 1 < len(lse)}
    return o, c, i0, targets


def lqq3_strategy(lq: pd.DataFrame, qqq: pd.Series, cash_r: pd.Series, start: str = START, n: int = 200,
                  confirm: int = 1, at_close: bool = False, on: pd.Series | None = None,
                  cost_bps: float = COST_BPS) -> tuple[pd.Series, float]:
    """Equity (GBP, starting at 1) of a rule traded in LQQ3 on London sessions from `start`: the trend rule
    (QQQ above its n-day average) unless `on` gives the state after each US close."""
    on = regime(qqq.dropna(), n, confirm) if on is None else on
    o, c, i0, targets = _plan(lq, on, start, at_close)
    book = Book.build(o, c, cost_bps)
    cash = cash_r.reindex(lq.index).fillna(0.0).to_numpy()
    eq, turnover, _ = run(book, targets, cash, i0, lq.index)
    return eq, turnover


def lqq3_trades(lq: pd.DataFrame, on: pd.Series, start: str = START, at_close: bool = False,
                cost_bps: float = COST_BPS) -> pd.DataFrame:
    """The round trips lqq3_strategy makes for state `on`: London entry and exit sessions, fill prices (pence)
    and the return after costs on both sides."""
    o, _, _, targets = _plan(lq, on, start, at_close)
    px, lse, cost = o.iloc[:, 0].to_numpy(), lq.index, cost_bps / 1e4
    rows, entry = [], None
    for k in sorted(targets):
        if k + 1 >= len(lse):
            continue
        if targets[k] and entry is None:
            entry = k + 1
        elif not targets[k] and entry is not None:
            rows.append({"entry": lse[entry], "exit": lse[k + 1], "entry_px": px[entry], "exit_px": px[k + 1],
                         "ret": (1 - cost) ** 2 * px[k + 1] / px[entry] - 1, "days": k + 1 - entry})
            entry = None
    return pd.DataFrame(rows, columns=["entry", "exit", "entry_px", "exit_px", "ret", "days"])


def dip_state(bars: pd.DataFrame, entry: tuple, leave: str, trend: bool) -> pd.Series:
    """Holding (True) or not after each US close: in from a close with a buy signal (backtest.daily rules)
    until a close with a sell signal. bars: QQQ with columns o, h, l, c on US dates."""
    from .daily import dip_signals
    e, x = dip_signals(bars["c"], bars["h"], bars["l"], entry, leave, trend)
    state, held = [], False
    for buy, sell in zip(e.to_numpy(), x.to_numpy()):
        held = (not sell) if held else bool(buy)
        state.append(held)
    return pd.Series(state, index=bars.index)


def to_gbp(eq_usd: pd.Series, gbpusd: pd.Series) -> pd.Series:
    fx = gbpusd.reindex(eq_usd.index).ffill()
    return eq_usd / (fx / fx.iloc[0])


def shadow(start: str = SHADOW_START) -> dict:
    """What the rule did in LQQ3, in pounds, since the paper account started (for `python -m trading`)."""
    lq = fetch_lqq3(max_age_days=0)
    qqq = alpaca_daily(["QQQ"], start=str((pd.Timestamp(start) - pd.Timedelta(days=420)).date()))["QQQ"]
    _, cash = sterling(str((pd.Timestamp(start) - pd.Timedelta(days=30)).date()))
    lse = lq.index
    if lse[-1] < pd.Timestamp(start):
        return {"start": start, "status": "no London session since the start yet"}
    eq, _ = lqq3_strategy(lq, qqq, cash, start)
    on = regime(qqq.dropna(), 200, 1)
    return {"start": start, "through": str(eq.index[-1].date()),
            "return since start (GBP)": f"{eq.iloc[-1] - 1:+.2%}",
            "LQQ3 itself since start": f"{lq['close'].iloc[-1] / lq.loc[lq.index >= pd.Timestamp(start), 'open'].iloc[0] - 1:+.2%}",
            "signal at last US close": "above 200-day: hold LQQ3" if on.iloc[-1] else "below 200-day: hold cash"}


def report() -> str:
    us = alpaca_daily(["QQQ", "SPY"])
    gbpusd, cash = sterling()
    raw = fetch_lqq3(max_age_days=4)
    lq = fill_stale_prints(raw, us["QQQ"], 3, gbpusd)
    stale = int(((lq["close"] != raw["close"]) & (lq.index >= pd.Timestamp(START))).sum())
    qqq = us["QQQ"]
    base, _ = lqq3_strategy(lq, qqq, cash)
    rows = {"LQQ3 when QQQ > 200-day avg (real prices)": base}
    for label, kw in (("fill at the London close instead of the open", {"at_close": True}),
                      ("3-day confirmation", {"confirm": 3}), ("150-day average", {"n": 150}),
                      ("250-day average", {"n": 250})):
        rows[f"  ... {label}"] = lqq3_strategy(lq, qqq, cash, **kw)[0]
    hold = lq["close"][lq.index >= base.index[0]]
    rows["LQQ3 buy and hold"] = hold / hold.iloc[0]
    spy = us["SPY"][us.index >= base.index[0]]
    sp_gbp = to_gbp(spy / spy.iloc[0], gbpusd)
    rows["S&P 500 in pounds (SPY / GBPUSD)"] = sp_gbp
    # the earlier simulation (3x QQQ with TQQQ's costs), converted to pounds, as a check
    from .strategies import build_lab, etf_trend, LQQ3
    lab = build_lab()
    sim_usd, _, _ = run(lab.etfs, etf_trend(lab, "QQQ", LQQ3), lab.cash_r, lab.i0, lab.dates)
    rows["Simulated LQQ3 strategy (USD model, converted to pounds)"] = to_gbp(sim_usd, gbpusd)
    tbl = {}
    sp_stats = stats(sp_gbp)
    split = pd.Timestamp(SPLIT)
    for name, eq in rows.items():
        s = stats(eq)
        yr, ys = yearly_returns(eq), yearly_returns(sp_gbp)
        a, b = eq[eq.index <= split], eq[eq.index >= split]
        sa, sb = sp_gbp[sp_gbp.index <= split], sp_gbp[sp_gbp.index >= split]
        tbl[name] = {"CAGR": f"{s['CAGR']:.1%}", "vs S&P 500 (GBP)": f"{s['CAGR'] - sp_stats['CAGR']:+.1%}",
                     "Max drawdown": f"{s['Max drawdown']:.1%}", "Worst year": f"{yr.min():+.0%}",
                     "Sharpe (rf=0)": f"{s['Sharpe (rf=0)']:.2f}",
                     "2017-21 vs S&P": f"{stats(a)['CAGR'] - stats(sa)['CAGR']:+.1%}",
                     "2022-26 vs S&P": f"{stats(b)['CAGR'] - stats(sb)['CAGR']:+.1%}",
                     "Years beating S&P": int((yr > ys.reindex(yr.index)).sum())}
    t = pd.DataFrame(tbl).T
    years = pd.DataFrame({"LQQ3 trend (real)": yearly_returns(base), "LQQ3 held": yearly_returns(rows["LQQ3 buy and hold"]),
                          "S&P 500 (GBP)": yearly_returns(sp_gbp)}).T.map(lambda v: f"{v:+.0%}")
    on = regime(qqq.dropna(), 200, 1)
    on = on[on.index >= base.index[0]]
    switches = int((on != on.shift(1)).iloc[1:].sum())
    first_close = lq.loc[lq.index >= base.index[0], "close"].iloc[0]
    lines = [
        f"# LQQ3 in a sterling ISA ({base.index[0].date()} to {base.index[-1].date()})", "",
        f"Generated by `python -m backtest.lqq3` on {dt.date.today()}. Rule: hold LQQ3 (WisdomTree NASDAQ 100 3x "
        "Daily Leveraged, LSE, pence) while QQQ closes above its 200-day average, otherwise sterling cash. The signal "
        "is known at the US close (after London shuts), so each switch trades at LQQ3's open in the next London "
        f"session, paying {COST_BPS:.0f} bps a side. Returns are in pounds, as in an ISA (no tax).", "",
        md(t, "Strategy"), "",
        f"The rule switched {switches} times ({switches / ((on.index[-1] - on.index[0]).days / 365.25):.1f} a year) "
        f"and was invested {on.mean():.0%} of the time. LQQ3 went from {first_close:,.0f}p to "
        f"{lq['close'].iloc[-1]:,.0f}p over the period.", "",
        "## Calendar-year returns (pounds)", "", md(years, "Strategy"), "",
        "## Data and assumptions", "",
        "- LQQ3 prices: Yahoo Finance daily bars for LQQ3.L (approved for this one series). Isolated bad prints are "
        "removed (24 Oct 2025: a flat bar at +39% with no volume). On days nothing traded Yahoo often repeats the "
        f"last price (about a third of 2018); those {stale} days are valued at the price implied by 3x QQQ in "
        "pounds since the last real price (fill_stale_prints), which comes within about 2% of real prices.",
        "- Signal: QQQ adjusted close from Alpaca. S&P 500: SPY total return from Alpaca, converted to pounds with "
        "FXB (pound sterling trust) as the exchange rate. A sterling S&P 500 fund (VUSA, CSP1) loses about 0.2% a "
        "year more to US dividend withholding.",
        "- Cash when out of the market: what FXB's pound deposits paid after its fee (0% until 2022, then about "
        "2.5-3% a year). A money-market fund in an ISA paid more since 2023, so this errs low.",
        f"- Not included: platform dealing fees (a few pounds a trade, about {switches / ((on.index[-1] - on.index[0]).days / 365.25):.0f} "
        "trades a year). LQQ3's own costs are already in its price.",
        "- 3x leverage: drawdowns of 60% and more are part of this rule. A crash faster than the 200-day signal "
        "(the Nasdaq-100 fell 30% before crossing it in 2000) would cost about two-thirds of the position.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ daily dip trading in LQQ3 (ISA)

ISA_WIN_RATE, ISA_EXCESS = 0.60, 0.15  # the targets: 60%+ winning trades, 15 points a year above the S&P 500


def dip_variants() -> list[tuple[str, dict]]:
    from .daily import ENTRIES, ENTRY_LABEL, EXIT_LABEL, EXITS
    v = []
    for entry in ENTRIES:
        kind, x = entry
        for leave in EXITS + (["nhigh"] if kind == "low" else []):
            for trend in (True, False):
                for at_close in (False, True):
                    label = (f"QQQ {ENTRY_LABEL[kind].format(x=x)}{', above 200-day' if trend else ''}; "
                             f"sell {EXIT_LABEL[leave].format(x=x)}; LQQ3 at the London {'close' if at_close else 'open'}")
                    v.append((label, dict(entry=entry, leave=leave, trend=trend, at_close=at_close)))
    return v


def dip_measures(eq: pd.Series, t: pd.DataFrame, bench: pd.Series) -> dict:
    split = pd.Timestamp(SPLIT)
    part = lambda s, early: s[s.index <= split] if early else s[s.index >= split]  # noqa: E731
    cagr = lambda s: stats(s)["CAGR"]  # noqa: E731
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    early, late = t[t["exit"] <= split], t[t["exit"] > split]
    wins, losses = t.loc[t["ret"] > 0, "ret"], t.loc[t["ret"] <= 0, "ret"]
    return {"CAGR": cagr(eq), "vs S&P 500": cagr(eq) - cagr(bench),
            "2017-21": cagr(part(eq, True)), "2017-21 vs S&P": cagr(part(eq, True)) - cagr(part(bench, True)),
            "2022-26": cagr(part(eq, False)), "2022-26 vs S&P": cagr(part(eq, False)) - cagr(part(bench, False)),
            "Win rate": (t["ret"] > 0).mean() if len(t) else np.nan,
            "2017-21 win rate": (early["ret"] > 0).mean() if len(early) else np.nan,
            "2022-26 win rate": (late["ret"] > 0).mean() if len(late) else np.nan,
            "Trades per year": len(t) / years, "Avg hold (sessions)": t["days"].mean() if len(t) else np.nan,
            "Avg win": wins.mean() if len(wins) else np.nan, "Avg loss": losses.mean() if len(losses) else np.nan,
            "Worst trade": t["ret"].min() if len(t) else np.nan,
            "Time invested": t["days"].sum() / len(eq) if len(t) else 0.0,
            "Max drawdown": stats(eq)["Max drawdown"], "Worst year": yearly_returns(eq).min()}


def us_version(bars: pd.DataFrame, entry: tuple, leave: str, trend: bool) -> dict:
    """The same rule in TQQQ at the next US open, in dollars (backtest.daily's engine), for comparison."""
    from .daily import ETF_COST, dip_signals, run as daily_run
    q = bars["QQQ"]
    e, x = dip_signals(q["c"], q["h"], q["l"], entry, leave, trend)
    dates = q.index
    i0 = int(dates.searchsorted(pd.Timestamp(START))) - 1
    cash = bars["BIL"]["c"].ffill().pct_change().fillna(0.0).to_numpy()
    eq, _, tr = daily_run(bars["TQQQ"][["o"]].to_numpy(), bars["TQQQ"][["c"]].ffill().to_numpy(),
                          e.to_numpy(bool)[:, None], x.to_numpy(bool)[:, None], np.ones((len(dates), 1)), 1, ETF_COST,
                          cash, i0)
    eq = pd.Series(eq, index=dates).dropna()
    spy = bars["SPY"]["c"].iloc[i0:]
    return {"US CAGR (TQQQ, $)": stats(eq)["CAGR"], "US vs SPY": stats(eq)["CAGR"] - stats(spy)["CAGR"],
            "US win rate": np.mean([r[-1] > 0 for r in tr]) if tr else np.nan}


def dfmt(df: pd.DataFrame) -> pd.DataFrame:
    """Percent columns as 12.3% (differences as +1.2%), counts as whole numbers."""
    out = df.copy().astype(object)
    for c in df.columns:
        if c in ("Trades per year", "Avg hold (sessions)"):
            out[c] = [("" if pd.isna(v) else f"{v:.0f}") for v in df[c]]
        elif " vs " in f" {c} " or c.startswith("vs") or c.endswith("vs SPY"):
            out[c] = [("" if pd.isna(v) else f"{v:+.1%}") for v in df[c]]
        else:
            out[c] = [("" if pd.isna(v) else f"{v:.1%}") for v in df[c]]
    return out


def dip_report() -> str:
    from .strategies import etf_bars
    eb = etf_bars()
    bars = {t: eb[eb["ticker"] == t].set_index("date")[["o", "h", "l", "c"]].astype(float).sort_index()
            for t in ("QQQ", "SPY", "TQQQ", "BIL")}
    gbpusd, cash = sterling()
    lq = fill_stale_prints(fetch_lqq3(max_age_days=4), bars["QQQ"]["c"], 3, gbpusd)
    rows, curves, trades = {}, {}, {}
    bench = None
    for label, kw in dip_variants():
        on = dip_state(bars["QQQ"], kw["entry"], kw["leave"], kw["trend"])
        eq, _ = lqq3_strategy(lq, bars["QQQ"]["c"], cash, at_close=kw["at_close"], on=on)
        if bench is None:
            spy = bars["SPY"]["c"][bars["SPY"].index >= eq.index[0]]
            bench = to_gbp(spy / spy.iloc[0], gbpusd)
        t = lqq3_trades(lq, on, at_close=kw["at_close"])
        rows[label], curves[label], trades[label] = dip_measures(eq, t, bench), eq, t
    tbl = pd.DataFrame(rows).T
    kws = dict(dip_variants())
    ok = tbl[tbl["2017-21 win rate"] >= ISA_WIN_RATE]
    pick = ok["2017-21"].idxmax() if len(ok) else None
    hit = tbl[(tbl["Win rate"] >= ISA_WIN_RATE) & (tbl["vs S&P 500"] >= ISA_EXCESS)]
    both = hit[(hit["2017-21 vs S&P"] >= ISA_EXCESS) & (hit["2022-26 vs S&P"] >= ISA_EXCESS)]
    sp = {k: stats(v)["CAGR"] for k, v in (("all", bench), ("a", bench[bench.index <= pd.Timestamp(SPLIT)]),
                                           ("b", bench[bench.index >= pd.Timestamp(SPLIT)]))}
    trend_eq, _ = lqq3_strategy(lq, bars["QQQ"]["c"], cash)
    hold = lq["close"][lq.index >= trend_eq.index[0]]
    refs = {"LQQ3 when QQQ > 200-day avg (your current ISA rule)": trend_eq, "LQQ3 buy and hold": hold / hold.iloc[0]}
    main = ["CAGR", "vs S&P 500", "2017-21 vs S&P", "2022-26 vs S&P", "Win rate", "Trades per year",
            "Avg hold (sessions)", "Max drawdown"]
    shown = ([pick] if pick else []) + [x for x in both.sort_values("CAGR", ascending=False).index if x != pick]
    us = pd.DataFrame({k: us_version(bars, kws[k]["entry"], kws[k]["leave"], kws[k]["trend"]) for k in shown[:8]}).T
    first, last = trend_eq.index[0].date(), trend_eq.index[-1].date()
    by_fill = tbl.groupby(np.where(tbl.index.str.endswith("close"), "next London close", "next London open")).agg(
        settings=("CAGR", "size"), median_cagr=("CAGR", "median"), median_vs=("vs S&P 500", "median"),
        median_late=("2022-26 vs S&P", "median"), median_win=("Win rate", "median"),
        passed=("vs S&P 500", lambda v: int((v >= ISA_EXCESS).sum())))
    by_fill.columns = ["Settings", "Median CAGR", "Median vs S&P", "Median 2022-26 vs S&P", "Median win rate",
                       f"Beat the S&P by {ISA_EXCESS:.0%}+"]
    for c in ("Median CAGR", "Median win rate"):
        by_fill[c] = by_fill[c].map(lambda v: f"{v:.1%}")
    for c in ("Median vs S&P", "Median 2022-26 vs S&P"):
        by_fill[c] = by_fill[c].map(lambda v: f"{v:+.1%}")
    lines = [
        f"# Daily dip trading in LQQ3 for a sterling ISA ({first} to {last})", "",
        f"Generated by `python -m backtest.lqq3 dips` on {dt.date.today()}. The rules of backtest/daily.py, signalled "
        "by QQQ at the US close (9pm UK), traded in LQQ3 (3x Nasdaq-100, London, pence) at the next London open, or "
        f"at the next London close (the US market is then open); {COST_BPS:g} bps a side; idle cash earns sterling "
        f"interest (FXB). Benchmark: the S&P 500 in pounds, {sp['all']:.1%} a year ({sp['a']:.1%} in 2017-21, "
        f"{sp['b']:.1%} in 2022-26). Targets: at least {ISA_WIN_RATE:.0%} winning trades and {ISA_EXCESS:.0%} a "
        "year above the S&P 500.", "",
        "## Answer", "",
        f"{len(tbl)} settings tested; {len(hit)} reached both targets over the whole period and {len(both)} in both "
        "2017-21 and 2022-26. Pick chosen on 2017-21 only (best return among settings that won at least "
        f"{ISA_WIN_RATE:.0%} of their 2017-21 trades), and the settings that passed in both halves:", "",
        md(dfmt(tbl.loc[shown, main]), "Setting"), "",
        "All settings, by when LQQ3 is traded:", "", md(by_fill, "Fill"), "",
        "The same rules in the US (TQQQ at the next US open, in dollars, against SPY in dollars), to show what "
        "trading in London costs:", "",
        md(dfmt(us), "Setting"), "",
        "For comparison:", "",
        md(pd.DataFrame({k: {"CAGR": f"{stats(v)['CAGR']:.1%}", "vs S&P 500": f"{stats(v)['CAGR'] - sp['all']:+.1%}",
                             "Max drawdown": f"{stats(v)['Max drawdown']:.0%}"} for k, v in refs.items()}).T, "Rule"),
        ""]
    if pick:
        detail = ["Win rate", "2017-21 win rate", "2022-26 win rate", "Trades per year", "Avg hold (sessions)",
                  "Avg win", "Avg loss", "Worst trade", "Time invested", "Max drawdown", "Worst year"]
        cost_rows = {}
        for bps in (20.0, 30.0):
            on = dip_state(bars["QQQ"], kws[pick]["entry"], kws[pick]["leave"], kws[pick]["trend"])
            eq, _ = lqq3_strategy(lq, bars["QQQ"]["c"], cash, at_close=kws[pick]["at_close"], on=on, cost_bps=bps)
            t = lqq3_trades(lq, on, at_close=kws[pick]["at_close"], cost_bps=bps)
            cost_rows[f"{bps:g} bps a side"] = dip_measures(eq, t, bench)
        years = pd.DataFrame({pick: yearly_returns(curves[pick]), "S&P 500 in pounds": yearly_returns(bench)})
        worst = trades[pick].sort_values("ret").head(5)
        lines += ["## The 2017-21 pick in detail", "", md(dfmt(tbl.loc[[pick], detail]).T, "Measure"), "",
                  "With higher trading costs (spread plus any platform commission, per side):", "",
                  md(dfmt(pd.DataFrame(cost_rows).T[["CAGR", "vs S&P 500", "2022-26 vs S&P", "Win rate"]]), "Costs"),
                  "", "Calendar years:", "", md(years.map(lambda v: f"{v:+.1%}"), "Year"), "",
                  "Worst trades: " + ", ".join(f"{r.entry.date()} to {r.exit.date()} {r.ret:+.1%}"
                                               for r in worst.itertuples()) + ".", ""]
    lines += ["## Caveats", "",
              "- LQQ3 is the only 3x fund tested, because it is the only London price series approved for this "
              "repository. The US multi-fund result (backtest/DAILY.md) relied on nine funds; most have no London "
              "equivalent with a long history (WisdomTree's S&P 500 3x, 3LUS / 3USL, does).",
              f"- {len(tbl)} settings were tried; the whole-period winners are flattered by selection. The pick was "
              "chosen on 2017-21 only, so its 2022-26 column is the honest test.",
              "- The London open of a 3x fund can have a wide spread; the close (16:30 UK, with the US market open) "
              "is usually tighter. Platform commissions come on top; inside an ISA there is no tax on the gains.",
              "- A 3x fund can lose most of its value in a crash; nothing can be sold before the next London "
              "session after a US-close signal.",
              "- On days nothing traded, Yahoo often repeats LQQ3's last price (about a third of 2018); those days are "
              "valued at the price implied by 3x QQQ in pounds (fill_stale_prints). Trading at the stale prices "
              "instead would move single years by tens of points, so 2017-18 are the least certain years."]
    out = "\n".join(lines)
    out_dir = paths().root / "backtests" / "lqq3"
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_dir / "dip_variants.csv")
    return out


def main(argv: list[str] | None = None) -> None:
    import sys
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    dips = "dips" in (sys.argv[1:] if argv is None else argv)
    text = dip_report() if dips else report()
    (REPO_ROOT / "backtest" / ("LQQ3_DAILY.md" if dips else "LQQ3.md")).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

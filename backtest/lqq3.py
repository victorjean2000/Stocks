"""LQQ3 (WisdomTree NASDAQ 100 3x Daily Leveraged, London Stock Exchange, priced in pence) in a sterling ISA.

The trend rule: hold LQQ3 while QQQ closes above its 200-day average, otherwise cash. The signal comes at
the US close, after London has shut, so each switch trades at LQQ3's open in the next London session.

Sources: LQQ3.L daily prices from Yahoo Finance, the one source outside Alpaca and SEC EDGAR, approved by
the user on 2026-11-05 for LQQ3 only (cached in data/benchmarks/, never committed). Everything else comes
from Alpaca: QQQ for the signal, SPY for the S&P 500, and FXB (Invesco CurrencyShares British Pound) for the
pound/dollar rate and for sterling cash interest (what FXB's pound deposits paid after its 0.40% fee, about
0% until 2022: conservative next to a money-market fund).

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


def fetch_lqq3(max_age_days: int | None = None) -> pd.DataFrame:
    """Daily LQQ3 prices in pence (open, high, low, close, volume), bad prints removed.

    max_age_days: reuse the cache if its last bar is at most this old (None: always download).
    Today's bar is dropped until London has closed (16:30 plus a margin)."""
    path = paths().root / "benchmarks" / f"{SYMBOL}.parquet"
    if max_age_days is not None and path.exists():
        df = pd.read_parquet(path)
        if (pd.Timestamp.now(tz=LONDON).tz_localize(None).normalize() - df.index.max()).days <= max_age_days:
            return df
    r = requests.get(YAHOO_CHART + SYMBOL, params={"period1": 0, "period2": int(time.time()), "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    df = parse_chart(r.json())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def parse_chart(payload: dict, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Yahoo chart JSON -> daily bars in pence. A last bar whose close is still blank takes the quoted
    market price once London has closed that day; a bar of today before the close is left out."""
    res = payload["chart"]["result"][0]
    meta = res["meta"]
    if meta.get("currency") != "GBp":
        raise ValueError(f"{SYMBOL} is quoted in {meta.get('currency')}, expected GBp (pence)")
    q = res["indicators"]["quote"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(LONDON).tz_localize(None).normalize()
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")}, index=idx).astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    now = (now or pd.Timestamp.now(tz=LONDON)).tz_convert(LONDON)
    mkt = pd.Timestamp(meta.get("regularMarketTime", 0), unit="s", tz="UTC").tz_convert(LONDON)
    last = df.index[-1]
    if pd.isna(df["close"].iloc[-1]) and mkt.tz_localize(None).normalize() == last and \
            mkt.hour * 60 + mkt.minute >= 16 * 60 + 30 and meta.get("regularMarketPrice"):
        df.loc[last, "close"] = float(meta["regularMarketPrice"])
    if now.hour * 60 + now.minute < 16 * 60 + 45:
        df = df[df.index < now.tz_localize(None).normalize()]
    df = df.dropna(subset=["open", "close"])
    return drop_bad_prints(df)


def alpaca_daily(symbols: list[str], adjustment: str = "all", start: str = "2016-01-01") -> pd.DataFrame:
    bars = alpaca.fetch_bars(alpaca.session(), symbols, start, alpaca.safe_end(), adjustment, "sip")
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


def lqq3_strategy(lq: pd.DataFrame, qqq: pd.Series, cash_r: pd.Series, start: str = START, n: int = 200,
                  confirm: int = 1, at_close: bool = False) -> tuple[pd.Series, float]:
    """Equity (GBP, starting at 1) of the trend rule traded in LQQ3 on London sessions from `start`."""
    lse = lq.index
    o, c = lq[["open"]].rename(columns={"open": SYMBOL}), lq[["close"]].rename(columns={"close": SYMBOL})
    if at_close:  # fill at the close of the first London session after the signal instead of its open
        o = c.shift(1).fillna(o)
    book = Book.build(o, c, COST_BPS)
    on = regime(qqq.dropna(), n, confirm)
    i0 = int(lse.searchsorted(pd.Timestamp(start)))
    targets = schedule(on, lse, i0)
    if at_close:
        targets = {k + 1: v for k, v in targets.items() if k + 1 < len(lse)}
    cash = cash_r.reindex(lse).fillna(0.0).to_numpy()
    eq, turnover, _ = run(book, targets, cash, i0, lse)
    return eq, turnover


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
    lq = fetch_lqq3(max_age_days=4)
    us = alpaca_daily(["QQQ", "SPY"])
    gbpusd, cash = sterling()
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
        "removed (24 Oct 2025: a flat bar at +39% with no volume).",
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


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    text = report()
    (REPO_ROOT / "backtest" / "LQQ3.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

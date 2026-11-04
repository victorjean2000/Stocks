"""Indicators on wide frames (index = dates, columns = tickers), all using data up to each row only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close.rolling(n, min_periods=n).mean()


def atr_wilder(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Average true range with Wilder smoothing (RMA), as TradingView's ta.atr."""
    prev = close.shift(1)
    h, l, pc = high.to_numpy(), low.to_numpy(), prev.to_numpy()
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))  # fmax ignores a missing previous close
    tr = pd.DataFrame(tr, index=close.index, columns=close.columns)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def supertrend_bull(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
                    period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """1.0 when SuperTrend(period, mult) is bullish, 0.0 when bearish, NaN when unknown.

    Same recursion as TradingView ta.supertrend: bands on hl2 +- mult * ATR, the lower
    band only rises and the upper band only falls while price stays on their side, and
    the trend flips when the close crosses the active band.
    """
    atr = atr_wilder(high, low, close, period).to_numpy()
    hl2 = ((high + low) / 2).to_numpy()
    c = close.to_numpy()
    basic_up, basic_lo = hl2 + mult * atr, hl2 - mult * atr
    n_t, n_s = c.shape
    fu = np.full(n_s, np.nan)
    fl = np.full(n_s, np.nan)
    c_prev = np.full(n_s, np.nan)
    bull = np.zeros(n_s, dtype=bool)
    out = np.full((n_t, n_s), np.nan)
    for i in range(n_t):
        bu, bl, ci = basic_up[i], basic_lo[i], c[i]
        ok = np.isfinite(bu) & np.isfinite(ci)
        first = ok & np.isnan(fu)
        with np.errstate(invalid="ignore"):
            new_fl = np.where(np.isnan(fl) | (bl > fl) | (c_prev < fl), bl, fl)
            new_fu = np.where(np.isnan(fu) | (bu < fu) | (c_prev > fu), bu, fu)
            new_bull = np.where(bull, ~(ci < new_fl), ci > new_fu)
        new_bull = np.where(first, False, new_bull)  # TradingView starts in the down state
        fl = np.where(ok, new_fl, fl)
        fu = np.where(ok, new_fu, fu)
        bull = np.where(ok, new_bull, bull)
        c_prev = np.where(ok, ci, c_prev)
        out[i] = np.where(ok, bull.astype(float), np.nan)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def confirmed(state: pd.DataFrame, value: float, days: int) -> pd.DataFrame:
    """True where `state` equals `value` on each of the last `days` trading days."""
    hit = (state == value).astype(float).where(state.notna())
    return hit.rolling(days, min_periods=days).min().fillna(0).astype(bool)


def momentum_features(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """3-month, 6-month and 12-1-month returns (trading-day offsets 63, 126, 252/21)."""
    return {
        "r3": close / close.shift(63) - 1,
        "r6": close / close.shift(126) - 1,
        "r12_1": close.shift(21) / close.shift(252) - 1,
        "vol126": close.pct_change(fill_method=None).rolling(126, min_periods=100).std() * np.sqrt(252),
    }


def regime(close: pd.Series, n: int = 200, days: int = 3) -> pd.Series:
    """Market filter state, True = risk on.

    Switches off after `days` closes in a row below the n-day average and back on after
    `days` closes in a row above it; in between it keeps its last state (on during warm-up).
    """
    avg = close.rolling(n, min_periods=n).mean()
    side = pd.Series(np.where(close > avg, 1.0, np.where(close < avg, 0.0, np.nan)), index=close.index)
    side = side.where(avg.notna() & close.notna()).to_frame()
    off = confirmed(side, 0.0, days).iloc[:, 0].to_numpy()
    on = confirmed(side, 1.0, days).iloc[:, 0].to_numpy()
    state = pd.Series(np.where(off, 0.0, np.where(on, 1.0, np.nan)), index=close.index)
    return state.ffill().fillna(1.0).astype(bool)


def event_window(events: pd.DataFrame, index: pd.DatetimeIndex, columns, days: int) -> pd.DataFrame:
    """True on each event's session and on the `days` sessions before and after it.

    events: columns `ticker` and `session`; sessions not in `index` are ignored.
    """
    cols = pd.Index(columns)
    r = index.get_indexer(pd.DatetimeIndex(events["session"]).as_unit(index.unit))
    c = cols.get_indexer(events["ticker"])
    keep = (r >= 0) & (c >= 0)
    r, c = r[keep], c[keep]
    out = np.zeros((len(index), len(cols)), dtype=bool)
    for k in range(-days, days + 1):
        rk = r + k
        ok = (rk >= 0) & (rk < len(index))
        out[rk[ok], c[ok]] = True
    return pd.DataFrame(out, index=index, columns=cols)

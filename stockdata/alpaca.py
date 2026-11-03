"""Alpaca client: assets, trading calendar and daily bars.

Docs (checked 2026-11): GET /v2/stocks/bars takes a comma-separated `symbols`
list; `limit` (max 10000) counts bars across all symbols, results come symbol
by symbol and pages chain through `next_page_token`. `adjustment` is one of
raw/split/dividend/spin-off/all. `asof` defaults to today, so a current ticker
also returns history recorded under its previous names (FB -> META).
The free plan allows 200 requests/min and refuses SIP data from the last
15 minutes, so `end` is always at least 16 minutes in the past.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .http import HttpError, RateLimiter, get

log = logging.getLogger(__name__)

DATA_URL = "https://data.alpaca.markets"
TRADING_URL = os.environ.get("ALPACA_TRADING_URL", "https://paper-api.alpaca.markets")
NY = ZoneInfo("America/New_York")

# 200/min on the free plan; stay a little under it.
LIMITER = RateLimiter(185 / 60)
SIP_DELAY = dt.timedelta(minutes=16)


def session() -> requests.Session:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")
    s = requests.Session()
    s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                      "Accept": "application/json"})
    return s


def get_assets(sess: requests.Session) -> list[dict]:
    r = get(sess, f"{TRADING_URL}/v2/assets",
            params={"status": "active", "asset_class": "us_equity"}, limiter=LIMITER)
    return r.json()


def get_calendar(sess: requests.Session, start: str, end: str) -> pd.DataFrame:
    """Trading sessions with open/close as New York timestamps."""
    r = get(sess, f"{TRADING_URL}/v2/calendar", params={"start": start, "end": end}, limiter=LIMITER)
    cal = pd.DataFrame(r.json())
    cal["date"] = pd.to_datetime(cal["date"])
    cal["close_ts"] = [pd.Timestamp(f"{d.date()} {c}", tz=NY) for d, c in zip(cal["date"], cal["close"])]
    return cal[["date", "open", "close", "close_ts"]]


def last_complete_session(cal: pd.DataFrame, now: dt.datetime | None = None) -> pd.Timestamp:
    """Latest session whose close is at least the SIP delay in the past."""
    now = now or dt.datetime.now(dt.timezone.utc)
    done = cal[cal["close_ts"] + SIP_DELAY <= pd.Timestamp(now)]
    return done["date"].max()


def safe_end(now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    return (now - SIP_DELAY).strftime("%Y-%m-%dT%H:%M:%SZ")


class FeedChooser:
    """SIP by default; switch to IEX only if SIP is refused outright."""

    def __init__(self, preferred: str = "sip"):
        self.feed = preferred
        self.fell_back = False
        self.reason = ""

    def check(self, sess: requests.Session) -> str:
        params = {"symbols": "AAPL", "timeframe": "1Day", "start": "2016-01-04",
                  "end": "2016-01-08", "adjustment": "raw", "feed": self.feed}
        try:
            get(sess, f"{DATA_URL}/v2/stocks/bars", params=params, limiter=LIMITER)
        except HttpError as e:
            if e.status == 403 and self.feed == "sip":
                self.fell_back, self.reason, self.feed = True, e.body[:200], "iex"
                log.error("SIP feed refused (%s); FALLING BACK TO IEX", self.reason)
            else:
                raise
        return self.feed


BAR_FIELDS = ("o", "h", "l", "c", "v", "vw", "n")


def fetch_bars(sess: requests.Session, symbols: list[str], start: str, end: str,
               adjustment: str, feed: str) -> pd.DataFrame:
    """All daily bars for `symbols` (one request chain, any number of pages)."""
    cols: dict[str, list] = {k: [] for k in ("ticker", "date", *BAR_FIELDS)}
    token = None
    while True:
        params = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start, "end": end,
                  "adjustment": adjustment, "feed": feed, "limit": 10000, "sort": "asc"}
        if token:
            params["page_token"] = token
        data = get(sess, f"{DATA_URL}/v2/stocks/bars", params=params, limiter=LIMITER).json()
        for sym, bars in (data.get("bars") or {}).items():
            for b in bars:
                cols["ticker"].append(sym)
                cols["date"].append(b["t"][:10])  # midnight New York time, so the UTC date is the session date
                for k in BAR_FIELDS:
                    cols[k].append(b.get(k))
        token = data.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(cols)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for k in ("o", "h", "l", "c", "vw"):
        df[k] = df[k].astype("float64")
    df["v"] = df["v"].fillna(0).astype("int64")
    df["n"] = df["n"].fillna(0).astype("int64")
    return df


def fetch_bars_batched(sess: requests.Session, symbols: list[str], start: str, end: str,
                       adjustment: str, feed: str, batch_size: int = 100, workers: int = 4,
                       progress_every: int = 10) -> pd.DataFrame:
    """Split symbols into batches and fetch them in parallel under the shared rate limit."""
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    out: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_bars, sess, b, start, end, adjustment, feed): i for i, b in enumerate(batches)}
        for n, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if n % progress_every == 0 or n == len(batches):
                log.info("bars %s %s..%s: %d/%d batches", adjustment, start, end[:10], n, len(batches))
    out = [d for d in out if not d.empty]
    if not out:
        return pd.DataFrame(columns=["ticker", "date", *BAR_FIELDS])
    return pd.concat(out, ignore_index=True)


def save_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def split_factor_from_volumes(v_adj: pd.Series, v_raw: pd.Series) -> np.ndarray:
    """Cumulative split factor per day from split-adjusted / raw volume.

    With adjustment=all Alpaca scales volume only for splits, so the ratio is
    the product of all later split ratios (AAPL before 2020-08-31: 4.0). The
    ratio is noisy when volume is tiny (integer rounding) or zero, so each run
    of equal-ish ratios is replaced by its median over well-traded days.
    Input must be one ticker, sorted by date.
    """
    va = v_adj.to_numpy(dtype="float64")
    vr = v_raw.to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        # Both volumes >= 1000 keeps the integer-rounding error below 0.05%.
        r = np.where((vr >= 1000) & (va >= 1000), va / vr, np.nan)
    if np.all(np.isnan(r)):
        return np.ones(len(va))
    # Fill gaps from the next reliable day (factors only change at split dates).
    s = pd.Series(r).bfill().ffill().to_numpy()
    logs = np.log(s)
    # Segment boundaries where the factor jumps by more than 2%.
    seg = np.concatenate([[0], np.cumsum(np.abs(np.diff(logs)) > 0.02)])
    out = np.empty_like(s)
    for g in np.unique(seg):
        m = seg == g
        rel = r[m]
        rel = rel[~np.isnan(rel)]
        val = np.median(rel) if rel.size else np.median(s[m])
        out[m] = _snap(val)
    return out


def _snap(x: float) -> float:
    """Round a factor to a clean split ratio (e.g. 3.9998 -> 4, 0.12501 -> 0.125)."""
    for den in (1, 2, 3, 4, 5, 8, 10, 16, 20, 25, 40, 50, 100, 200, 1000):
        cand = round(x * den) / den
        if cand > 0 and abs(cand - x) / x < 2e-3:
            return cand
        inv = 1 / x
        cand = round(inv * den) / den
        if cand > 0 and abs(cand - inv) / inv < 2e-3:
            return 1 / cand
    return float(x)

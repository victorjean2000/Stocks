"""Daily price store (data/prices/daily.parquet), built from Alpaca bars.

Columns
  date, ticker
  open, high, low, close   adjustment=all (splits, dividends, spin-offs)
  close_raw                unadjusted close (as traded that day)
  volume                   unadjusted share volume
  vwap_raw, trades         unadjusted VWAP and trade count
  split_factor             cumulative split factor after this day; e.g. AAPL
                           before 2020-08-31 = 4.0. close_raw / split_factor
                           is the split-adjusted price, volume * split_factor
                           the split-adjusted volume.

Updates are incremental. Each run re-downloads the last few sessions for every
ticker; if Alpaca's adjusted values for those days changed (a new dividend or
split re-adjusts the whole history), that ticker's adjusted history is
re-downloaded in full. Raw history never changes, so it is only appended.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import alpaca
from .config import PRICE_START, paths

log = logging.getLogger(__name__)

OVERLAP_SESSIONS = 5
COLUMNS = ["date", "ticker", "open", "high", "low", "close", "close_raw", "volume", "vwap_raw", "trades",
           "split_factor"]


def store_path():
    return paths().prices / "daily.parquet"


def meta_path():
    return paths().prices / "meta.json"


def load_store() -> pd.DataFrame:
    p = store_path()
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_parquet(p)
    df["date"] = df["date"].astype("datetime64[ns]")
    return df


def _merge(adj: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    a = adj.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume_adj"})[
        ["ticker", "date", "open", "high", "low", "close", "volume_adj"]]
    r = raw.rename(columns={"c": "close_raw", "v": "volume", "vw": "vwap_raw", "n": "trades"})[
        ["ticker", "date", "close_raw", "volume", "vwap_raw", "trades"]]
    m = a.merge(r, on=["ticker", "date"], how="outer", indicator=True)
    bad = m["_merge"] != "both"
    if bad.any():
        log.warning("%d bars present in only one of adjusted/raw responses (dropped)", int(bad.sum()))
    m = m[~bad].drop(columns="_merge")
    m["date"] = m["date"].astype("datetime64[ns]")
    return m


def _add_split_factor(df: pd.DataFrame) -> pd.DataFrame:
    """df needs ticker, date, volume (raw), volume_adj. Sorted output."""
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    sf = np.ones(len(df))
    starts = np.flatnonzero(np.r_[True, df["ticker"].to_numpy()[1:] != df["ticker"].to_numpy()[:-1]])
    ends = np.r_[starts[1:], len(df)]
    va = df["volume_adj"].to_numpy()
    vr = df["volume"].to_numpy()
    for s, e in zip(starts, ends):
        seg_va, seg_vr = va[s:e], vr[s:e]
        if np.array_equal(seg_va, seg_vr):
            continue  # no splits in this ticker's history
        sf[s:e] = alpaca.split_factor_from_volumes(pd.Series(seg_va), pd.Series(seg_vr))
    df["split_factor"] = sf
    return df


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLUMNS].copy()
    df["volume"] = df["volume"].astype("int64")
    df["trades"] = df["trades"].astype("int64")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def _write(df: pd.DataFrame, meta: dict) -> None:
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False, compression="zstd", row_group_size=500_000)
    tmp.replace(p)
    alpaca.save_json(meta, meta_path())


def _fetch_full_history(sess, symbols: list[str], end: str, feed: str, batch_size: int = 100,
                        workers: int = 4) -> pd.DataFrame:
    """Adjusted + raw history, merged batch by batch so peak memory stays small."""
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]

    def one(batch):
        adj = alpaca.fetch_bars(sess, batch, PRICE_START, end, "all", feed)
        raw = alpaca.fetch_bars(sess, batch, PRICE_START, end, "raw", feed)
        if adj.empty or raw.empty:
            return None
        return _finish(_add_split_factor(_merge(adj, raw)))

    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, b) for b in batches]
        for n, f in enumerate(as_completed(futs), 1):
            df = f.result()
            if df is not None:
                out.append(df)
            if n % 10 == 0 or n == len(batches):
                log.info("full history: %d/%d batches", n, len(batches))
    # Multi-symbol paging occasionally skips a symbol entirely (seen for DOW, whose
    # symbol was used by Dow Chemical before). Ask for those one at a time.
    got = set().union(*[set(d["ticker"].unique()) for d in out]) if out else set()
    missing = [s for s in symbols if s not in got]
    if missing:
        log.info("retrying %d symbols one by one: %s", len(missing), ",".join(missing[:20]))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for df in pool.map(lambda s: one([s]), missing):
                if df is not None:
                    out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLUMNS)


def update(symbols: list[str], full: bool = False, now: dt.datetime | None = None) -> dict:
    """Bring the store up to the latest complete session for `symbols`."""
    t0 = time.time()
    sess = alpaca.session()
    feed = alpaca.FeedChooser("sip")
    feed.check(sess)
    cal = alpaca.get_calendar(sess, PRICE_START, (pd.Timestamp.today() + pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
    last_session = alpaca.last_complete_session(cal, now)
    end = alpaca.safe_end(now)
    sessions = cal.loc[cal["date"] <= last_session, "date"].reset_index(drop=True)

    store = pd.DataFrame(columns=COLUMNS) if full else load_store()
    have = set(store["ticker"].unique()) if len(store) else set()
    want = sorted(set(symbols))
    new_syms = [s for s in want if s not in have]
    old_syms = [s for s in want if s in have]
    stats = {"feed": feed.feed, "fell_back_to_iex": feed.fell_back, "fallback_reason": feed.reason,
             "last_session": str(last_session.date()), "new_symbols": len(new_syms), "readjusted": 0,
             "appended_rows": 0}

    parts = []
    if new_syms:
        log.info("full history for %d symbols", len(new_syms))
        parts.append(_fetch_full_history(sess, new_syms, end, feed.feed))

    keep = store
    if old_syms and len(store):
        last_by_ticker = store.groupby("ticker")["date"].max()
        global_last = last_by_ticker.max()
        idx = int(sessions.searchsorted(global_last))
        win_start = sessions.iloc[max(0, idx - OVERLAP_SESSIONS)]
        stale = [s for s in old_syms if last_by_ticker[s] < win_start]
        fresh = [s for s in old_syms if last_by_ticker[s] >= win_start]
        window = []
        groups = [(win_start, fresh)] + [(last_by_ticker[s] - pd.Timedelta(days=10), [s]) for s in stale]
        for start, syms in groups:
            if not syms:
                continue
            a = alpaca.fetch_bars_batched(sess, syms, start.strftime("%Y-%m-%d"), end, "all", feed.feed, batch_size=200)
            r = alpaca.fetch_bars_batched(sess, syms, start.strftime("%Y-%m-%d"), end, "raw", feed.feed, batch_size=200)
            if len(a) and len(r):
                window.append(_merge(a, r))
        win = pd.concat(window, ignore_index=True) if window else pd.DataFrame()
        readj = _readjusted(store, win)
        stats["readjusted"] = len(readj)
        if readj:
            log.info("%d tickers re-adjusted by Alpaca (new dividend/split): re-downloading adjusted history",
                     len(readj))
            a = alpaca.fetch_bars_batched(sess, sorted(readj), PRICE_START, end, "all", feed.feed)
            a = a.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume_adj"})
            raw_hist = store[store["ticker"].isin(readj)][["ticker", "date", "close_raw", "volume", "vwap_raw", "trades"]]
            raw_new = win[win["ticker"].isin(readj)][["ticker", "date", "close_raw", "volume", "vwap_raw", "trades"]] \
                if len(win) else raw_hist.iloc[:0]
            raw_all = pd.concat([raw_hist, raw_new]).drop_duplicates(["ticker", "date"], keep="last")
            a["date"] = a["date"].astype("datetime64[ns]")
            redo = a[["ticker", "date", "open", "high", "low", "close", "volume_adj"]].merge(
                raw_all, on=["ticker", "date"], how="inner")
            parts.append(_add_split_factor(redo))
        if len(win):
            w = win[~win["ticker"].isin(readj)]
            # Overwrite the overlap days (picks up late corrections) and append new days.
            key = pd.MultiIndex.from_frame(w[["ticker", "date"]])
            skey = pd.MultiIndex.from_frame(store[["ticker", "date"]])
            keep = store[~skey.isin(key) & ~store["ticker"].isin(readj)]
            # Split factor of new rows: today's adjusted volume / raw volume (1.0 unless a split
            # happened, and then the ticker is in `readj` and handled above).
            w = w.copy()
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(w["volume"] > 0, w["volume_adj"] / w["volume"], 1.0)
            w["split_factor"] = [alpaca._snap(x) if np.isfinite(x) and x > 0 else 1.0 for x in ratio]
            stats["appended_rows"] = int((~key.isin(skey)).sum())
            parts.append(w)
        else:
            keep = store[~store["ticker"].isin(readj)]
    out = pd.concat([keep] + [_finish(p_) for p_ in parts], ignore_index=True) if parts else store
    out = out[out["date"] <= last_session]
    out = _finish(out.drop_duplicates(["ticker", "date"], keep="last"))
    meta = {**stats, "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "rows": int(len(out)), "tickers": int(out["ticker"].nunique()), "seconds": round(time.time() - t0, 1)}
    _write(out, meta)
    log.info("prices: %s", meta)
    return meta


def _readjusted(store: pd.DataFrame, win: pd.DataFrame, tol: float = 1e-6) -> set[str]:
    """Tickers whose adjusted close on overlapping days no longer matches the store."""
    if not len(win) or not len(store):
        return set()
    m = win[["ticker", "date", "close"]].merge(store[["ticker", "date", "close"]], on=["ticker", "date"],
                                                suffixes=("_new", "_old"))
    rel = (m["close_new"] - m["close_old"]).abs() / m["close_old"].abs().clip(lower=1e-9)
    return set(m.loc[rel > tol, "ticker"])

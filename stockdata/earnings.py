"""Earnings release dates from SEC 8-K filings under Item 2.02.

A US company that announces results for a completed quarter or year furnishes an 8-K
under Item 2.02 ("Results of Operations and Financial Condition"), normally within
minutes of the press release. submissions.zip lists every filing with its items and its
EDGAR acceptance time (UTC, to the second), so each release can be placed before the
open, during the session or after the close.

Output: data/sec/earnings_events.parquet, one row per ticker and release:
  ticker, cik, filer_cik, accession, filed, accepted (UTC), timing, session
`session` is the first trading session whose close comes after the release: released
before 16:00 New York time on a trading day -> that day, otherwise the next trading day.
It is the first session whose price can react to the numbers.

Limits: foreign private issuers announce results on 6-K, which has no item codes, so
they have no events; amendments (8-K/A) are ignored; a few companies furnish Item 2.02
for other things too (preliminary figures, investor days), which count as extra events.
"""
from __future__ import annotations

import json
import logging
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .config import PRICE_START, paths

log = logging.getLogger(__name__)

ITEM = "2.02"
FORMS = ("8-K",)
SINCE = str((pd.Timestamp(PRICE_START) - pd.Timedelta(days=183)).date())  # covers windows at the start
NEW_YORK = "America/New_York"
OPEN_HOUR, CLOSE_HOUR = 9.5, 16.0
COLUMNS = ["ticker", "cik", "filer_cik", "accession", "filed", "accepted", "timing", "session"]


def _rows(block: dict, cik: int) -> list[tuple]:
    """Item 2.02 8-Ks from one columnar filings block of a submissions JSON."""
    forms = block.get("form", [])
    items = block.get("items") or [""] * len(forms)
    out = []
    for form, it, filed, acc, accn in zip(forms, items, block.get("filingDate", []),
                                          block.get("acceptanceDateTime", []),
                                          block.get("accessionNumber", [])):
        if form in FORMS and filed >= SINCE and ITEM in (it or "").split(","):
            out.append((cik, accn, form, it, filed, acc))
    return out


def _worker(args) -> list[tuple]:
    zip_path, ciks = args
    out: list[tuple] = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for cik in ciks:
            n = f"CIK{cik:010d}.json"
            if n not in names:
                continue
            filings = json.loads(z.read(n)).get("filings", {})
            out += _rows(filings.get("recent", {}), cik)
            for f in filings.get("files", []):  # older filings live in CIK...-submissions-00N.json
                if f.get("filingTo", "") >= SINCE and f.get("name") in names:
                    out += _rows(json.loads(z.read(f["name"])), cik)
    return out


def extract_filings(ciks, workers: int = 4) -> pd.DataFrame:
    """Item 2.02 8-K filings since SINCE for these CIKs, read from submissions.zip."""
    zip_path = paths().raw_sec / "submissions.zip"
    ciks = sorted(set(int(c) for c in ciks))
    n = max(1, min(len(ciks), workers * 4))
    chunks = [(zip_path, ciks[i::n]) for i in range(n)]
    rows: list[tuple] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(_worker, chunks):
                rows.extend(part)
    else:
        for c in chunks:
            rows.extend(_worker(c))
    df = pd.DataFrame(rows, columns=["cik", "accession", "form", "items", "filed", "accepted"])
    df = df.drop_duplicates(["cik", "accession"])
    df["filed"] = pd.to_datetime(df["filed"]).astype("datetime64[ns]")
    df["accepted"] = pd.to_datetime(df["accepted"], utc=True).astype("datetime64[ns, UTC]")
    return df.reset_index(drop=True)


def reaction_session(accepted: pd.Series, calendar) -> tuple[pd.Series, pd.Series]:
    """(session, timing) for UTC acceptance times, given the trading sessions.

    session: first session whose close is after the release (before 16:00 New York time
    on a session -> that session, otherwise the next one). timing: before_open,
    during_session, after_close or non_trading_day (New York time).
    """
    et = pd.to_datetime(accepted, utc=True).dt.tz_convert(NEW_YORK)
    day = et.dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    hour = (et.dt.hour + et.dt.minute / 60 + et.dt.second / 3600).to_numpy()
    cal = pd.DatetimeIndex(calendar).astype("datetime64[ns]").normalize().unique().sort_values()
    d = pd.DatetimeIndex(day)
    if len(cal) and d.notna().any():  # outside the known sessions, weekdays stand in for sessions
        lo, hi = min(d.min(), cal[0]), max(d.max(), cal[-1])
        before = pd.bdate_range(lo - pd.Timedelta(days=7), cal[0] - pd.Timedelta(days=1))
        after = pd.bdate_range(cal[-1] + pd.Timedelta(days=1), hi + pd.Timedelta(days=14))
        cal = before.append(cal).append(after).astype("datetime64[ns]")
    on_session = d.isin(cal)
    same = on_session & (hour < CLOSE_HOUR)
    pos = np.where(same, cal.searchsorted(d, side="left"), cal.searchsorted(d, side="right"))
    ok = pos < len(cal)
    session = pd.Series(pd.NaT, index=accepted.index, dtype="datetime64[ns]")
    session[ok] = cal[pos[ok]]
    timing = np.select([~on_session, hour < OPEN_HOUR, hour < CLOSE_HOUR],
                       ["non_trading_day", "before_open", "during_session"], "after_close")
    return session, pd.Series(timing, index=accepted.index)


def _predecessors() -> dict[int, list[int]]:
    """CIK -> earlier CIKs of the same company (successors.py links, followed transitively)."""
    f = paths().reports / "successor_links.csv"
    if not f.exists():
        return {}
    links = pd.read_csv(f)
    direct = links.groupby("cik")["predecessor_cik"].apply(lambda s: [int(x) for x in s]).to_dict()
    out = {}
    for cik in direct:
        seen, todo = [], list(direct[cik])
        while todo:
            c = todo.pop()
            if c not in seen and c != cik:
                seen.append(c)
                todo += direct.get(c, [])
        out[int(cik)] = seen
    return out


def build(securities: pd.DataFrame, calendar, filings: pd.DataFrame | None = None,
          workers: int = 4) -> pd.DataFrame:
    """Per-ticker earnings events for the kept securities -> data/sec/earnings_events.parquet.

    A company that moved to a new CIK (holding-company reorganisation, see successors.py)
    keeps the releases its predecessor filed before the new CIK's first one.
    """
    keep = securities.loc[securities["keep"] & securities["cik"].notna(), ["ticker", "cik"]].copy()
    keep["cik"] = keep["cik"].astype(int)
    pred = _predecessors()
    pairs = []
    for t, c in keep.itertuples(index=False):
        pairs.append((t, c, c))
        pairs += [(t, c, pc) for pc in pred.get(c, [])]
    m = pd.DataFrame(pairs, columns=["ticker", "cik", "filer_cik"])
    if filings is None:
        filings = extract_filings(m["filer_cik"].unique(), workers=workers)
    ev = m.merge(filings.drop(columns=["items"], errors="ignore").rename(columns={"cik": "filer_cik"}),
                 on="filer_cik")
    own_first = ev[ev["filer_cik"] == ev["cik"]].groupby("ticker")["accepted"].min()
    first = ev["ticker"].map(own_first)
    ev = ev[(ev["filer_cik"] == ev["cik"]) | first.isna() | (ev["accepted"] < first)].copy()
    ev["session"], ev["timing"] = reaction_session(ev["accepted"], calendar)
    ev = ev.sort_values(["ticker", "accepted"]).drop_duplicates(["ticker", "session"])
    ev = ev[COLUMNS].reset_index(drop=True)
    p = paths().ensure()
    ev.to_parquet(p.sec / "earnings_events.parquet", index=False)
    log.info("earnings events: %d releases for %d tickers", len(ev), ev["ticker"].nunique())
    return ev


def load(tickers=None) -> pd.DataFrame:
    f = paths().sec / "earnings_events.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found: run `python -m stockdata earnings` (needs "
                                "data/raw/sec/submissions.zip) or `python -m stockdata update --sec always`")
    df = pd.read_parquet(f)
    return df if tickers is None else df[df["ticker"].isin(list(tickers))].reset_index(drop=True)

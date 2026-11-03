"""SEC EDGAR inputs: ticker map, submissions (SIC, filer type), share facts.

Sources (all free, need a User-Agent header, max 10 requests/second):
- www.sec.gov/files/company_tickers_exchange.json  ticker -> CIK
- Archives/edgar/daily-index/bulkdata/submissions.zip   company metadata
- Archives/edgar/daily-index/xbrl/companyfacts.zip      XBRL facts (untagged by class)
- files/dera/data/financial-statement-data-sets/YYYYqN.zip  same facts with
  their class-of-stock breakdown, needed for multi-class companies (V, BRK)
"""
from __future__ import annotations

import email.utils
import json
import logging
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import requests

from .config import FILINGS_START, paths
from .http import RateLimiter, get

log = logging.getLogger(__name__)

SEC_LIMITER = RateLimiter(8)
TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
BULK_URLS = {
    "companyfacts": "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip",
    "submissions": "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
}
FSDS_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip"

ANNUAL_FORMS = ("10-K", "10-K/A", "10-KT", "10-KT/A", "20-F", "20-F/A", "40-F", "40-F/A")
PERIODIC_FORMS = ANNUAL_FORMS + ("10-Q", "10-Q/A")
# Forms only registered investment companies (funds) file.
FUND_FORMS = ("N-CSR", "N-CSRS", "NPORT-P", "NPORT-EX", "N-CEN", "N-30D", "485BPOS", "485APOS",
              "497K", "24F-2NT", "N-1A", "N-30B-2", "N-MFP2", "N-MFP3")

# Share-count concepts read from companyfacts, in rough order of preference.
SHARE_CONCEPTS = (
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
    ("us-gaap", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"),
    ("ifrs-full", "NumberOfSharesOutstanding"),
    ("ifrs-full", "WeightedAverageShares"),
)
FSDS_TAGS = tuple(c for _, c in SHARE_CONCEPTS)


def session() -> requests.Session:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError("Set SEC_USER_AGENT (e.g. 'Name email@example.com')")
    s = requests.Session()
    s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
    return s


# ---------------------------------------------------------------- downloads

def download(sess: requests.Session, url: str, dest: Path, force: bool = False) -> bool:
    """Download url to dest unless the local copy is already current.

    Returns True if a new file was written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = sess.head(url, timeout=60, allow_redirects=True)
    head.raise_for_status()
    size = int(head.headers.get("Content-Length", -1))
    lm = head.headers.get("Last-Modified")
    remote_mtime = email.utils.parsedate_to_datetime(lm).timestamp() if lm else None
    if dest.exists() and not force:
        st = dest.stat()
        if (size < 0 or st.st_size == size) and (remote_mtime is None or st.st_mtime >= remote_mtime - 1):
            log.info("up to date: %s", dest.name)
            return False
    part = dest.with_suffix(dest.suffix + ".part")
    r = get(sess, url, limiter=SEC_LIMITER, stream=True, timeout=300)
    with open(part, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    if size >= 0 and part.stat().st_size != size:
        raise RuntimeError(f"incomplete download of {url}: {part.stat().st_size} of {size} bytes")
    if dest.suffix == ".zip":
        with zipfile.ZipFile(part) as z:  # raises on a corrupt archive
            z.namelist()
    part.replace(dest)
    if remote_mtime:
        os.utime(dest, (remote_mtime, remote_mtime))
    log.info("downloaded %s (%.0f MB)", dest.name, dest.stat().st_size / 1e6)
    return True


def download_bulk(sess: requests.Session, names=("submissions", "companyfacts"), tickers: bool = True,
                  force: bool = False) -> None:
    p = paths().ensure()
    if tickers:
        r = get(sess, TICKERS_URL, limiter=SEC_LIMITER)
        (p.raw_sec / "company_tickers_exchange.json").write_bytes(r.content)
    for name in names:
        download(sess, BULK_URLS[name], p.raw_sec / f"{name}.zip", force=force)


def load_company_tickers() -> pd.DataFrame:
    d = json.loads((paths().raw_sec / "company_tickers_exchange.json").read_text())
    df = pd.DataFrame(d["data"], columns=d["fields"])
    return df.rename(columns={"name": "sec_name", "exchange": "sec_exchange"})


# ------------------------------------------------------------- submissions

def _company_record(d: dict) -> dict:
    rec = d.get("filings", {}).get("recent", {})
    forms = rec.get("form", [])
    dates = rec.get("filingDate", [])
    accns = rec.get("accessionNumber", [])
    docs = rec.get("primaryDocument", [])
    latest_annual = next(((f, dates[i], accns[i], docs[i]) for i, f in enumerate(forms)
                          if f in ANNUAL_FORMS), (None, None, None, None))
    sic = d.get("sic") or None
    return {
        "cik": int(d["cik"]),
        "sec_name": d.get("name"),
        "entity_type": d.get("entityType"),
        "sic": int(sic) if sic and str(sic).isdigit() else None,
        "sic_description": d.get("sicDescription") or None,
        "owner_org": d.get("ownerOrg") or None,
        "category": d.get("category") or None,
        "state_of_incorporation": d.get("stateOfIncorporation") or None,
        "fiscal_year_end": d.get("fiscalYearEnd") or None,
        "sec_tickers": ",".join(t for t in (d.get("tickers") or []) if t),
        "latest_annual_form": latest_annual[0],
        "latest_annual_date": latest_annual[1],
        "latest_annual_accn": latest_annual[2],
        "latest_annual_doc": latest_annual[3],
        "n_fund_forms": sum(f in FUND_FORMS for f in forms),
        "n_periodic_forms": sum(f in PERIODIC_FORMS for f in forms),
        "n_6k": sum(f == "6-K" for f in forms),
    }


def company_info(ciks: list[int]) -> pd.DataFrame:
    """One row per CIK from submissions.zip."""
    rows = []
    with zipfile.ZipFile(paths().raw_sec / "submissions.zip") as z:
        names = set(z.namelist())
        for cik in sorted(set(int(c) for c in ciks)):
            n = f"CIK{cik:010d}.json"
            if n not in names:
                continue
            rows.append(_company_record(json.loads(z.read(n))))
    df = pd.DataFrame(rows)
    df["is_foreign_filer"] = df["latest_annual_form"].fillna("").str.match(r"^(20-F|40-F)") | (
        df["latest_annual_form"].isna() & (df["n_6k"] > 0))
    return df


# ----------------------------------------------------------- companyfacts

def _facts_worker(args) -> list[dict]:
    zip_path, ciks = args
    out = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for cik in ciks:
            n = f"CIK{cik:010d}.json"
            if n not in names:
                continue
            facts = json.loads(z.read(n)).get("facts", {})
            for tax, concept in SHARE_CONCEPTS:
                units = facts.get(tax, {}).get(concept, {}).get("units", {})
                for r in units.get("shares", []):
                    if r.get("filed", "") < FILINGS_START:
                        continue
                    out.append({"cik": cik, "concept": concept, "start": r.get("start"), "end": r["end"],
                                "val": r["val"], "accn": r["accn"], "form": r.get("form"),
                                "filed": r["filed"], "fp": r.get("fp")})
    return out


def share_facts_companyfacts(ciks: list[int], workers: int = 4) -> pd.DataFrame:
    """Untagged (company-total) share counts from companyfacts.zip."""
    zip_path = paths().raw_sec / "companyfacts.zip"
    ciks = sorted(set(int(c) for c in ciks))
    chunks = [(zip_path, ciks[i::workers * 4]) for i in range(workers * 4)]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_facts_worker, chunks):
            rows.extend(part)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["source"] = "companyfacts"
    df["segments"] = None
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


# --------------------------------------------------- financial statement data sets

def fsds_quarters(first: str = FILINGS_START, until: pd.Timestamp | None = None) -> list[str]:
    until = until or pd.Timestamp.today()
    qs = pd.period_range(pd.Timestamp(first), until, freq="Q")
    return [f"{q.year}q{q.quarter}" for q in qs]


def _fsds_extract(zip_path: Path) -> pd.DataFrame:
    """Share-count rows (with class breakdown) from one quarterly FSDS zip."""
    with zipfile.ZipFile(zip_path) as z:
        sub = pd.read_csv(z.open("sub.txt"), sep="\t", dtype=str, usecols=[
            "adsh", "cik", "form", "filed", "accepted", "period", "fp"], quoting=3)
        with z.open("num.txt") as f:
            tbl = pacsv.read_csv(
                f,
                read_options=pacsv.ReadOptions(block_size=1 << 26),
                parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False,
                                                 invalid_row_handler=lambda row: "skip"),
                convert_options=pacsv.ConvertOptions(
                    include_columns=["adsh", "tag", "ddate", "qtrs", "uom", "segments", "coreg", "value"],
                    column_types={"adsh": pa.string(), "tag": pa.string(), "ddate": pa.string(),
                                  "qtrs": pa.int32(), "uom": pa.string(), "segments": pa.string(),
                                  "coreg": pa.string(), "value": pa.float64()},
                    strings_can_be_null=True))
    mask = pc.and_(pc.is_in(tbl["tag"], value_set=pa.array(FSDS_TAGS)), pc.equal(tbl["uom"], "shares"))
    num = tbl.filter(mask).to_pandas()
    num = num[num["coreg"].isna()]  # rows about co-registrants (subsidiaries) are not the parent
    df = num.merge(sub, on="adsh", how="inner")
    return df


def update_fsds(sess: requests.Session, workers: int = 4) -> pd.DataFrame:
    """Download any missing quarters and return all extracted share rows."""
    p = paths().ensure()
    quarters = fsds_quarters()
    cache_files = []
    todo = []
    for q in quarters:
        cache = p.raw_fsds / f"{q}_shares.parquet"
        cache_files.append(cache)
        if cache.exists():
            continue
        zp = p.raw_fsds / f"{q}.zip"
        try:
            if not zp.exists():
                download(sess, FSDS_URL.format(q=q), zp)
        except Exception as e:  # the current quarter is not published yet
            log.info("FSDS %s not available (%s)", q, str(e)[:80])
            continue
        todo.append((zp, cache))
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for (zp, cache), df in zip(todo, pool.map(_fsds_extract, [t[0] for t in todo])):
                df.to_parquet(cache, index=False)
                zp.unlink()  # keep only the small extract
                log.info("FSDS %s: %d share rows", cache.name, len(df))
    parts = [pd.read_parquet(c) for c in cache_files if c.exists()]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["cik"] = df["cik"].astype(int)
    return df


def share_facts_fsds(raw: pd.DataFrame, ciks: list[int]) -> pd.DataFrame:
    """FSDS rows in the companyfacts layout (one row per fact)."""
    if raw.empty:
        return raw
    df = raw[raw["cik"].isin(set(int(c) for c in ciks))].copy()
    end = pd.to_datetime(df["ddate"], format="%Y%m%d", errors="coerce")
    q = df["qtrs"].astype(int)
    start = [e - pd.DateOffset(months=3 * n) + pd.Timedelta(days=1) if n > 0 else pd.NaT
             for e, n in zip(end, q)]
    out = pd.DataFrame({
        "cik": df["cik"].astype(int), "concept": df["tag"], "start": pd.to_datetime(start),
        "end": end, "val": df["value"], "accn": df["adsh"], "form": df["form"],
        "filed": pd.to_datetime(df["filed"], format="%Y%m%d", errors="coerce"),
        "fp": df["fp"], "source": "fsds", "segments": df["segments"],
        "accepted": pd.to_datetime(df["accepted"], errors="coerce"),
    })
    return out[out["filed"] >= pd.Timestamp(FILINGS_START)]

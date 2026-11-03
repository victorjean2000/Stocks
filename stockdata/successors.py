"""Link a listed company's CIK to its predecessor CIK.

Holding-company reorganizations and redomiciles give the listed company a new
CIK while its filing history stays on the old one: XOM moved to
"ExxonMobil Holdings Corp" (CIK 2115436) in 2026, history under CIK 34088.
SEC's ticker file only knows the new CIK, so without this link the company
would have no share counts before the reorganization.

Evidence used (all from submissions.zip):
- the new CIK's filing list contains accession numbers filed by the old CIK
  (joint or predecessor filings, accession prefix = filer CIK);
- the old CIK has no ticker any more;
- the names match on their first 5+ letters ("exxonmobil...", "bunge...").
"""
from __future__ import annotations

import json
import logging
import re
import zipfile

import pandas as pd

from .config import paths

log = logging.getLogger(__name__)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _names_match(a: str, b: str, n: int = 5) -> bool:
    a, b = _norm(a), _norm(b)
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k >= n


def find_predecessors(candidates: list[int]) -> pd.DataFrame:
    """successor_cik -> predecessor_cik for the given (late-starting) CIKs."""
    p = paths()
    rows = []
    with zipfile.ZipFile(p.raw_sec / "submissions.zip") as zs, \
            zipfile.ZipFile(p.raw_sec / "companyfacts.zip") as zc:
        sub_names = set(zs.namelist())
        cf_names = set(zc.namelist())
        for cik in candidates:
            n = f"CIK{cik:010d}.json"
            if n not in sub_names:
                continue
            d = json.loads(zs.read(n))
            prefixes = {int(a[:10]) for a in d["filings"]["recent"].get("accessionNumber", [])}
            prefixes.discard(cik)
            best = None
            for pre in prefixes:
                pn = f"CIK{pre:010d}.json"
                if pn not in sub_names or pn not in cf_names:
                    continue
                pd_ = json.loads(zs.read(pn))
                if pd_.get("tickers"):
                    continue  # still listed itself: a related company, not a predecessor
                if not _names_match(d.get("name"), pd_.get("name")):
                    continue
                n_filings = len(pd_["filings"]["recent"].get("accessionNumber", []))
                if best is None or n_filings > best[2]:
                    best = (pre, pd_.get("name"), n_filings)
            if best:
                rows.append({"cik": cik, "predecessor_cik": best[0], "name": d.get("name"),
                             "predecessor_name": best[1]})
    out = pd.DataFrame(rows, columns=["cik", "predecessor_cik", "name", "predecessor_name"])
    out.to_csv(p.reports / "successor_links.csv", index=False)
    log.info("successor links: %d (%s)", len(out), ", ".join(out["name"].head(10)))
    return out


def late_starters(securities: pd.DataFrame, filings: pd.DataFrame, prices: pd.DataFrame,
                  slack_days: int = 120) -> list[int]:
    """Kept CIKs whose share history starts well after their price history (or never)."""
    s = securities[securities["keep"] & securities["cik"].notna()][["ticker", "cik"]].copy()
    s["cik"] = s["cik"].astype(int)
    first_px = prices.groupby("ticker")["date"].min()
    s["first_px"] = s["ticker"].map(first_px)
    s = s.dropna(subset=["first_px"]).groupby("cik")["first_px"].min()
    first_f = filings.groupby("cik")["filed"].min()
    late = [c for c, fp in s.items() if c not in first_f.index or first_f[c] > fp + pd.Timedelta(days=slack_days)]
    return late

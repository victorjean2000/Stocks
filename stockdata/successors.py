"""Link a listed company's CIK to its predecessor CIK.

Holding-company reorganizations and redomiciles give the listed company a new
CIK while its filing history stays on the old one: XOM moved to
"ExxonMobil Holdings Corp" (CIK 2115436) in 2026, history under CIK 34088.
SEC's ticker file only knows the new CIK, so without this link the company
would have no share counts before the reorganization.

A predecessor must be an XBRL filer with no ticker today, linked either by a
joint filing (an accession number filed by the old CIK appears in the new CIK's
list and the old CIK is a SPAC merging in) or by name: with legal words
removed (Inc, Corp, Ltd, plc, Holdings, Group...), one name is a prefix of the
other or of a former name ("bunge" / "bungeglobal", "quidel" / "quidelortho"). It must also hand over: its
last share count comes at most 400 days before the new CIK's first one, and the
two counts are within 2x (holding-company exchanges are usually 1:1). Finally the
predecessor's own public float (dei:EntityPublicFloat, USD) must fit its share
count times this ticker's price (15%-160%); without float data the core names
must be identical. Entities that are still listed (Brookfield Corp for BAM) can
never be a predecessor.
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


_LEGAL = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "sa", "nv",
          "ag", "se", "ab", "asa", "spa", "llc", "lp", "holdings", "holding", "group", "the", "new", "old", "de",
          "md", "bv", "pte", "pty", "lllp", "trust"}


def _core(name: str) -> str:
    """Name without legal-form words: 'Bunge Global SA' -> 'bungeglobal', 'BUNGE LTD' -> 'bunge'."""
    words = re.findall(r"[a-z0-9]+", (name or "").lower().replace("&", " "))
    return "".join(w for w in words if w not in _LEGAL)


def _names_match(a: str, b: str, n: int = 4) -> bool:
    """One core name is a prefix of the other (quidel / quidelortho, cigna / cigna)."""
    a, b = _core(a), _core(b)
    if min(len(a), len(b)) < n:
        return False
    return a.startswith(b) or b.startswith(a)


def _xbrl_filer_index(zs: zipfile.ZipFile, cf_names: set[str]) -> pd.DataFrame:
    """Name, former names and ticker status of every XBRL filer (companyfacts CIKs)."""
    rows = []
    for n in cf_names:
        try:
            d = json.loads(zs.read(n))
        except KeyError:
            continue
        rows.append({"cik": int(n[3:13]), "name": d.get("name") or "",
                     "former": [f.get("name") or "" for f in d.get("formerNames", [])],
                     "has_ticker": bool(d.get("tickers")), "sic": str(d.get("sic") or "")})
    return pd.DataFrame(rows)


def _cached_index(zs: zipfile.ZipFile, cf_names: set[str]) -> pd.DataFrame:
    """_xbrl_filer_index, cached until submissions.zip changes (it takes ~40s to build)."""
    p = paths()
    st = (p.raw_sec / "submissions.zip").stat()
    key = f"{st.st_size}-{int(st.st_mtime)}"
    cache = p.sec / "xbrl_filers.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        if len(df) and df["cache_key"].iloc[0] == key:
            return df.drop(columns="cache_key")
    df = _xbrl_filer_index(zs, cf_names)
    df.assign(cache_key=key).to_parquet(cache, index=False)
    return df


def _public_float(zc: zipfile.ZipFile, cik: int) -> pd.DataFrame:
    """dei:EntityPublicFloat (USD) facts of one CIK: end, val."""
    try:
        d = json.loads(zc.read(f"CIK{cik:010d}.json"))
    except KeyError:
        return pd.DataFrame(columns=["end", "val"])
    rows = d.get("facts", {}).get("dei", {}).get("EntityPublicFloat", {}).get("units", {}).get("USD", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["end", "val"])
    df["end"] = pd.to_datetime(df["end"])
    return df[["end", "val"]].drop_duplicates("end")


def _float_consistent(pf: pd.DataFrame, pred_filings: pd.DataFrame, closes: pd.Series) -> bool | None:
    """Is the predecessor's public float consistent with its shares x this ticker's price?

    None when there is nothing to compare (no float, or no price on those dates).
    """
    if pf.empty or closes is None or closes.empty:
        return None
    pred_filings = pred_filings.sort_values("as_of")
    votes = []
    for r in pf.sort_values("end").tail(3).itertuples(index=False):
        known = pred_filings[pred_filings["as_of"] <= r.end + pd.Timedelta(days=120)]
        px = closes[closes.index <= r.end]
        if known.empty or px.empty or (r.end - px.index[-1]).days > 10:
            continue
        implied = float(known["shares_reported"].iloc[-1]) * float(px.iloc[-1])
        if implied > 0:
            votes.append(0.15 <= r.val / implied <= 1.6)
    return None if not votes else (sum(votes) >= len(votes) / 2)


def find_predecessors(candidates: list[int], filings: pd.DataFrame, candidates_fn,
                      closes_by_cik: dict | None = None, listed_ciks: set[int] | None = None) -> pd.DataFrame:
    """successor_cik -> predecessor_cik for the given (late-starting) CIKs.

    filings: chosen share counts of the listed CIKs (cik, filed, shares_reported).
    candidates_fn(ciks) -> filing candidates (as shares.filing_values) for other CIKs.
    """
    p = paths()
    rows = []
    with zipfile.ZipFile(p.raw_sec / "submissions.zip") as zs, \
            zipfile.ZipFile(p.raw_sec / "companyfacts.zip") as zc:
        sub_names = set(zs.namelist())
        cf_names = set(zc.namelist()) & sub_names
        idx = _cached_index(zs, cf_names)
        succ = idx.set_index("cik")
        # Possible predecessors: XBRL filers that no kept security maps to today. (An old CIK can
        # keep listing its former ticker or a bond ticker, e.g. Navient LLC still shows NAVI.)
        if listed_ciks is not None:
            pool = idx[~idx["cik"].isin(listed_ciks)]
        else:
            pool = idx[~idx["has_ticker"]]
        # Evidence 1: accessions filed by another CIK (joint/predecessor filings).
        by_prefix = {}
        for cik in candidates:
            n = f"CIK{cik:010d}.json"
            if n in sub_names:
                d = json.loads(zs.read(n))
                by_prefix[cik] = {int(a[:10]) for a in d["filings"]["recent"].get("accessionNumber", [])} - {cik}
    # Evidence 2: a ticker-less XBRL filer with a matching name (looked up by 5-letter prefix).
    by_name: dict[str, list[tuple[int, str]]] = {}
    for c, nm, former in zip(pool["cik"], pool["name"], pool["former"]):
        for variant in [nm, *list(former)]:  # "BUNGELTD" was "BUNGE LTD"
            k = _core(variant)[:4]
            if len(k) == 4:
                by_name.setdefault(k, []).append((int(c), variant))
    # A joint filing alone only links to a blank-check company (a SPAC merging into the listed company);
    # otherwise affiliates filing about each other would be linked.
    spac = set(pool.loc[(pool["sic"] == "6770") | pool["name"].str.contains(r"\bAcquisition\b", case=False), "cik"])
    pairs = []
    for cik in candidates:
        if cik not in succ.index:
            continue
        names = [nm for nm in [succ.at[cik, "name"], *succ.at[cik, "former"]] if nm]
        hits = {c for nm in names for c, pn in by_name.get(_core(nm)[:4], []) if _names_match(pn, nm)}
        joint = by_prefix.get(cik, set()) & spac
        for pre in hits | joint:
            if pre != cik:
                pairs.append((cik, int(pre), pre in joint))
    if not pairs:
        out = pd.DataFrame(columns=["cik", "predecessor_cik", "name", "predecessor_name"])
        out.to_csv(p.reports / "successor_links.csv", index=False)
        return out
    pred_fv = candidates_fn(sorted({b for _, b, _ in pairs}))
    from .shares import choose
    pred = choose(pred_fv) if len(pred_fv) else pd.DataFrame(columns=["cik", "filed", "shares_reported"])
    pl = pred.sort_values("filed").groupby("cik").agg(last_filed=("filed", "last"), last_val=("shares_reported", "last"))
    own = filings.sort_values("filed").groupby("cik").agg(first_filed=("filed", "first"),
                                                         first_val=("shares_reported", "first"))
    names = idx.set_index("cik")["name"]
    closes_by_cik = closes_by_cik or {}
    zc = zipfile.ZipFile(p.raw_sec / "companyfacts.zip")
    for succ_cik, pre, joint in pairs:
        if pre not in pl.index:
            continue
        last_f, last_v = pl.loc[pre]
        if succ_cik in own.index:
            first_f, first_v = own.loc[succ_cik]
            gap = (first_f - last_f).days
            # contiguous hand-over and a continuous share count (1:1-ish exchange)
            if not (-120 <= gap <= 400 and 0.5 <= first_v / last_v <= 2.0):
                continue
        else:
            # no own count yet: the predecessor must have stopped filing recently
            gap = None
            if (pd.Timestamp.today() - last_f).days > 400:
                continue
        # Independent check: the predecessor's reported public float must fit its share count
        # times THIS ticker's price (a shell or an unrelated company with a similar name fails).
        ok = _float_consistent(_public_float(zc, pre), pred[pred["cik"] == pre], closes_by_cik.get(succ_cik))
        succ_names = [names.get(succ_cik, ""), *list(succ.at[succ_cik, "former"])]
        pred_names = [names.get(pre, ""), *list(succ.at[pre, "former"])] if pre in succ.index else [names.get(pre, "")]
        same_core = any(_core(a) == _core(b) for a in succ_names for b in pred_names if a and b)
        if ok is False or (ok is None and not (same_core or joint)):
            continue
        if succ_cik not in own.index and not (ok or joint):
            continue  # nothing to cross-check a count-less successor against
        rows.append({"cik": succ_cik, "predecessor_cik": pre, "name": names.get(succ_cik),
                     "predecessor_name": names.get(pre), "gap_days": gap,
                     "evidence": "joint filing (SPAC)" if joint else ("public float" if ok else "same name")})
    zc.close()
    out = pd.DataFrame(rows, columns=["cik", "predecessor_cik", "name", "predecessor_name", "gap_days", "evidence"])
    if len(out):  # one predecessor per successor: the one handing over closest in time
        out = out.assign(a=out["gap_days"].abs().fillna(1e9)).sort_values("a").drop_duplicates("cik").drop(columns="a")
    out.to_csv(p.reports / "successor_links.csv", index=False)
    log.info("successor links: %d (%s)", len(out), ", ".join(str(x) for x in out["name"].head(12)))
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

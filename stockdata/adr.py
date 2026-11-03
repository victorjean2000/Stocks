"""ADR detection and ADS ratio for foreign filers (20-F).

A 20-F filer's share count (dei / IFRS facts) is in ORDINARY shares. When the
US listing is an American Depositary Share, market cap needs
ordinary_shares / ordinary_per_ADS. Alpaca names sometimes say "American
Depositary Shares, each representing two ordinary shares"; otherwise the
cover page of the latest 20-F is read. Results are cached per filing.
"""
from __future__ import annotations

import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from . import sec
from .config import paths
from .http import get

log = logging.getLogger(__name__)

_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "eighteen": 18, "twenty": 20, "twenty-five": 25, "thirty": 30, "forty": 40, "forty-five": 45,
    "fifty": 50, "sixty": 60, "seventy-five": 75, "eighty": 80, "one hundred": 100, "hundred": 100,
    "two hundred": 200, "two hundred fifty": 250, "five hundred": 500, "one thousand": 1000,
    "a": 1, "an": 1, "single": 1,
}
_FRAC = {
    "half": 2, "third": 3, "fourth": 4, "quarter": 4, "fifth": 5, "sixth": 6, "eighth": 8, "tenth": 10,
    "twentieth": 20, "fortieth": 40, "fiftieth": 50, "hundredth": 100,
}
_NUMBER = (r"(?:\d+(?:\.\d+)?(?:\s*/\s*\d+(?:th|st|nd|rd)?)?|(?:one|two|three|four|five|six|seven|eight|nine|ten|"
           r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|eighteen|twenty|twenty-five|thirty|forty|forty-five|"
           r"fifty|sixty|seventy-five|eighty|one hundred|two hundred fifty|two hundred|five hundred|one thousand|"
           r"hundred|a|an|single)(?:[\s-]+(?:half|third|fourth|quarter|fifth|sixth|eighth|tenth|twentieth|"
           r"fortieth|fiftieth|hundredth)s?)?)(?:\s*\(\s*[\d./]+\s*\))?")
_SHARE = r"(?:ordinary|common|class\s+[a-z]\s+(?:ordinary|common)|series\s+[a-z]|[a-z]+\s+)?\s*(?:shares?|stock|units?)"
_ADS = r"(?:american\s+depositary\s+shares?|american\s+depository\s+shares?|ADSs?|ADRs?)"

# "ADSs, each representing five ordinary shares" / "each ADS represents 1/10 of one share"
RATIO_EACH = re.compile(
    _ADS + r"[^.;]{0,120}?\beach\s+(?:of\s+which\s+)?(?:representing|represents|represent|equal\s+to|"
           r"corresponding\s+to|representing\s+the\s+right\s+to\s+receive)\s+(" + _NUMBER + r")"
           r"(?:\s+of\s+(?:an?|one))?\s+" + _SHARE, re.I)
RATIO_EACH2 = re.compile(
    r"\beach\s+" + _ADS + r"\s+(?:representing|represents|represent|is\s+equal\s+to|corresponds\s+to)\s+(" +
    _NUMBER + r")(?:\s+of\s+(?:an?|one))?\s+" + _SHARE, re.I)
# "two ADSs represent one ordinary share"
RATIO_MANY = re.compile(
    r"\b(" + _NUMBER + r")\s+" + _ADS + r"\s+(?:together\s+)?(?:represent|represents|representing)\s+(" + _NUMBER +
    r")\s+" + _SHARE, re.I)
ADS_ON_COVER = re.compile(r"american\s+depositary\s+(?:shares?|receipts?)|american\s+depository\s+shares?|"
                          r"\bADSs?\b|\bADRs?\b", re.I)


def parse_number(s: str) -> float | None:
    s = s.lower().strip()
    s = re.sub(r"\(\s*[\d./]+\s*\)", "", s).strip()  # "five (5)" -> "five"
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*/\s*(\d+)(?:th|st|nd|rd)?", s)
    if m:
        return float(m.group(1)) / float(m.group(2))
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)
    parts = re.split(r"[\s-]+", s)
    if parts and parts[-1].rstrip("s") in _FRAC and len(parts) >= 2:
        num = _NUM.get(" ".join(parts[:-1]), None)
        if num is None:
            return None
        return num / _FRAC[parts[-1].rstrip("s")]
    return _NUM.get(s, _NUM.get(s.replace("-", " ")))


def parse_ratio(text: str) -> tuple[float | None, str]:
    """Ordinary shares per ADS, and the phrase it came from."""
    for rx in (RATIO_EACH, RATIO_EACH2):
        m = rx.search(text)
        if m:
            v = parse_number(m.group(1))
            if v:
                return v, m.group(0)[:200]
    m = RATIO_MANY.search(text)
    if m:
        a, b = parse_number(m.group(1)), parse_number(m.group(2))
        if a and b:
            return b / a, m.group(0)[:200]
    return None, ""


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<ix:header>.*?</ix:header>", " ", raw)
    raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


def fetch_doc_text(sess: requests.Session, cik: int, accn: str, doc: str,
                   after_header: int = 4_000_000, hard_cap: int = 24_000_000) -> str:
    """Start of a filing document as plain text.

    Inline-XBRL documents begin with a hidden <ix:header> that can be several
    MB, so reading continues until 4 MB past its end (or 4 MB total for
    plain HTML).
    """
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/{doc}"
    r = get(sess, url, limiter=sec.SEC_LIMITER, stream=True, timeout=180)
    buf = bytearray()
    stop_at = after_header
    seen_header = False
    for chunk in r.iter_content(chunk_size=1 << 16):
        buf.extend(chunk)
        if not seen_header:
            if b"<ix:header" in buf[:200_000] or b"<ix:header" in chunk:
                k = buf.find(b"</ix:header>")
                if k >= 0:
                    seen_header, stop_at = True, k + after_header
                else:
                    stop_at = hard_cap
        if len(buf) >= stop_at:
            break
    r.close()
    return html_to_text(buf.decode("utf-8", errors="replace"))


def check_filing(sess: requests.Session, cik: int, accn: str, doc: str) -> dict:
    try:
        text = fetch_doc_text(sess, cik, accn, doc)
    except Exception as e:  # keep going; the security is listed as a problem
        return {"cik": cik, "accn": accn, "cover_ads": None, "ratio": None, "ratio_text": "", "error": str(e)[:200]}
    end = text.lower().find("indicate by check mark")
    cover = text[: max(end, 4000) if end > 0 else 15000][:30000]
    ratio, phrase = parse_ratio(text)
    return {"cik": cik, "accn": accn, "cover_ads": bool(ADS_ON_COVER.search(cover)), "ratio": ratio,
            "ratio_text": phrase, "error": ""}


def update(securities: pd.DataFrame, workers: int = 4) -> pd.DataFrame:
    """Check the latest 20-F of every kept foreign filer (cached by accession)."""
    p = paths().ensure()
    cache_path = p.sec / "adr_checks.parquet"
    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(
        columns=["cik", "accn", "cover_ads", "ratio", "ratio_text", "error"])
    f = securities[securities["keep"] & securities["latest_annual_form"].fillna("").str.startswith("20-F")]
    f = f.dropna(subset=["latest_annual_accn", "latest_annual_doc"]).drop_duplicates("cik")
    done = set(cache.loc[cache["error"].fillna("") == "", "accn"])
    todo = f[~f["latest_annual_accn"].isin(done)]
    if len(todo):
        log.info("ADR check: reading %d 20-F cover pages", len(todo))
        sess = sec.session()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda r: check_filing(sess, int(r.cik), r.latest_annual_accn, r.latest_annual_doc),
                                 todo.itertuples(index=False)))
        new = pd.DataFrame(rows)
        cache = pd.concat([cache[~cache["accn"].isin(new["accn"])], new], ignore_index=True)
        cache.to_parquet(cache_path, index=False)
    return cache


def assign(securities: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    """Set security_type='adr' and ads_ratio (ordinary shares per ADS)."""
    s = securities.copy()
    s["ads_ratio"] = pd.NA
    s["ads_ratio_source"] = ""
    # 1) ratio stated in the Alpaca name
    for i, row in s[s["keep"]].iterrows():
        if re.search(_ADS, row["name"] or "", re.I):
            r, _ = parse_ratio(row["name"])
            if r:
                s.at[i, "ads_ratio"], s.at[i, "ads_ratio_source"] = r, "alpaca name"
    # 2) latest 20-F text
    c = checks.drop_duplicates("cik", keep="last").set_index("cik")
    fpi = s["keep"] & s["latest_annual_form"].fillna("").str.startswith("20-F") & s["cik"].notna()
    for i, row in s[fpi].iterrows():
        cik = int(row["cik"])
        if cik not in c.index:
            continue
        chk = c.loc[cik]
        name_has_ads = bool(re.search(_ADS, row["name"] or "", re.I))
        name_says_ordinary = bool(re.search(r"ordinary|common|registry|class [a-z]\b|capital stock",
                                            row["name"] or "", re.I))
        text_says_ads = bool(chk["cover_ads"]) or pd.notna(chk["ratio"])
        is_adr = name_has_ads or (text_says_ads and not name_says_ordinary)
        if is_adr:
            s.at[i, "security_type"] = "adr"
            if pd.isna(s.at[i, "ads_ratio"]) and pd.notna(chk["ratio"]):
                s.at[i, "ads_ratio"], s.at[i, "ads_ratio_source"] = float(chk["ratio"]), "20-F text"
    s["ads_ratio"] = pd.to_numeric(s["ads_ratio"], errors="coerce")
    return s

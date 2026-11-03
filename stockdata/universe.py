"""Security master: Alpaca assets -> SEC CIK, plus a security-type label.

Only `common` (and `adr`, kept for reference) get prices downloaded. Every
dropped symbol is written to universe/dropped.csv with the reason.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from . import sec
from .config import MAJOR_EXCHANGES, paths

log = logging.getLogger(__name__)

KEEP_TYPES = ("common", "adr")

_I = re.IGNORECASE
SYMBOL_RULES = [
    (re.compile(r"\.PR[A-Z]{0,2}$"), "preferred", "symbol suffix .PR (preferred)"),
    (re.compile(r"\.WS(\.[A-Z])?$"), "warrant", "symbol suffix .WS (warrant)"),
    (re.compile(r"\.U$"), "unit", "symbol suffix .U (unit)"),
    (re.compile(r"\.RT$"), "right", "symbol suffix .RT (right)"),
    (re.compile(r"\.WI$"), "when_issued", "symbol suffix .WI (when issued)"),
]
NAME_RULES = [
    (re.compile(r"\bwarrants?\b", _I), "warrant", "name: warrant"),
    (re.compile(r"\brights?\b", _I), "right", "name: right"),
    (re.compile(r"\bpreferred\b|\bpfd\.?\b|\bcumulative\b|\bperpetual\b|\bnon-?cumulative\b"
                r"|\btrust preferred\b|\bcapital securities\b|(?<!american )\bdepositary shares?\b", _I),
     "preferred", "name: preferred / depositary share"),
    (re.compile(r"\bETFs?\b|\bETNs?\b|exchange[- ]traded|\bETP\b", _I), "etf_fund", "name: ETF/ETN"),
    (re.compile(r"\bnotes?\b|\bdebentures?\b|\bdue \d{4}\b|\bdue (january|february|march|april|may|june|july"
                r"|august|september|october|november|december)\b|\bsenior secured\b|\bsubordinated\b", _I),
     "debt", "name: notes/debentures"),
    (re.compile(r"\bfunds?\b", _I), "etf_fund", "name: fund"),
    (re.compile(r"\bL\.?P\.?(?=$|[\s,])|\blimited partnership\b|\bcommon units\b|\bunits representing\b"
                r"|\bpartnership units\b|\bdepositary units\b", _I), "partnership", "name: limited partnership units"),
    (re.compile(r"\bunits?\b", _I), "unit", "name: unit"),
    (re.compile(r"\broyalty trust\b|\bunits of beneficial interest\b", _I), "trust_units", "name: royalty/unit trust"),
    (re.compile(r"\bwhen[- ]issued\b", _I), "when_issued", "name: when issued"),
]
ETF_FAMILY_RE = re.compile(
    r"\bProShares\b|ADRhedged|\bTrust,? Series\b|\biShares\b|\bSPDR\b|\bDirexion\b|\bVanEck\b"
    r"|\bInvesco QQQ\b|\bGraniteShares\b|\bDefiance\b|\bYieldMax\b|\bRoundhill\b|\bLeverage Shares\b"
    r"|\bTradr\b|\bT-Rex\b|\bMicroSectors\b|\bETRACS\b|\biPath\b", _I)
TRUST_FUND_RE = re.compile(r"\b(trust|fund)\b", _I)
ETF_VENUES = ("ARCA", "BATS", "NYSEARCA")
ADR_RE = re.compile(r"\bADRs?\b|\bADSs?\b|american deposit[ao]ry|depositary receipts?|\bsponsored\b", _I)
# Nasdaq fifth-letter codes for non-common securities (used only when the name is silent).
NASDAQ_SUFFIX = {"W": ("warrant", "nasdaq 5th letter W"), "R": ("right", "nasdaq 5th letter R"),
                 "U": ("unit", "nasdaq 5th letter U")}
COMMON_WORDS = re.compile(r"common|ordinary|capital stock|class [a-z]\b|shares", _I)


def classify(symbol: str, name: str, exchange: str, attributes: str = "",
             sec_info: dict | None = None) -> tuple[str, str]:
    """Return (security_type, reason). security_type 'common' means keep."""
    name = name or ""
    if exchange not in MAJOR_EXCHANGES:
        return "otc", f"exchange {exchange} is not a major US exchange"
    for rx, kind, why in SYMBOL_RULES:
        if rx.search(symbol):
            return kind, why
    sic = sec_info.get("sic") if sec_info else None
    if sic == 6221:
        return "etf_fund", "SIC 6221 commodity/crypto pool"
    if ETF_FAMILY_RE.search(name):
        return "etf_fund", "name: ETF issuer family"
    if sic == 6199 and TRUST_FUND_RE.search(name):
        return "etf_fund", "SIC 6199 trust/fund (commodity or crypto vehicle)"
    if exchange in ETF_VENUES and (sec_info is None or TRUST_FUND_RE.search(name)):
        return "etf_fund", f"{exchange} listing without SEC operating-company match (ETF venue)"
    # Business development companies file 10-K/10-Q, have no SIC and may be called "Fund".
    is_bdc_like = bool(sec_info) and sic is None and (sec_info.get("n_periodic_forms") or 0) > 0 and \
        (sec_info.get("n_fund_forms") or 0) == 0
    for rx, kind, why in NAME_RULES:
        if rx.search(name):
            if kind == "etf_fund" and why == "name: fund" and is_bdc_like:
                continue  # files 10-K/10-Q like an operating company (e.g. a BDC)
            return kind, why
    if "ptp_no_exception" in attributes or "ptp_with_exception" in attributes:
        return "partnership", "Alpaca PTP attribute (publicly traded partnership)"
    if sec_info is not None and (sec_info.get("n_fund_forms") or 0) > 0 and \
            (sec_info.get("n_periodic_forms") or 0) == 0:
        return "etf_fund", "files fund forms (N-CSR/N-PORT) and no 10-K/10-Q"
    if exchange == "NASDAQ" and len(symbol) == 5 and symbol[-1] in NASDAQ_SUFFIX and \
            not COMMON_WORDS.search(name):
        return NASDAQ_SUFFIX[symbol[-1]]
    if ADR_RE.search(name):
        return "adr", "name: American Depositary Shares"
    return "common", ""


def sec_ticker(symbol: str) -> str:
    """Alpaca 'BRK.B' -> SEC 'BRK-B'."""
    return symbol.replace(".", "-")


def build(assets: list[dict]) -> pd.DataFrame:
    """Classify Alpaca assets and attach CIK + SEC company info."""
    p = paths().ensure()
    a = pd.DataFrame(assets)
    a = a[a["class"] == "us_equity"].copy()
    a["attributes"] = a["attributes"].apply(lambda x: ",".join(x) if isinstance(x, list) else "")
    a = a.rename(columns={"symbol": "ticker", "id": "alpaca_id"})
    a["sec_ticker"] = a["ticker"].map(sec_ticker)

    tick = sec.load_company_tickers()
    a = a.merge(tick[["ticker", "cik", "sec_exchange"]].rename(columns={"ticker": "sec_ticker"}),
                on="sec_ticker", how="left")
    a["cik"] = a["cik"].astype("Int64")
    info = sec.company_info(a["cik"].dropna().astype(int).tolist())
    a = a.merge(info, on="cik", how="left")

    types, reasons = [], []
    for row in a.itertuples(index=False):
        si = None if pd.isna(row.cik) else {
            "sic": None if pd.isna(row.sic) else int(row.sic),
            "n_periodic_forms": 0 if pd.isna(row.n_periodic_forms) else int(row.n_periodic_forms),
            "n_fund_forms": 0 if pd.isna(row.n_fund_forms) else int(row.n_fund_forms),
        }
        t, why = classify(row.ticker, row.name, row.exchange, row.attributes, si)
        if t in ("common", "adr") and not row.tradable:
            t, why = "not_tradable", "Alpaca tradable=false"
        types.append(t)
        reasons.append(why)
    a["security_type"] = types
    a["drop_reason"] = reasons
    # A foreign filer whose name has no ADR marker may still be an ADR; adr.py checks the filing.
    a["is_foreign_filer"] = a["is_foreign_filer"].fillna(False).astype(bool)
    a["keep"] = a["security_type"].isin(KEEP_TYPES)
    a.loc[a["keep"], "drop_reason"] = ""

    cols = ["ticker", "name", "exchange", "tradable", "attributes", "alpaca_id", "security_type", "keep",
            "drop_reason", "cik", "sec_ticker", "sec_exchange", "sec_name", "entity_type", "sic",
            "sic_description", "owner_org", "category", "state_of_incorporation", "fiscal_year_end",
            "is_foreign_filer", "latest_annual_form", "latest_annual_date", "latest_annual_accn",
            "latest_annual_doc", "n_periodic_forms", "n_fund_forms", "sec_tickers"]
    a = a[cols].sort_values("ticker").reset_index(drop=True)
    a.to_parquet(p.universe / "securities.parquet", index=False)
    dropped = a[~a["keep"]][["ticker", "name", "exchange", "security_type", "drop_reason", "cik"]]
    dropped.to_csv(p.universe / "dropped.csv", index=False)
    log.info("securities: %d assets, kept %d (common %d, adr %d), dropped %d",
             len(a), a["keep"].sum(), (a.security_type == "common").sum(),
             (a.security_type == "adr").sum(), (~a["keep"]).sum())
    return a


def load() -> pd.DataFrame:
    return pd.read_parquet(paths().universe / "securities.parquet")

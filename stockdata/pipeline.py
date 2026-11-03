"""End-to-end update: SEC + Alpaca -> security master, prices, shares, panel, reports.

update() is incremental:
- prices: only missing sessions are downloaded (plus a short overlap to catch
  Alpaca re-adjustments, which trigger a full re-download of that ticker);
- SEC: the bulk files are re-downloaded and re-parsed only when the last SEC
  refresh is older than SEC_REFRESH_HOURS (or forced);
- derived tables (split-basis shares, panel, reports) are always rebuilt from
  local files, because a new split changes the basis of every older count.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import adr, alpaca, panel, prices, sec, sectors, shares, successors, universe, validate
from .config import PRICE_START, paths

log = logging.getLogger(__name__)

SEC_REFRESH_HOURS = 20


def _state() -> dict:
    p = paths().state_file
    return json.loads(p.read_text()) if p.exists() else {}


def _save_state(st: dict) -> None:
    p = paths().state_file
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, default=str))
    tmp.replace(p)


class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.steps: dict[str, float] = {}
        self._last = self.t0

    def lap(self, name: str) -> None:
        now = time.time()
        self.steps[name] = round(now - self._last, 1)
        self._last = now
        log.info("step %-14s %7.1fs", name, self.steps[name])

    @property
    def total(self) -> float:
        return round(time.time() - self.t0, 1)


def build_securities(alp_sess) -> pd.DataFrame:
    """Alpaca assets -> classified security master (needs submissions.zip + ticker map)."""
    p = paths().ensure()
    assets = alpaca.get_assets(alp_sess)
    alpaca.save_json(assets, p.raw_alpaca / "assets.json")
    return universe.build(assets)


def finish_securities(s: pd.DataFrame) -> pd.DataFrame:
    """ADR detection/ratio (reads 20-F covers, cached) and sector; saves the security master."""
    p = paths().ensure()
    s = adr.assign(s, adr.update(s))
    bdc = s["sic"].isna() & (s["n_periodic_forms"].fillna(0) > 0) & (s["n_fund_forms"].fillna(0) == 0)
    s["sector"] = [sectors.sic_to_sector(None if pd.isna(sic) else int(sic), None if pd.isna(c) else int(c), b)
                   for sic, c, b in zip(s["sic"], s["cik"], bdc)]
    s.to_parquet(p.universe / "securities.parquet", index=False)
    s[~s["keep"]][["ticker", "name", "exchange", "security_type", "drop_reason", "cik"]].to_csv(
        p.universe / "dropped.csv", index=False)
    return s


def refresh_securities(alp_sess) -> pd.DataFrame:
    return finish_securities(build_securities(alp_sess))


def _candidates(ciks: list[int], fsds_raw: pd.DataFrame) -> pd.DataFrame:
    t = time.time()
    cf = sec.share_facts_companyfacts(ciks)
    fs = sec.share_facts_fsds(fsds_raw, ciks)
    fv = shares.filing_values(cf, fs)
    log.info("share candidates: %d filings for %d CIKs (%.0fs)", len(fv), len(ciks), time.time() - t)
    return fv


def extract_share_filings(s: pd.DataFrame, fsds_raw: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Expensive SEC step: one chosen share count per filing (as reported, no split basis yet).

    Companies whose CIK changed in a reorganization inherit the predecessor CIK's
    filings from before their own first filing (see successors.py).
    """
    p = paths().ensure()
    ciks = s.loc[s["keep"] & s["cik"].notna(), "cik"].astype(int).unique().tolist()
    fv = _candidates(ciks, fsds_raw)
    trimmed, _ = panel.trim_dead_history(px)
    own = shares.choose(fv)
    late = successors.late_starters(s, own, trimmed)
    prim = panel.primary_tickers(s, trimmed)
    prim = prim[prim["is_primary"]][["cik", "ticker"]]
    late_tickers = prim[prim["cik"].astype(int).isin(set(late))]
    tp = trimmed[trimmed["ticker"].isin(set(late_tickers["ticker"]))]
    by_ticker = {t: g.set_index("date")["close_raw"] for t, g in tp.groupby("ticker")}
    closes = {int(c): by_ticker.get(t) for c, t in zip(late_tickers["cik"], late_tickers["ticker"])}
    listed = set(s.loc[s["keep"] & s["cik"].notna(), "cik"].astype(int))
    links = successors.find_predecessors(late, own, lambda c: _candidates(c, fsds_raw), closes, listed)
    if len(links):
        pred = _candidates(links["predecessor_cik"].astype(int).tolist(), fsds_raw)
        to_succ = dict(zip(links["predecessor_cik"], links["cik"]))
        pred = pred[pred["cik"].isin(to_succ)].copy()
        pred["cik"] = pred["cik"].map(to_succ)
        own_first = fv.groupby("cik")["filed"].min()
        cutoff = pred["cik"].map(own_first)
        pred = pred[cutoff.isna() | (pred["filed"] < cutoff)]
        fv = pd.concat([fv, pred], ignore_index=True).drop_duplicates(["cik", "accn"], keep="first")
        log.info("inherited %d predecessor filings for %d companies", len(pred), pred["cik"].nunique())
    fv.to_parquet(p.sec / "share_candidates.parquet", index=False)  # lets choose() be re-run cheaply
    chosen = shares.choose(fv)
    chosen.to_parquet(p.sec / "share_filings.parquet", index=False)
    chosen[chosen["note"] != ""].to_csv(p.reports / "cover_values_rejected.csv", index=False)
    log.info("share filings: %d filings, %d CIKs", len(chosen), chosen["cik"].nunique())
    return chosen


def derive_share_events(s: pd.DataFrame, px: pd.DataFrame, filings: pd.DataFrame):
    """Cheap step: today's split basis, outlier cleaning, effective (usable-from) dates."""
    p = paths().ensure()
    prim = panel.primary_tickers(s, px)
    ev = shares.to_today_basis(filings, px[["ticker", "date", "split_factor"]], prim[prim["is_primary"]])
    ev, probs = shares.clean_series(ev)
    adr_notes = ev[ev["note"].fillna("").str.contains("ADS ratio change inferred")]
    if len(adr_notes):
        probs = pd.concat([probs, pd.DataFrame({"cik": adr_notes["cik"], "accn": adr_notes["accn"],
                                                "filed": adr_notes["filed"], "issue": adr_notes["note"],
                                                "value": adr_notes["shares_reported"]})], ignore_index=True)
    sessions = pd.DatetimeIndex(sorted(px["date"].unique()))
    future = pd.bdate_range(sessions[-1] + pd.Timedelta(days=1), periods=10)
    ev["effective"] = shares.effective_dates(ev["filed"], sessions.append(future))
    ev.to_parquet(p.sec / "share_events.parquet", index=False)
    probs.to_csv(p.reports / "share_problems.csv", index=False)
    prim.to_parquet(p.universe / "primary_tickers.parquet", index=False)
    return ev, prim


def _sec_due(st: dict) -> bool:
    last = st.get("last_sec_refresh")
    if not last:
        return True
    age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(last)
    return age > dt.timedelta(hours=SEC_REFRESH_HOURS)


def update(full: bool = False, sec_refresh: str = "auto", skip_prices: bool = False) -> dict:
    """sec_refresh: 'auto' (when older than SEC_REFRESH_HOURS), 'always' or 'never'."""
    p = paths().ensure()
    st = _state()
    tm = Timer()
    have_local = (p.universe / "securities.parquet").exists() and (p.sec / "share_filings.parquet").exists()
    do_sec = full or not have_local or sec_refresh == "always" or (sec_refresh == "auto" and _sec_due(st))
    alp_sess = alpaca.session()
    sec_sess = sec.session() if do_sec else None

    if do_sec:
        # Small SEC files first: the security list is all the price download needs, so prices
        # (Alpaca) and the rest of the SEC work (other host, own rate limit) then run in parallel.
        sec.download_bulk(sec_sess, names=("submissions",))
        tm.lap("sec_submissions")
        s = build_securities(alp_sess)
        tm.lap("securities")
    else:
        s = universe.load()
    tickers = s.loc[s["keep"], "ticker"].tolist()

    price_meta = st.get("prices", {})
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = None if skip_prices else pool.submit(prices.update, tickers, full)
        if do_sec:
            s = finish_securities(s)
            tm.lap("adr_sector")
            sec.download_bulk(sec_sess, names=("companyfacts",), tickers=False)
            tm.lap("sec_companyfacts")
            fsds_raw = sec.update_fsds(sec_sess)
            tm.lap("fsds")
        if fut is not None:
            price_meta = fut.result()
            tm.lap("prices_wait")
    px = prices.load_store()
    if do_sec:
        filings = extract_share_filings(s, fsds_raw, px)
        tm.lap("share_filings")
    else:
        filings = pd.read_parquet(p.sec / "share_filings.parquet")

    ev, prim = derive_share_events(s, px, filings)
    tm.lap("share_events")
    pnl = panel.build(s, px, ev, prim)
    tm.lap("panel")
    report = validate.run(s, px, ev, pnl)
    tm.lap("validate")

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    tm.steps["prices_download"] = price_meta.get("seconds") if not skip_prices else None
    st.update({"updated_at": now, "full_rebuild": full, "prices": price_meta,
               "last_session": str(px["date"].max().date()), "price_start": PRICE_START,
               "timings_s": tm.steps, "total_s": tm.total, "coverage": report.get("coverage", {})})
    if do_sec:
        st["last_sec_refresh"] = now
    _save_state(st)
    log.info("update done in %.0fs", tm.total)
    return st


def rebuild_derived(rechoose: bool = True) -> dict:
    """Re-run share choice, share events, panel and validation from local files (no downloads)."""
    p = paths()
    s = universe.load()
    px = prices.load_store()
    cand = p.sec / "share_candidates.parquet"
    if rechoose and cand.exists():
        filings = shares.choose(pd.read_parquet(cand))
        filings.to_parquet(p.sec / "share_filings.parquet", index=False)
        filings[filings["note"] != ""].to_csv(p.reports / "cover_values_rejected.csv", index=False)
    else:
        filings = pd.read_parquet(p.sec / "share_filings.parquet")
    ev, prim = derive_share_events(s, px, filings)
    pnl = panel.build(s, px, ev, prim)
    return validate.run(s, px, ev, pnl)

"""Point-in-time shares outstanding per company (CIK), keyed by filing date.

For every filing (accession number) one company-total share count is chosen:

  1. dei:EntityCommonStockSharesOutstanding (cover page, as of the cover date)
  2. us-gaap:CommonStockSharesOutstanding / ifrs NumberOfSharesOutstanding
     (balance sheet date)
  3. weighted-average basic shares for the latest period (us-gaap / ifrs)
  4. weighted-average diluted shares

Values come from companyfacts (untagged = company total). When a company only
reports per share class (Visa, Berkshire, Meta...), the class rows from the
Financial Statement Data Sets are combined into a total.

The cover value wins unless two other counts from the same filing agree with
each other and the cover is more than 3x below them (BABA's 2026 20-F cover said
1.86bn; balance sheet and weighted average said 18.6bn) or about 1,000x above
(units error). A cover far above period-end counts is usually real (shares
issued in a merger or SPAC closing after quarter end) and is kept. Balance-sheet counts
are often issued shares incl. treasury, and counts tagged in thousands are
rescaled, so a single disagreeing reference never overrides the cover.

Split basis: a cover count is as of its cover date; balance-sheet and weighted
counts are restated for splits up to the filing date (US GAAP restates share
data retroactively for splits before issuance). Each value is converted to
today's split basis with the primary ticker's split factor at that basis date.

A filing is usable from the first trading day AFTER its filing date.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from .config import FILING_LAG_TRADING_DAYS

log = logging.getLogger(__name__)

KIND = {
    "EntityCommonStockSharesOutstanding": "eso",
    "CommonStockSharesOutstanding": "cso",
    "NumberOfSharesOutstanding": "cso",
    "WeightedAverageNumberOfSharesOutstandingBasic": "wab",
    "WeightedAverageShares": "wab",
    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted": "wab",
    "WeightedAverageNumberOfDilutedSharesOutstanding": "wad",
}
INSTANT = {"eso", "cso"}
MIN_SHARES = 10_000
# Companies whose only company-total count is in another class's units:
# cik -> (below this, the count is in the other unit, multiply by, note)
CLASS_UNITS = {
    1067983: (1e7, 1500.0, "Berkshire Class A equivalents converted to Class B (1 A = 1,500 B)"),
}
# Clean factors for ADS ratio changes (home-market splits with an unchanged ADS)
_NICE = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100], dtype=float)

_SEG_EQUITY = re.compile(r"^EquityComponents=CommonStock[A-Za-z]*;$")
_SEG_CLASS = re.compile(r"^ClassOfStock=([A-Za-z0-9]+);$")


def _latest_period(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each (cik, accn, kind)'s current-period rows: latest end, and for
    durations the shortest period ending then (the quarter, not year-to-date)."""
    df = df.copy()
    df["dur"] = (df["end"] - df["start"]).dt.days.fillna(0)
    key = ["cik", "accn", "kind"]
    df = df[df["end"] == df.groupby(key)["end"].transform("max")]
    df = df[df["dur"] == df.groupby(key)["dur"].transform("min")]
    return df


def _segment_total(rows: pd.DataFrame) -> float | None:
    """Company total from per-class rows of one fact (FSDS segments)."""
    seg = rows["segments"].fillna("")
    eq = rows[seg.str.match(_SEG_EQUITY)]
    if len(eq):
        return float(eq["value"].max())
    m = seg.str.extract(_SEG_CLASS)[0]
    cls = rows.assign(member=m).dropna(subset=["member"])
    cls = cls[~cls["member"].str.contains("Preferred|Treasury", regex=True)]
    if not len(cls):
        return None
    members = cls["member"]
    if members.str.startswith("Equivalent").all():
        return float(cls["value"].max())  # the same total in different units (Berkshire A/B)
    if (members == "CommonStock").any():
        return float(cls.loc[members == "CommonStock", "value"].max())
    cls = cls[~members.str.contains("And")]  # drop aggregate members like CommonClassB1AndB2
    return float(cls["value"].sum()) if len(cls) else None


def _fsds_totals(f: pd.DataFrame) -> pd.DataFrame:
    """Vectorised _segment_total for every (cik, accn, kind) of the FSDS rows.

    Priority: untagged value > EquityComponents=CommonStock* > Equivalent* (max)
    > CommonStock member > sum of class members (aggregates like B1AndB2 dropped).
    """
    key = ["cik", "accn", "kind"]
    seg = f["segments"]
    meta = f.groupby(key).agg(end=("end", "max"), form=("form", "first"), filed=("filed", "min"))
    nondim = f[seg.isna()].groupby(key)["value"].max()
    s = seg.fillna("")
    equity = f[s.str.match(_SEG_EQUITY)].groupby(key)["value"].max()
    cl = f.assign(member=s.str.extract(_SEG_CLASS)[0]).dropna(subset=["member"])
    cl = cl[~cl["member"].str.contains("Preferred|Treasury", regex=True)]
    cl = cl.assign(is_equiv=cl["member"].str.startswith("Equivalent").astype(int))
    g = cl.groupby(key)
    all_equiv = g["is_equiv"].min().astype(bool)
    equiv_max = g["value"].max()
    common = cl[cl["member"] == "CommonStock"].groupby(key)["value"].max()
    summed = cl[~cl["member"].str.contains("And")].groupby(key)["value"].sum()
    out = meta.copy()
    out["val"] = nondim
    out["origin"] = np.where(out["val"].notna(), "fsds", "fsds_classes")
    for cand in (equity, equiv_max.where(all_equiv), common, summed):
        out["val"] = out["val"].fillna(cand)
    return out.dropna(subset=["val"]).reset_index()


def filing_values(cf: pd.DataFrame, fsds: pd.DataFrame) -> pd.DataFrame:
    """One row per (cik, accn) with eso/cso/wab/wad values and their dates."""
    parts = []
    if len(cf):
        c = cf.assign(kind=cf["concept"].map(KIND)).dropna(subset=["kind"])
        c = _latest_period(c)
        c = c.groupby(["cik", "accn", "kind"], as_index=False).agg(
            val=("val", "max"), end=("end", "max"), form=("form", "last"), filed=("filed", "min"))
        c["origin"] = "companyfacts"
        parts.append(c)
    if len(fsds):
        f = fsds.assign(kind=fsds["concept"].map(KIND)).dropna(subset=["kind"]).rename(columns={"val": "value"})
        f = _latest_period(f)
        fs = _fsds_totals(f)
        if len(fs):
            parts.append(fs)
    if not parts:
        return pd.DataFrame()
    allv = pd.concat(parts, ignore_index=True)
    # companyfacts wins over FSDS for the same filing and kind.
    allv["prio"] = allv["origin"].map({"companyfacts": 0, "fsds": 1, "fsds_classes": 2})
    allv = allv.sort_values("prio").drop_duplicates(["cik", "accn", "kind"], keep="first")
    wide = allv.pivot_table(index=["cik", "accn"], columns="kind", values=["val", "end"], aggfunc="first")
    wide.columns = [f"{k}_{v}" if v == "end" else k for v, k in wide.columns]
    org = allv.pivot_table(index=["cik", "accn"], columns="kind", values="origin", aggfunc="first")
    org.columns = [f"{k}_origin" for k in org.columns]
    meta = allv.groupby(["cik", "accn"]).agg(form=("form", "first"), filed=("filed", "min"))
    out = wide.join(org).join(meta).reset_index()
    for k in ("eso", "cso", "wab", "wad"):
        for col in (k, f"{k}_end", f"{k}_origin"):
            if col not in out:
                out[col] = np.nan if not col.endswith("_end") else pd.NaT
    return out


def _agree(a: float, b: float, tol: float) -> bool:
    return a > 0 and b > 0 and max(a, b) / min(a, b) <= tol


MAX_PLAUSIBLE_SHARES = 1e11  # above any real share count (largest US-listed: ~60bn ordinary shares)


def _rescale(ref: float, target: float) -> float:
    """A reference about 1,000x or 1,000,000x below the target was tagged in thousands/millions,
    unless the rescaled count would be implausible (then the target is the wrong one)."""
    r = target / ref
    for k in (1e3, 1e6):
        if 0.5 * k <= r <= 2 * k and ref * k <= MAX_PLAUSIBLE_SHARES:
            return ref * k
    return ref


def choose(fv: pd.DataFrame) -> pd.DataFrame:
    """Pick one share count per filing (see module docstring)."""
    rows = []
    for r in fv.itertuples(index=False):
        cands = {k: getattr(r, k) for k in ("eso", "cso", "wab", "wad")}
        cands = {k: float(v) for k, v in cands.items() if pd.notna(v) and v >= MIN_SHARES}
        if not cands:
            continue
        note = ""
        if "eso" in cands:
            eso = cands["eso"]
            # References tagged in thousands or millions are rescaled before comparing.
            refs = {k: _rescale(cands[k], eso) for k in ("cso", "wab", "wad") if k in cands}
            kind = "eso"
            if refs and not any(_agree(eso, v, 1.5) for v in refs.values()):
                # Trust the references over the cover only if two of them agree with each
                # other and all differ from the cover by more than 3x (BABA 2026: 1.86bn vs 18.6bn).
                # A cover far ABOVE the period-end counts is often real (stock issued in a merger or
                # SPAC closing after quarter end), so only a cover far below them, or one ~1,000x /
                # ~1,000,000x above (units error), is replaced.
                ks = sorted(refs)
                pairs = [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:] if _agree(refs[a], refs[b], 1.15)]
                if pairs:
                    k = "wab" if any("wab" in p for p in pairs) else pairs[0][0]
                    x = eso / refs[k]
                    units_error = 500 <= x <= 2000 or 5e5 <= x <= 2e6
                    if x < 1 / 3 or units_error:
                        kind = k
                        cands[kind] = refs[kind]
                        note = f"cover value {eso:.0f} rejected (x{x:.3g} vs {kind}, confirmed by {pairs[0]})"
        else:
            kind = next(k for k in ("cso", "wab", "wad") if k in cands)
            # Balance-sheet counts are sometimes issued shares (incl. treasury); prefer the
            # weighted average when the two disagree.
            if kind == "cso" and "wab" in cands and not _agree(cands["cso"], cands["wab"], 1.15):
                kind = "wab"
                note = f"balance-sheet value {cands['cso']:.0f} not used (x{cands['cso'] / cands['wab']:.3g} vs wab)"
        if cands[kind] != getattr(r, kind):
            note = (note + "; " if note else "") + "rescaled from thousands/millions"
        v = cands[kind]
        if kind in INSTANT and abs(v - round(v)) > 1e-6 and v < 1e7:
            # A point-in-time share count is a whole number; a fractional one under 10M
            # was tagged in thousands (QXO's cover: 673,556.656 shares).
            cands[kind] = v * 1000.0
            note = (note + "; " if note else "") + "fractional count read as thousands"
        unit = CLASS_UNITS.get(int(r.cik))
        if unit and cands[kind] < unit[0]:
            cands[kind] *= unit[1]
            note = (note + "; " if note else "") + unit[2]
        end = getattr(r, f"{kind}_end")
        if pd.notna(end) and end > r.filed:  # mis-tagged cover date after the filing itself
            end = r.filed
            note = (note + "; " if note else "") + "as-of date after filing date, clamped"
        rows.append({
            "cik": int(r.cik), "accn": r.accn, "form": r.form, "filed": r.filed,
            "shares_reported": float(cands[kind]), "kind": kind, "origin": getattr(r, f"{kind}_origin"),
            "as_of": end,
            # date whose split basis the number is on (see module docstring)
            "basis_date": end if kind == "eso" else r.filed,
            "note": note,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["cik", "filed", "as_of"]).reset_index(drop=True)


def effective_dates(filed: pd.Series, sessions: pd.DatetimeIndex, lag: int = FILING_LAG_TRADING_DAYS) -> pd.Series:
    """First trading session strictly after the filing date (lag=1), or `lag` sessions after."""
    sessions = pd.DatetimeIndex(sessions).as_unit("ns")
    f = filed.to_numpy(dtype="datetime64[ns]")
    pos = sessions.searchsorted(f, side="right")  # first session > filed
    pos = pos + (lag - 1) if lag >= 1 else sessions.searchsorted(f, side="left")
    out = pd.Series(pd.NaT, index=filed.index, dtype="datetime64[ns]")
    ok = pos < len(sessions)
    out[ok] = sessions[pos[ok]]
    return out


def to_today_basis(events: pd.DataFrame, split_factors: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    """Add shares_adj: shares on today's split basis (and in ADS units for ADRs).

    split_factors: ticker, date, split_factor (from the price store)
    primary: cik, ticker, security_type, ads_ratio
    """
    ev = events.merge(primary[["cik", "ticker", "security_type", "ads_ratio"]], on="cik", how="inner")
    sf = split_factors.sort_values(["date"])
    ev = ev.assign(filed=ev["filed"].astype("datetime64[ns]"), as_of=ev["as_of"].astype("datetime64[ns]"))
    ev = ev.sort_values("basis_date")
    ev["basis_date"] = ev["basis_date"].astype("datetime64[ns]")
    sf = sf.assign(date=sf["date"].astype("datetime64[ns]"))
    m = pd.merge_asof(ev, sf[["ticker", "date", "split_factor"]], left_on="basis_date", right_on="date",
                      by="ticker", direction="backward")
    # Basis date before the first price: use the first available factor.
    first = sf.groupby("ticker")["split_factor"].first()
    m["split_factor"] = m["split_factor"].fillna(m["ticker"].map(first))
    m = m.drop(columns=["date"]).sort_values(["cik", "filed", "as_of"]).reset_index(drop=True)
    is_adr = m["security_type"].eq("adr")
    m["ads_ratio_used"] = np.nan
    if is_adr.any():
        for cik, idx in m[is_adr].groupby("cik").groups.items():
            g = m.loc[idx]
            ratios, notes = _adr_ratio_history(g, float(g["ads_ratio"].iloc[0]))
            m.loc[idx, "ads_ratio_used"] = ratios
            for i, txt in notes:
                m.loc[i, "note"] = (m.loc[i, "note"] + "; " if m.loc[i, "note"] else "") + txt
    m["shares_adj"] = np.where(
        is_adr,
        m["shares_reported"] / m["ads_ratio_used"],  # ADS units; ADS splits are in the price factor
        m["shares_reported"] * m["split_factor"].fillna(1.0))
    return m


def _adr_ratio_history(g: pd.DataFrame, ratio_now: float) -> tuple[np.ndarray, list]:
    """Ordinary shares per ADS at each filing, walking back from today's ratio.

    A home-market split leaves the ADS untouched and changes the ratio instead
    (BABA 2019: 8-for-1 split, 1 ADS = 1 -> 8 shares), so a jump in reported
    ordinary shares by a clean factor with no ADS split in between means the
    ratio was that much smaller (or larger) before.
    """
    v = g["shares_reported"].to_numpy(dtype="float64")
    sf = g["split_factor"].to_numpy(dtype="float64")
    ratios = np.full(len(v), ratio_now)
    notes = []
    cur = ratio_now
    if not np.isfinite(cur):
        return ratios, notes
    for i in range(len(v) - 2, -1, -1):
        r = v[i + 1] / v[i]
        if np.isfinite(r) and r > 0 and sf[i] == sf[i + 1]:
            k = r if r >= 1 else 1 / r
            j = int(np.argmin(np.abs(np.log(k) - np.log(_NICE))))
            if k >= 1.9 and abs(np.log(k / _NICE[j])) < np.log(1.04):
                cur = cur / _NICE[j] if r > 1 else cur * _NICE[j]
                notes.append((g.index[i + 1], f"ADS ratio change inferred: ordinary shares x{r:.3g}, "
                                              f"ratio before = {cur:g}"))
        ratios[i] = cur
    return ratios, notes


def fix_units_errors(ev: pd.DataFrame, prices: pd.DataFrame, primary: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Fix runs of filings tagged in thousands (or millions) vs units.

    Some filers report several consecutive quarters 1,000x off (CPK and SKY covers
    showed 23.5bn / 57.6bn shares; RY's and DKNG's weighted counts were in thousands).
    The series is split at ~1,000x / ~1,000,000x breaks, and every piece is put on the
    level whose implied daily turnover (dollar volume / market cap) is most plausible:
    the true level trades about 0.01%-5% of the company per day, a 1,000x error does not.
    """
    ev = ev.copy()
    fixes = []
    prim = primary[["cik", "ticker"]].drop_duplicates("cik").set_index("cik")["ticker"]
    # First find the companies with a ~1,000x / ~1,000,000x break, then load only their prices.
    todo = []
    for cik, g in ev.groupby("cik", sort=False):
        g = g.sort_values(["filed", "as_of"])
        v = g["shares_adj"].to_numpy(dtype="float64")
        if len(v) < 2 or not np.isfinite(v).all() or (v <= 0).any():
            continue
        r = np.log10(v[1:] / v[:-1])
        k = np.where(np.abs(r - np.round(r / 3) * 3) < 0.35, np.round(r / 3) * 3, 0)  # 0, +-3, +-6
        if k.any() and cik in prim.index:
            todo.append((cik, g, v, np.concatenate([[0.0], np.cumsum(k)])))
    if not todo:
        return ev, fixes
    tickers = {prim[c] for c, *_ in todo}
    px_by = {t: q.sort_values("date") for t, q in
             prices.loc[prices["ticker"].isin(tickers), ["ticker", "date", "close_raw", "volume",
                                                         "split_factor"]].groupby("ticker")}
    for cik, g, v, level in todo:  # level: each filing's exponent relative to the first
        q = px_by.get(prim[cik])
        if q is None or q.empty:
            continue
        dv = (q["close_raw"] * q["volume"]).to_numpy()
        adj_px = (q["close_raw"] / q["split_factor"]).to_numpy()
        dates = q["date"].to_numpy(dtype="datetime64[ns]")
        eff = g["effective"].to_numpy(dtype="datetime64[ns]")
        turnover = {}
        for L in np.unique(level):
            vals = []
            for i in np.flatnonzero(level == L):
                lo = np.searchsorted(dates, eff[i])
                hi = np.searchsorted(dates, eff[i + 1]) if i + 1 < len(eff) else lo + 60
                hi = min(hi, lo + 60, len(dates))
                if hi > lo:
                    mc = adj_px[lo:hi] * v[i]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        tv = dv[lo:hi] / mc
                    vals.extend(tv[np.isfinite(tv) & (tv > 0)].tolist())
            if vals:
                turnover[L] = float(np.median(vals))
        if not turnover:
            continue
        best = min(turnover, key=lambda L: abs(np.log10(turnover[L] / 3e-3)))

        def plausible(x):
            return 2e-5 <= x <= 0.3

        idx = g.index.to_numpy()
        for i, L in enumerate(level):
            # Only a piece whose OWN turnover is implausible is rescaled, and only if that makes it
            # plausible: a genuine jump (QXO: 5M -> 409M shares in 2024) leaves both pieces alone.
            if L != best and L in turnover and not plausible(turnover[L]) and \
                    plausible(turnover[L] * 10.0 ** (L - best)):
                factor = 10.0 ** (best - L)
                ev.loc[idx[i], ["shares_reported", "shares_adj"]] = ev.loc[idx[i], ["shares_reported", "shares_adj"]] * factor
                note = ev.loc[idx[i], "note"]
                note = note if isinstance(note, str) else ""
                ev.loc[idx[i], "note"] = (note + "; " if note else "") + f"units error fixed (x{factor:g})"
                fixes.append({"cik": cik, "accn": ev.loc[idx[i], "accn"], "filed": ev.loc[idx[i], "filed"],
                              "issue": f"units error fixed: x{factor:g} (turnover check)",
                              "value": ev.loc[idx[i], "shares_reported"]})
    return ev, fixes


def rescue_units(ev: pd.DataFrame, prices: pd.DataFrame, primary: pd.DataFrame,
                 public_float: pd.DataFrame | None) -> tuple[pd.DataFrame, list]:
    """Fix companies whose EVERY filing is in thousands (QXO's cover reads 673,556.656 shares).

    With no break in the series, the signal is an absurd median daily turnover
    (dollar volume / market cap, normally 0.01%-5%). The whole series is rescaled by
    1,000 or 1,000,000 only if that makes turnover normal AND the company's own reported
    public float (USD) agrees with the rescaled market cap; without float data only
    extreme cases (turnover > 300%/day) are rescaled. A genuinely tiny company with a
    hot stock has a small public float, so it is never inflated.
    """
    ev = ev.copy()
    fixes = []
    prim = primary[["cik", "ticker"]].drop_duplicates("cik")
    q = prices.merge(prim, on="ticker", how="inner")[["cik", "date", "close_raw", "volume", "split_factor"]]
    q = q.assign(cik=q["cik"].astype("int64"), date=q["date"].astype("datetime64[ns]")).sort_values("date")
    e = ev[["cik", "effective", "shares_adj"]].dropna().assign(
        cik=lambda d: d["cik"].astype("int64"), effective=lambda d: d["effective"].astype("datetime64[ns]"))
    m = pd.merge_asof(q, e.sort_values("effective"), left_on="date", right_on="effective", by="cik")
    m = m[(m["volume"] > 0) & m["shares_adj"].notna()]
    tv = (m["close_raw"] * m["volume"]) / (m["close_raw"] / m["split_factor"] * m["shares_adj"])
    turnover = tv.groupby(m["cik"]).median()
    pf = public_float if public_float is not None else pd.DataFrame(columns=["cik", "end", "val"])
    for cik, T in turnover.items():
        if 1e-6 <= T <= 0.3:
            continue
        for k in (1e3, 1e6):
            k = k if T > 0.3 else 1 / k
            T2 = T / k
            if not (3e-5 <= T2 <= 0.05):
                continue
            g = m[m["cik"] == cik]
            votes = []
            for r in pf[pf["cik"] == cik].sort_values("end").tail(3).itertuples(index=False):
                day = g[g["date"] <= pd.Timestamp(r.end)].tail(1)
                if len(day) and (pd.Timestamp(r.end) - day["date"].iloc[0]).days <= 10:
                    mc = float(day["close_raw"].iloc[0] / day["split_factor"].iloc[0] * day["shares_adj"].iloc[0] * k)
                    votes.append(0.15 <= r.val / mc <= 1.6)
            ok = (sum(votes) > len(votes) / 2) if votes else (T > 3 or T < 1e-7)
            if ok:
                sel = ev["cik"] == cik
                ev.loc[sel, ["shares_reported", "shares_adj"]] = ev.loc[sel, ["shares_reported", "shares_adj"]] * k
                ev.loc[sel, "note"] = ev.loc[sel, "note"].fillna("").astype(str) + f"; whole series x{k:g} (turnover)"
                fixes.append({"cik": cik, "accn": "", "filed": ev.loc[sel, "filed"].max(),
                              "issue": f"units error fixed: whole series x{k:g} (turnover {T:.3g}/day -> {T2:.3g}, "
                                       f"public float {'agrees' if votes else 'not available'})",
                              "value": float(ev.loc[sel, "shares_reported"].iloc[-1])})
            break
    return ev, fixes


def clean_series(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop going-back-in-time amendments and isolated outliers. Returns (kept, problems)."""
    kept, probs = [], []
    for cik, g in ev.groupby("cik", sort=False):
        g = g.sort_values(["filed", "as_of"]).copy()
        # A later filing whose count is older than one already known adds nothing.
        g = g[g["as_of"] >= g["as_of"].cummax()]
        g = g.drop_duplicates(["filed"], keep="last")
        v = g["shares_adj"].to_numpy(dtype="float64")
        bad = np.zeros(len(v), dtype=bool)
        for i in range(1, len(v) - 1):
            a, b, c = v[i - 1], v[i], v[i + 1]
            if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
                continue
            jump = b / a
            revert = c / a
            if (jump > 2.5 or jump < 0.4) and 0.8 < revert < 1.25:
                bad[i] = True
        if bad.any():
            for _, r in g[bad].iterrows():
                probs.append({"cik": cik, "accn": r["accn"], "filed": r["filed"], "issue": "isolated outlier dropped",
                              "value": r["shares_reported"]})
        g = g[~bad]
        # Persistent big moves are kept but reported (mergers, spin-offs, or data errors).
        v = g["shares_adj"].to_numpy(dtype="float64")
        if len(v) > 1:
            r_ = v[1:] / v[:-1]
            for j in np.flatnonzero((r_ > 2.5) | (r_ < 0.4)):
                probs.append({"cik": cik, "accn": g["accn"].iloc[j + 1], "filed": g["filed"].iloc[j + 1],
                              "issue": f"share count changed x{r_[j]:.2f} vs previous filing",
                              "value": g["shares_reported"].iloc[j + 1]})
        kept.append(g)
    out = pd.concat(kept, ignore_index=True) if kept else ev.iloc[:0]
    return out, pd.DataFrame(probs)

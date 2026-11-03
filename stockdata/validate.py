"""Checks and coverage report -> data/reports/validation.md, problems.csv."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from .config import MIN_ADV, MIN_PRICE, paths

log = logging.getLogger(__name__)


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Minimal markdown table (avoids a tabulate dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and v != v) else str(v) for v in row) + " |")
    return "\n".join(lines)

# (ticker, ex-date, split ratio new/old)
KNOWN_SPLITS = [
    ("AAPL", "2020-08-31", 4.0),
    ("TSLA", "2020-08-31", 5.0),
    ("TSLA", "2022-08-25", 3.0),
    ("NVDA", "2021-07-20", 4.0),
    ("NVDA", "2024-06-10", 10.0),
]

# Rough public year-end market caps (USD bn) to sanity-check against; +-20% passes.
KNOWN_MCAPS = [
    ("AAPL", "2020-12-31", 2255), ("MSFT", "2021-12-31", 2525), ("NVDA", "2023-12-29", 1223),
    ("TSLA", "2021-12-31", 1061), ("AMZN", "2020-12-31", 1634), ("GOOGL", "2021-12-31", 1920),
    ("META", "2022-12-30", 316), ("BRK.B", "2022-12-30", 676), ("JPM", "2019-12-31", 429),
    ("XOM", "2016-12-30", 374), ("V", "2021-12-31", 460), ("JNJ", "2018-12-31", 346),
    ("WMT", "2023-12-29", 424), ("KO", "2017-12-29", 196),
]


def split_checks(pnl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tic, ex, ratio in KNOWN_SPLITS:
        g = pnl[pnl["ticker"] == tic].set_index("date").sort_index()
        exd = pd.Timestamp(ex)
        if exd not in g.index:
            rows.append({"ticker": tic, "ex_date": ex, "ok": False, "detail": "ex-date not in data"})
            continue
        i = g.index.get_loc(exd)
        prev, cur = g.iloc[i - 1], g.iloc[i]
        sf_ratio = prev["split_factor"] / cur["split_factor"]
        raw_ratio = prev["close_raw"] / cur["close_raw"]
        adj_ret = cur["close"] / prev["close"] - 1
        mcap_ret = cur["mcap"] / prev["mcap"] - 1 if pd.notna(prev["mcap"]) and pd.notna(cur["mcap"]) else np.nan
        sh_ratio = cur["shares"] / prev["shares"] if pd.notna(prev["shares"]) else np.nan
        ok = (abs(sf_ratio - ratio) < 1e-6 and abs(adj_ret) < 0.15 and abs(raw_ratio / ratio - 1) < 0.15
              and abs(mcap_ret) < 0.15)
        rows.append({"ticker": tic, "ex_date": ex, "expected_ratio": ratio, "split_factor_ratio": round(sf_ratio, 4),
                     "raw_close_ratio": round(raw_ratio, 3), "adj_return": round(adj_ret, 4),
                     "mcap_change": round(mcap_ret, 4), "shares_ratio": round(sh_ratio, 3), "ok": bool(ok)})
    return pd.DataFrame(rows)


def mcap_checks(pnl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tic, d, bn in KNOWN_MCAPS:
        r = pnl[(pnl["ticker"] == tic) & (pnl["date"] == pd.Timestamp(d))]
        got = float(r["mcap"].iloc[0]) / 1e9 if len(r) and pd.notna(r["mcap"].iloc[0]) else np.nan
        rows.append({"ticker": tic, "date": d, "approx_known_bn": bn, "panel_bn": round(got, 1),
                     "ratio": round(got / bn, 3) if got == got else np.nan,
                     "ok": bool(got == got and abs(got / bn - 1) < 0.2)})
    return pd.DataFrame(rows)


def problems(s: pd.DataFrame, px: pd.DataFrame, ev: pd.DataFrame, pnl: pd.DataFrame) -> pd.DataFrame:
    """Tickers that could plausibly be in the universe but have a data gap."""
    out = []
    p = paths()
    elig = (pnl["close_raw"] >= MIN_PRICE) & (pnl["adv50"] >= MIN_ADV)
    e = pnl.loc[elig, ["ticker", "mcap"]]
    liquid = sorted(e["ticker"].unique())
    miss = e["mcap"].isna().groupby(e["ticker"]).mean()
    first_ok = pnl.loc[pnl["mcap"].notna()].groupby("ticker")["date"].min()
    issue = pnl.loc[pnl["data_issue"].notna()].groupby("ticker")["data_issue"].first()
    keep = s[s["keep"]].set_index("ticker")
    for t in liquid:
        if t not in keep.index:
            continue
        r = keep.loc[t]
        if pd.isna(r["cik"]):
            out.append((t, "no_sec_cik", "ticker not in SEC company_tickers_exchange.json (bank filing with FDIC/OCC, "
                                         "or new listing); no shares, no market cap"))
            continue
        adr_no_ratio = r["security_type"] == "adr" and pd.isna(r.get("ads_ratio"))
        if adr_no_ratio:
            out.append((t, "adr_ratio_unknown", "ADS ratio not found in the 20-F; market cap left empty"))
        m = miss.get(t, 0.0)
        if m > 0.02 and not adr_no_ratio:
            f = first_ok.get(t)
            out.append((t, "mcap_missing", f"market cap missing on {m:.0%} of liquid days"
                        f" (first value {f.date() if f is not None and pd.notna(f) else 'never'})"))
        if pd.isna(r["sector"]):
            out.append((t, "sector_missing", f"no sector for SIC {r['sic']}"))
        if t in issue.index:
            out.append((t, "price_jump", issue[t]))
    sp = p.reports / "share_problems.csv"
    if sp.exists() and sp.stat().st_size > 1:
        try:
            spdf = pd.read_csv(sp)
        except pd.errors.EmptyDataError:
            spdf = pd.DataFrame()
        if len(spdf):
            prim = pnl[pnl["is_primary"]].drop_duplicates("cik")[["cik", "ticker"]]
            spdf = spdf.merge(prim, on="cik", how="inner")
            spdf = spdf[spdf["ticker"].isin(set(liquid)) & (pd.to_datetime(spdf["filed"]) >= "2015-07-01")]
            for r in spdf.itertuples(index=False):
                out.append((r.ticker, "share_count", f"{r.issue} (filed {str(r.filed)[:10]}, accn {r.accn})"))
    return pd.DataFrame(out, columns=["ticker", "issue", "detail"]).drop_duplicates()


def universe_by_year(pnl: pd.DataFrame) -> pd.DataFrame:
    u = pnl[pnl["in_universe"]]
    daily = u.groupby("date")["ticker"].nunique()
    yr = daily.groupby(daily.index.year)
    out = pd.DataFrame({"avg_daily": yr.mean().round(0), "min_daily": yr.min(), "max_daily": yr.max(),
                        "year_end": yr.last(), "distinct_tickers": u.groupby(u["date"].dt.year)["ticker"].nunique()})
    sectors = u[u["date"] == u["date"].max()].groupby("sector", observed=True)["ticker"].nunique()
    return out, sectors.sort_values(ascending=False)


def run(s: pd.DataFrame, px: pd.DataFrame, ev: pd.DataFrame, pnl: pd.DataFrame) -> dict:
    p = paths().ensure()
    sc = split_checks(pnl)
    mc = mcap_checks(pnl)
    pr = problems(s, px, ev, pnl)
    uby, sect_now = universe_by_year(pnl)
    keep = s[s["keep"]]
    has_px = set(px["ticker"].unique())
    tick_mcap = set(pnl.loc[pnl["mcap"].notna(), "ticker"].unique())
    cov = {
        "securities_kept": int(len(keep)),
        "common": int((keep["security_type"] == "common").sum()),
        "adr": int((keep["security_type"] == "adr").sum()),
        "with_prices": int(keep["ticker"].isin(has_px).sum()),
        "with_cik": int(keep["cik"].notna().sum()),
        "with_shares_ever": int(len(tick_mcap)),
        "with_sector": int(keep["sector"].notna().sum()),
        "common_with_prices": int(((keep["security_type"] == "common") & keep["ticker"].isin(has_px)).sum()),
        "common_with_shares_ever": int(((keep["security_type"] == "common") & keep["ticker"].isin(tick_mcap)).sum()),
        "common_with_sector": int(((keep["security_type"] == "common") & keep["sector"].notna()).sum()),
        "dropped": int((~s["keep"]).sum()),
        "dropped_by_type": s.loc[~s["keep"], "security_type"].value_counts().to_dict(),
        "panel_rows": int(len(pnl)),
        "panel_first_date": str(pnl["date"].min().date()),
        "panel_last_date": str(pnl["date"].max().date()),
        "universe_today": int(pnl.loc[pnl["date"] == pnl["date"].max(), "in_universe"].sum()),
        "split_checks_ok": int(sc["ok"].sum()), "split_checks": int(len(sc)),
        "mcap_checks_ok": int(mc["ok"].sum()), "mcap_checks": int(len(mc)),
        "problem_tickers": int(pr["ticker"].nunique()),
    }
    pr.to_csv(p.reports / "problems.csv", index=False)
    (p.reports / "coverage.json").write_text(json.dumps(cov, indent=2))
    md = ["# Validation report", "", f"Panel {cov['panel_first_date']} .. {cov['panel_last_date']}, "
          f"{cov['panel_rows']:,} rows.", "", "## Coverage", "",
          "| measure | count |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in cov.items() if not isinstance(v, dict)]
    md += ["", "Dropped by type: " + ", ".join(f"{k} {v}" for k, v in cov["dropped_by_type"].items()), ""]
    md += ["## Split spot-checks", "", md_table(sc), "",
           "## Market cap sanity checks (approximate public figures, +-20%)", "", md_table(mc), "",
           "## $3bn+ universe size by year (in_universe = True)", "", md_table(uby, index=True), "",
           "## Universe by sector on the last date", "", md_table(sect_now.to_frame("tickers"), index=True), "",
           f"## Problem tickers ({cov['problem_tickers']}) - liquid names with a data gap", "",
           "Full list: problems.csv. Counts by issue:", "",
           md_table(pr["issue"].value_counts().to_frame("tickers"), index=True), ""]
    (p.reports / "validation.md").write_text("\n".join(md))
    sc.to_csv(p.reports / "split_checks.csv", index=False)
    mc.to_csv(p.reports / "mcap_checks.csv", index=False)
    uby.to_csv(p.reports / "universe_by_year.csv")
    log.info("validation: %s", {k: v for k, v in cov.items() if not isinstance(v, dict)})
    return {"coverage": cov, "split_checks": sc, "mcap_checks": mc, "problems": pr, "universe_by_year": uby}

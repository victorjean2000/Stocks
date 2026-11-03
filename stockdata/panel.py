"""Daily panel: prices + point-in-time shares, market cap, sector, universe flag.

Use load_panel() to read it; build() is called by the pipeline.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import (ADV_WINDOW, MAX_SHARES_STALENESS_DAYS, MIN_ADV, MIN_MARKET_CAP, MIN_PRICE,
                     UNIVERSE_SECURITY_TYPES, paths)

log = logging.getLogger(__name__)

PANEL_COLUMNS = [
    "date", "ticker", "open", "high", "low", "close", "close_raw", "volume", "dollar_volume", "adv50",
    "split_factor", "shares", "shares_asof", "shares_filed", "shares_source", "mcap", "sector", "cik",
    "security_type", "is_primary", "data_issue", "in_universe",
]
JUMP_EXCLUSION_SESSIONS = 260  # a 12-month lookback spanning a bad jump is unusable
DEAD_RUN_SESSIONS = 20         # this many sessions without trading = a different security before


def trim_dead_history(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop each ticker's bars before its last stretch of >= 20 sessions with no trading.

    Alpaca returns everything ever traded under a symbol string, so a reused ticker
    (today's POM vs Pepco Holdings in 2016) or a successor (Dow Inc. vs Dow Chemical,
    carried forward with zero volume from 2017-09 to 2019-04) brings another company's
    prices. A long run of missing or zero-volume sessions marks the boundary.
    """
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    keep_from = {}
    rows = []
    for tic, g in prices.groupby("ticker", sort=False):
        pos = cal.get_indexer(g["date"])
        lo = pos.min()
        alive = np.zeros(pos.max() - lo + 1, dtype=bool)
        alive[pos[g["volume"].to_numpy() > 0] - lo] = True
        edges = np.flatnonzero(np.diff(np.r_[0, (~alive).astype(np.int8), 0])).reshape(-1, 2)
        runs = [(a, b) for a, b in edges if b - a >= DEAD_RUN_SESSIONS and b < len(alive)]
        if runs:
            a, b = runs[-1]
            keep_from[tic] = cal[lo + b]
            rows.append({"ticker": tic, "first_bar": cal[lo], "kept_from": cal[lo + b], "dead_sessions": int(b - a)})
    trims = pd.DataFrame(rows, columns=["ticker", "first_bar", "kept_from", "dead_sessions"])
    if not keep_from:
        return prices, trims
    start = prices["ticker"].map(keep_from)
    return prices[start.isna() | (prices["date"] >= start)], trims


def panel_path():
    return paths().panel / "panel.parquet"


def primary_tickers(securities: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """One ticker per CIK: the share class with the highest median dollar volume."""
    s = securities[securities["keep"] & securities["cik"].notna()][["ticker", "cik", "security_type", "ads_ratio"]]
    dv = (prices["close_raw"] * prices["volume"]).groupby(prices["ticker"]).median().rename("med_dv")
    s = s.merge(dv, left_on="ticker", right_index=True, how="inner")
    s = s.sort_values(["cik", "med_dv"], ascending=[True, False])
    s["is_primary"] = ~s.duplicated("cik")
    s["n_classes"] = s.groupby("cik")["ticker"].transform("size")
    s["cik"] = s["cik"].astype("int64")
    return s


def detect_unadjusted_jumps(prices: pd.DataFrame) -> pd.DataFrame:
    """One-day moves that look like a missed split / ADS ratio change.

    Signature: raw close ratio within 3% of a clean fraction (x2, x0.5, x3, x10 ...),
    no split recorded that day, median volume shifting by the inverse ratio (within
    35%), and no event-day spike (share volume < 2.5x the following median, dollar
    volume < 2x the preceding median). Tuned on 2016-2026: of 188 raw candidates
    this keeps AZN 2026-02-02 (ADS -> ordinary), IX 2025-02-28 (ADS ratio), SLGN
    2017-05-30 (2:1 split not applied), ZWS 2021-10-05 (Rexnord/Regal deal) and
    one sub-$10 crash day; real crashes are excluded so losers are not dropped.
    """
    p = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    tick = p["ticker"].to_numpy()
    close = p["close_raw"].to_numpy(dtype="float64")
    sf = p["split_factor"].to_numpy(dtype="float64")
    vol = p["volume"].to_numpy(dtype="float64")
    same = np.r_[False, tick[1:] == tick[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.r_[np.nan, close[1:] / close[:-1]]
    sf_same = np.r_[False, sf[1:] == sf[:-1]]
    nice = np.array([2, 3, 4, 5, 8, 10, 20, 25, 50, 100, 1.5, 2.5], dtype=float)
    nice = np.log(np.concatenate([nice, 1 / nice]))
    cand = np.flatnonzero(same & sf_same & ((ratio > 1.7) | (ratio < 1 / 1.7)))
    starts = np.r_[0, np.flatnonzero(~np.r_[True, tick[1:] == tick[:-1]][1:]) + 1]
    first_of = np.zeros(len(p), dtype=np.int64)
    first_of[starts] = starts
    first_of = np.maximum.accumulate(first_of)
    last_of = np.r_[starts[1:], len(p)]
    last_idx = np.repeat(last_of, np.diff(np.r_[starts, len(p)]))
    out = []
    for i in cand:
        lr = np.log(ratio[i])
        if np.min(np.abs(lr - nice)) > np.log(1.03):
            continue
        before = vol[max(first_of[i], i - 20):i]
        after = vol[i + 1:min(last_idx[i], i + 21)]
        if len(before) < 5 or len(after) < 5:
            continue
        b, a = np.median(before), np.median(after)
        if not (b > 0 and a > 0):
            continue
        # A missed split shifts the volume LEVEL inversely to the price, without the
        # event-day spike (in shares and in dollars) that a real crash or rally has.
        level_shift_ok = abs(np.log(a / b) + lr) < np.log(1.35)
        no_spike = vol[i] / a < 2.5 and (vol[i] * close[i]) / np.median(before * close[max(first_of[i], i - 20):i]) < 2.0
        if level_shift_ok and no_spike:
            out.append({"ticker": tick[i], "date": p["date"].iloc[i], "price_ratio": round(float(ratio[i]), 4),
                        "volume_ratio": round(float(a / b), 3)})
    return pd.DataFrame(out, columns=["ticker", "date", "price_ratio", "volume_ratio"])


def _issue_column(df: pd.DataFrame, jumps: pd.DataFrame) -> pd.Series:
    issue = pd.Series(pd.NA, index=df.index, dtype="string")
    if not len(jumps):
        return issue
    for r in jumps.itertuples(index=False):
        m = df["ticker"].eq(r.ticker)
        dates = df.loc[m, "date"].sort_values()
        after = dates[dates >= r.date].iloc[:JUMP_EXCLUSION_SESSIONS]
        sel = m & df["date"].isin(after)
        issue[sel] = f"unadjusted price jump x{r.price_ratio:g} on {r.date.date()}"
    return issue


def build(securities: pd.DataFrame, prices: pd.DataFrame, share_events: pd.DataFrame,
          primary: pd.DataFrame) -> pd.DataFrame:
    p = paths().ensure()
    sec = securities[securities["keep"]][["ticker", "cik", "security_type", "sector"]]
    prices = prices.assign(date=prices["date"].astype("datetime64[ns]"))
    prices, trims = trim_dead_history(prices)
    trims.to_csv(p.reports / "history_trims.csv", index=False)
    df = prices.merge(sec, on="ticker", how="inner")
    df = df.merge(primary[["ticker", "is_primary"]], on="ticker", how="left")
    df["is_primary"] = df["is_primary"].fillna(False).astype(bool)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["dollar_volume"] = df["close_raw"] * df["volume"]
    df["adv50"] = (df.groupby("ticker", sort=False)["dollar_volume"]
                   .rolling(ADV_WINDOW, min_periods=ADV_WINDOW).mean().reset_index(level=0, drop=True))

    # --- shares known on each date (company level), via the primary ticker
    ev = share_events.dropna(subset=["effective", "shares_adj"]).sort_values("effective")
    ev = ev[["cik", "effective", "shares_adj", "as_of", "filed", "kind", "origin"]]
    ev = ev.assign(cik=ev["cik"].astype("int64"), effective=ev["effective"].astype("datetime64[ns]"),
                   filed=ev["filed"].astype("datetime64[ns]"), as_of=ev["as_of"].astype("datetime64[ns]"))
    prim = df[df["is_primary"]][["date", "ticker", "cik", "close_raw", "split_factor"]].copy()
    prim["cik"] = prim["cik"].astype("int64")
    prim = prim.sort_values("date")
    prim = pd.merge_asof(prim, ev, left_on="date", right_on="effective", by="cik", direction="backward")
    stale = (prim["date"] - prim["filed"]).dt.days > MAX_SHARES_STALENESS_DAYS
    prim.loc[stale, ["shares_adj"]] = np.nan
    prim["mcap"] = prim["close_raw"] / prim["split_factor"] * prim["shares_adj"]
    comp = prim[["cik", "date", "mcap", "as_of", "filed", "kind", "origin"]].rename(
        columns={"as_of": "shares_asof", "filed": "shares_filed"})
    comp["shares_source"] = comp["kind"] + ":" + comp["origin"]
    comp = comp.drop(columns=["kind", "origin"])

    df["cik_i"] = df["cik"].astype("Int64")
    df = df.merge(comp.rename(columns={"cik": "cik_i"}), on=["cik_i", "date"], how="left")
    df = df.drop(columns=["cik_i"])
    df["shares"] = df["mcap"] / df["close_raw"]  # in this ticker's as-traded units on that date

    jumps = detect_unadjusted_jumps(prices[prices["ticker"].isin(sec["ticker"])])
    df["data_issue"] = _issue_column(df, jumps)

    df["in_universe"] = (
        df["security_type"].isin(UNIVERSE_SECURITY_TYPES) & df["is_primary"]
        & (df["mcap"] >= MIN_MARKET_CAP) & (df["close_raw"] >= MIN_PRICE) & (df["adv50"] >= MIN_ADV)
        & df["data_issue"].isna())

    df = df[PANEL_COLUMNS].sort_values(["date", "ticker"]).reset_index(drop=True)
    df["sector"] = df["sector"].astype("category")
    df["security_type"] = df["security_type"].astype("category")
    df["cik"] = df["cik"].astype("Int64")
    tmp = panel_path().with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False, compression="zstd", row_group_size=200_000)
    tmp.replace(panel_path())
    jumps.to_csv(p.reports / "price_jumps.csv", index=False)
    log.info("panel: %d rows, %d tickers, %s..%s", len(df), df["ticker"].nunique(),
             df["date"].min().date(), df["date"].max().date())
    return df


def load_panel(start=None, end=None, tickers=None, columns=None, universe_only: bool = False) -> pd.DataFrame:
    """Read the daily panel.

    start/end: inclusive dates (str or Timestamp). tickers: optional list.
    columns: optional subset (date and ticker are always included).
    universe_only: keep only rows where in_universe is True.
    """
    path = panel_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with `python -m stockdata update` "
            "(or wait for the SessionStart sync; see `python -m stockdata status`).")
    filters = []
    if start is not None:
        filters.append(("date", ">=", pd.Timestamp(start)))
    if end is not None:
        filters.append(("date", "<=", pd.Timestamp(end)))
    if tickers is not None:
        filters.append(("ticker", "in", list(tickers)))
    if universe_only:
        filters.append(("in_universe", "==", True))
    cols = None
    if columns is not None:
        cols = list(dict.fromkeys(["date", "ticker", *columns]))
    tbl = pq.read_table(path, columns=cols, filters=filters or None)
    df = tbl.to_pandas()
    return df.reset_index(drop=True)

"""load_panel() and the universe flag."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockdata.config import paths


@pytest.fixture
def small_panel(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKDATA_DIR", str(tmp_path))
    from stockdata import panel
    d = pd.bdate_range("2023-01-02", "2023-06-30")
    rows = []
    # BIG: passes everything. CHEAP: price < $10. THIN: low dollar volume. SMALL: mcap < $3bn.
    spec = {"BIG": (50.0, 3_000_000, 1e9), "CHEAP": (8.0, 30_000_000, 1e9),
            "THIN": (50.0, 100_000, 1e9), "SMALL": (50.0, 3_000_000, 2e7),
            "BIGB": (49.0, 1_000_000, 1e9)}  # second share class of BIG's company
    for t, (pxl, vol, _) in spec.items():
        rows.append(pd.DataFrame({"date": d, "ticker": t, "open": pxl, "high": pxl, "low": pxl, "close": pxl,
                                  "close_raw": pxl, "volume": vol, "vwap_raw": pxl, "trades": 100,
                                  "split_factor": 1.0}))
    px = pd.concat(rows, ignore_index=True)
    cik = {"BIG": 1, "CHEAP": 2, "THIN": 3, "SMALL": 4, "BIGB": 1}
    sec = pd.DataFrame({"ticker": list(spec), "keep": True, "cik": [cik[t] for t in spec],
                        "security_type": "common", "sector": "Industrials", "ads_ratio": np.nan})
    ev = pd.DataFrame({"cik": [1, 2, 3, 4], "filed": pd.Timestamp("2022-11-01"),
                       "effective": pd.Timestamp("2022-11-02"), "as_of": pd.Timestamp("2022-10-25"),
                       "shares_adj": [spec[t][2] for t in ("BIG", "CHEAP", "THIN", "SMALL")],
                       "kind": "eso", "origin": "companyfacts"})
    prim = panel.primary_tickers(sec, px)
    panel.build(sec, px, ev, prim)
    return d


def test_load_panel_filters(small_panel):
    from stockdata import load_panel
    df = load_panel("2023-03-01", "2023-03-31")
    assert df["date"].min() >= pd.Timestamp("2023-03-01") and df["date"].max() <= pd.Timestamp("2023-03-31")
    assert set(df["ticker"]) == {"BIG", "BIGB", "CHEAP", "THIN", "SMALL"}
    one = load_panel(tickers=["BIG"], columns=["close", "mcap"])
    assert list(one.columns) == ["date", "ticker", "close", "mcap"] and set(one["ticker"]) == {"BIG"}


def test_universe_flag(small_panel):
    from stockdata import load_panel
    df = load_panel()
    late = df[df["date"] >= "2023-04-01"]  # after the 50-day ADV warm-up
    flag = late.groupby("ticker")["in_universe"].all()
    assert flag["BIG"]
    assert not late.groupby("ticker")["in_universe"].any()[["CHEAP", "THIN", "SMALL", "BIGB"]].any()
    # ADV needs 50 sessions of history before a name can qualify
    early = df[(df["ticker"] == "BIG")].sort_values("date")
    assert not early["in_universe"].iloc[:49].any() and early["in_universe"].iloc[49]
    # second share class carries the company market cap but is not in the universe
    b = df[(df["ticker"] == "BIGB") & (df["date"] == "2023-05-01")].iloc[0]
    a = df[(df["ticker"] == "BIG") & (df["date"] == "2023-05-01")].iloc[0]
    assert b["mcap"] == pytest.approx(a["mcap"]) and not b["is_primary"]


def test_universe_only(small_panel):
    from stockdata import load_panel
    u = load_panel(universe_only=True)
    assert set(u["ticker"]) == {"BIG"} and u["in_universe"].all()


def test_missing_panel_raises_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKDATA_DIR", str(tmp_path / "empty"))
    from stockdata import load_panel
    with pytest.raises(FileNotFoundError, match="python -m stockdata update"):
        load_panel()


@pytest.mark.skipif(not (paths().panel / "panel.parquet").exists(), reason="no built panel in data/")
def test_real_panel_schema_and_range():
    from stockdata import load_panel
    from stockdata.panel import PANEL_COLUMNS
    df = load_panel("2024-01-01", "2024-01-31")
    assert list(df.columns) == PANEL_COLUMNS
    assert df["date"].between("2024-01-01", "2024-01-31").all()
    assert not df.duplicated(["date", "ticker"]).any()
    u = df[df["in_universe"]]
    assert (u["mcap"] >= 3e9).all() and (u["close_raw"] >= 10).all() and (u["adv50"] >= 25e6).all()


def test_trim_dead_history_drops_previous_company():
    from stockdata.panel import trim_dead_history
    d = pd.bdate_range("2017-01-02", "2019-06-28")
    old = d[d < "2017-09-01"]
    dead = d[(d >= "2017-09-01") & (d < "2019-04-02")]
    new = d[d >= "2019-04-02"]
    px = pd.concat([
        pd.DataFrame({"date": old, "ticker": "DOW", "volume": 1_000_000}),
        pd.DataFrame({"date": dead, "ticker": "DOW", "volume": 0}),        # carried forward, no trades
        pd.DataFrame({"date": new, "ticker": "DOW", "volume": 2_000_000}),
        pd.DataFrame({"date": d, "ticker": "OK", "volume": 500}),
    ], ignore_index=True)
    px.loc[(px["ticker"] == "OK") & px["date"].isin(d[100:110]), "volume"] = 0  # short quiet spell is kept
    out, trims = trim_dead_history(px)
    assert out[out["ticker"] == "DOW"]["date"].min() == pd.Timestamp("2019-04-02")
    assert len(out[out["ticker"] == "OK"]) == len(d)
    assert list(trims["ticker"]) == ["DOW"]

"""Point-in-time logic for share counts: no filing may be used before it is public."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockdata import shares
from stockdata.config import paths


def _sessions(start="2024-01-02", end="2024-12-31"):
    # Weekdays minus a couple of holidays is enough for these tests.
    d = pd.bdate_range(start, end)
    return d[~d.isin(pd.to_datetime(["2024-07-04", "2024-11-28"]))]


def test_effective_date_is_strictly_after_filing():
    s = _sessions()
    filed = pd.Series(pd.to_datetime(["2024-05-29", "2024-05-31", "2024-06-01", "2024-07-03", "2024-11-27"]))
    eff = shares.effective_dates(filed, s)
    # Wed -> Thu, Fri -> Mon, Sat -> Mon, before a holiday -> skip it, before Thanksgiving -> Friday
    assert list(eff.dt.strftime("%Y-%m-%d")) == ["2024-05-30", "2024-06-03", "2024-06-03", "2024-07-05",
                                                 "2024-11-29"]
    assert (eff > filed).all()


def test_effective_date_lag_two_sessions():
    s = _sessions()
    eff = shares.effective_dates(pd.Series(pd.to_datetime(["2024-05-29"])), s, lag=2)
    assert eff.iloc[0] == pd.Timestamp("2024-05-31")


def _fv(**kw):
    base = {"cik": 1, "accn": "a", "form": "10-Q", "filed": pd.Timestamp("2024-05-29"),
            "eso": np.nan, "cso": np.nan, "wab": np.nan, "wad": np.nan,
            "eso_end": pd.Timestamp("2024-05-24"), "cso_end": pd.Timestamp("2024-04-28"),
            "wab_end": pd.Timestamp("2024-04-28"), "wad_end": pd.Timestamp("2024-04-28"),
            "eso_origin": "companyfacts", "cso_origin": "companyfacts", "wab_origin": "companyfacts",
            "wad_origin": "companyfacts"}
    base.update(kw)
    return pd.DataFrame([base])


def test_choose_prefers_cover_value_and_dates_it_at_cover_date():
    c = shares.choose(_fv(eso=2.46e9, cso=2.47e9))
    assert c.loc[0, "kind"] == "eso"
    assert c.loc[0, "basis_date"] == pd.Timestamp("2024-05-24")


def test_choose_rejects_cover_value_only_when_two_references_agree():
    # BABA 2026: cover said 1.86bn; balance sheet 18.58bn and weighted average 18.57bn agree
    c = shares.choose(_fv(eso=1.858e9, cso=18.58e9, wab=18.57e9))
    assert c.loc[0, "kind"] == "wab" and c.loc[0, "shares_reported"] == 18.57e9
    assert "rejected" in c.loc[0, "note"]
    # weighted counts are already restated for splits up to the filing date
    assert c.loc[0, "basis_date"] == pd.Timestamp("2024-05-29")
    # a single disagreeing reference is not enough: BKNG tags issued shares (incl. treasury) as outstanding
    c = shares.choose(_fv(eso=33.5e6, cso=64.3e6))
    assert c.loc[0, "kind"] == "eso" and c.loc[0, "shares_reported"] == 33.5e6


def test_choose_keeps_cover_after_post_quarter_issuance():
    # SPAC merger closed after quarter end: cover 64M, period-end counts 10M -> cover is the right one
    c = shares.choose(_fv(eso=64.1e6, cso=10.0e6, wab=10.05e6))
    assert c.loc[0, "kind"] == "eso"
    # cover ~1000x too high (an implausible 150bn shares) is replaced by the agreeing references
    c = shares.choose(_fv(eso=150.2e9, cso=150.0e6, wab=150.2e6))
    assert c.loc[0, "kind"] == "wab"


def test_choose_rescales_references_tagged_in_thousands():
    # CAL: cover 40.7M shares, weighted average tagged as 39,951 (thousands) -> they agree, cover kept
    c = shares.choose(_fv(eso=40.7e6, wab=39_951.0, cso=40_100.0))
    assert c.loc[0, "kind"] == "eso" and c.loc[0, "shares_reported"] == 40.7e6


def test_choose_prefers_weighted_average_over_issued_count_without_cover():
    c = shares.choose(_fv(cso=64.3e6, wab=33.4e6))
    assert c.loc[0, "kind"] == "wab"
    c = shares.choose(_fv(cso=12.29e9, wab=12.26e9))  # agree -> balance sheet (point in time)
    assert c.loc[0, "kind"] == "cso"


def test_choose_ignores_tiny_values_and_falls_back():
    c = shares.choose(_fv(eso=1.0, wab=4.2e8))  # FOXA once tagged "1"
    assert c.loc[0, "kind"] == "wab" and c.loc[0, "shares_reported"] == 4.2e8


def test_segment_total_rules():
    def rows(pairs):
        return pd.DataFrame({"segments": [p[0] for p in pairs], "value": [p[1] for p in pairs]})
    visa = rows([("ClassOfStock=CommonClassA;", 1.66e9), ("ClassOfStock=CommonClassB1AndB2;", 1.25e8),
                 ("ClassOfStock=CommonClassB1;", 5e6), ("ClassOfStock=CommonClassB2;", 1.2e8),
                 ("ClassOfStock=CommonClassC;", 9e6),
                 ("EquityComponents=CommonStockIncludingAdditionalPaidInCapital;", 1.794e9)])
    assert shares._segment_total(visa) == 1.794e9
    no_total = visa.iloc[:5]
    assert shares._segment_total(no_total) == pytest.approx(1.66e9 + 5e6 + 1.2e8 + 9e6)  # B1AndB2 aggregate dropped
    brk = rows([("ClassOfStock=EquivalentClassA;", 1.438e6), ("ClassOfStock=EquivalentClassB;", 2.157e9)])
    assert shares._segment_total(brk) == 2.157e9


def test_split_basis_nvda_like():
    """A pre-split cover count is scaled by the split; a post-split one is not."""
    ev = pd.DataFrame({
        "cik": [1, 1], "accn": ["q1", "q2"], "form": ["10-Q", "10-Q"],
        "filed": pd.to_datetime(["2024-05-29", "2024-08-28"]),
        "shares_reported": [2.46e9, 24.53e9], "kind": ["eso", "eso"], "origin": ["companyfacts"] * 2,
        "as_of": pd.to_datetime(["2024-05-24", "2024-08-23"]),
        "basis_date": pd.to_datetime(["2024-05-24", "2024-08-23"]), "note": ["", ""]})
    d = _sessions()
    sf = pd.DataFrame({"ticker": "NVDA", "date": d, "split_factor": np.where(d < "2024-06-10", 10.0, 1.0)})
    prim = pd.DataFrame({"cik": [1], "ticker": ["NVDA"], "security_type": ["common"], "ads_ratio": [np.nan]})
    out = shares.to_today_basis(ev, sf, prim)
    assert out["shares_adj"].tolist() == pytest.approx([24.6e9, 24.53e9])


def test_adr_shares_are_converted_to_ads_units():
    ev = pd.DataFrame({"cik": [2], "accn": ["f"], "form": ["20-F"], "filed": pd.to_datetime(["2024-04-18"]),
                       "shares_reported": [25.93e9], "kind": ["eso"], "origin": ["companyfacts"],
                       "as_of": pd.to_datetime(["2023-12-31"]), "basis_date": pd.to_datetime(["2023-12-31"]),
                       "note": [""]})
    sf = pd.DataFrame({"ticker": "TSM", "date": _sessions(), "split_factor": 1.0})
    prim = pd.DataFrame({"cik": [2], "ticker": ["TSM"], "security_type": ["adr"], "ads_ratio": [5.0]})
    out = shares.to_today_basis(ev, sf, prim)
    assert out.loc[0, "shares_adj"] == pytest.approx(25.93e9 / 5)


def test_clean_series_drops_isolated_outlier_and_backdated_amendment():
    filed = pd.to_datetime(["2024-02-01", "2024-05-01", "2024-08-01", "2024-09-01", "2024-11-01"])
    as_of = pd.to_datetime(["2024-01-20", "2024-04-20", "2024-07-20", "2023-12-31", "2024-10-20"])
    ev = pd.DataFrame({"cik": 3, "accn": list("abcde"), "filed": filed, "as_of": as_of,
                       "shares_reported": [1e9, 1.01e8, 1.02e9, 0.9e9, 1.03e9],
                       "shares_adj": [1e9, 1.01e8, 1.02e9, 0.9e9, 1.03e9]})
    kept, probs = shares.clean_series(ev)
    assert list(kept["accn"]) == ["a", "c", "e"]  # b = 10x typo, d = amendment about an older date
    assert "isolated outlier dropped" in set(probs["issue"])


def test_panel_never_uses_a_filing_before_it_is_filed(tmp_path, monkeypatch):
    """End to end through panel.build on synthetic data."""
    monkeypatch.setenv("STOCKDATA_DIR", str(tmp_path))
    from stockdata import panel
    d = _sessions("2024-01-02", "2024-03-29")
    px = pd.DataFrame({"date": d, "ticker": "AAA", "open": 20.0, "high": 21.0, "low": 19.0, "close": 20.0,
                       "close_raw": 20.0, "volume": 2_000_000, "vwap_raw": 20.0, "trades": 1000,
                       "split_factor": 1.0})
    sec = pd.DataFrame({"ticker": ["AAA"], "keep": [True], "cik": [7], "security_type": ["common"],
                        "sector": ["Industrials"], "ads_ratio": [np.nan]})
    filed = pd.to_datetime(["2023-11-01", "2024-02-07"])
    ev = pd.DataFrame({"cik": [7, 7], "filed": filed, "as_of": filed - pd.Timedelta(days=5),
                       "shares_adj": [1e8, 2e8], "kind": ["eso", "eso"], "origin": ["companyfacts"] * 2})
    ev["effective"] = shares.effective_dates(ev["filed"], pd.DatetimeIndex(d.append(pd.DatetimeIndex([]))).union(
        pd.bdate_range("2023-10-02", "2023-12-29")))
    prim = panel.primary_tickers(sec, px)
    out = panel.build(sec, px, ev, prim)
    assert (out["shares_filed"] < out["date"]).all()
    feb7 = out[out["date"] == "2024-02-07"].iloc[0]
    feb8 = out[out["date"] == "2024-02-08"].iloc[0]
    assert feb7["shares"] == pytest.approx(1e8)       # filed that day: not usable yet
    assert feb8["shares"] == pytest.approx(2e8)       # usable the next session
    assert feb8["mcap"] == pytest.approx(20.0 * 2e8)


@pytest.mark.skipif(not (paths().panel / "panel.parquet").exists(), reason="no built panel in data/")
def test_real_panel_shares_are_point_in_time():
    from stockdata import load_panel
    df = load_panel(columns=["shares_filed", "shares_asof", "mcap"])
    known = df.dropna(subset=["shares_filed"])
    assert len(known) > 0
    assert (known["shares_filed"] < known["date"]).all(), "a filing was used on or before its filing date"
    assert (known["shares_asof"] <= known["date"]).all()


def test_fsds_totals_vectorised_matches_rules():
    rows = [  # (accn, segments, value)
        ("visa", "ClassOfStock=CommonClassA;", 1.66e9), ("visa", "ClassOfStock=CommonClassB1AndB2;", 1.25e8),
        ("visa", "ClassOfStock=CommonClassB1;", 5e6), ("visa", "ClassOfStock=CommonClassB2;", 1.2e8),
        ("visa", "ClassOfStock=CommonClassC;", 9e6),
        ("visa", "EquityComponents=CommonStockIncludingAdditionalPaidInCapital;", 1.794e9),
        ("brk", "ClassOfStock=EquivalentClassA;", 1.438e6), ("brk", "ClassOfStock=EquivalentClassB;", 2.157e9),
        ("goog", None, 12.116e9), ("goog", "ClassOfStock=CommonClassA;", 5.824e9),
        ("meta", "ClassOfStock=CommonClassA;", 2.196e9), ("meta", "ClassOfStock=CommonClassB;", 3.42e8),
    ]
    f = pd.DataFrame([{"cik": 1, "accn": a, "kind": "cso", "end": pd.Timestamp("2026-03-31"), "form": "10-Q",
                       "filed": pd.Timestamp("2026-04-29"), "segments": s, "value": v} for a, s, v in rows])
    got = shares._fsds_totals(f).set_index("accn")
    assert got.loc["visa", "val"] == 1.794e9
    assert got.loc["brk", "val"] == 2.157e9
    assert got.loc["goog", "val"] == 12.116e9 and got.loc["goog", "origin"] == "fsds"
    assert got.loc["meta", "val"] == pytest.approx(2.538e9) and got.loc["meta", "origin"] == "fsds_classes"


def test_as_of_after_filing_is_clamped():
    c = shares.choose(_fv(eso=5e7, eso_end=pd.Timestamp("2034-05-24")))  # typo year
    assert c.loc[0, "as_of"] == c.loc[0, "filed"] and "clamped" in c.loc[0, "note"]


def test_units_error_fixed_by_turnover():
    """CPK-like: three covers 1,000x too high, then correct values."""
    d = pd.bdate_range("2025-01-02", "2026-06-30")
    px = pd.DataFrame({"ticker": "CPK", "date": d, "close_raw": 120.0, "volume": 170_000, "split_factor": 1.0})
    filed = pd.to_datetime(["2025-05-07", "2025-08-07", "2025-11-06", "2026-02-25", "2026-05-06"])
    vals = [23.4e9, 23.5e9, 23.6e9, 23.9e6, 24.0e6]
    ev = pd.DataFrame({"cik": 9, "accn": list("abcde"), "filed": filed, "as_of": filed - pd.Timedelta(days=3),
                       "shares_reported": vals, "shares_adj": vals, "note": "", "kind": "eso"})
    ev["effective"] = shares.effective_dates(ev["filed"], d)
    prim = pd.DataFrame({"cik": [9], "ticker": ["CPK"]})
    out, fixes = shares.fix_units_errors(ev, px, prim)
    assert out["shares_adj"].tolist() == pytest.approx([23.4e6, 23.5e6, 23.6e6, 23.9e6, 24.0e6])
    assert len(fixes) == 3
    # a genuine big change (shell -> listed company, x22,000) is left alone
    ev2 = ev.assign(shares_adj=[2.5e4, 5.5e8, 5.5e8, 5.5e8, 5.5e8], shares_reported=[2.5e4, 5.5e8, 5.5e8, 5.5e8, 5.5e8])
    out2, fixes2 = shares.fix_units_errors(ev2, px, prim)
    assert not fixes2 and out2["shares_adj"].iloc[0] == 2.5e4


def _rescue_case(shares_val, pf_val, close=50.0, volume=2_000_000):
    d = pd.bdate_range("2025-01-02", "2025-12-31")
    px = pd.DataFrame({"ticker": "TEM", "date": d, "close_raw": close, "volume": volume, "split_factor": 1.0})
    filed = pd.to_datetime(["2025-02-24", "2025-05-05", "2025-08-08", "2025-11-04"])
    ev = pd.DataFrame({"cik": 5, "accn": list("abcd"), "filed": filed, "as_of": filed - pd.Timedelta(days=30),
                       "shares_reported": shares_val, "shares_adj": shares_val, "note": "", "kind": "wab"})
    ev["effective"] = shares.effective_dates(ev["filed"], d)
    pf = pd.DataFrame({"cik": [5], "end": [pd.Timestamp("2025-06-30")], "val": [pf_val]})
    return shares.rescue_units(ev, px, pd.DataFrame({"cik": [5], "ticker": ["TEM"]}), pf)


def test_rescue_whole_series_in_thousands():
    # 173,381 "shares" at $50 trading $100m/day = 1,150% turnover; float $6bn agrees with 173M shares
    out, fixes = _rescue_case(173_381.0, 6.0e9)
    assert out["shares_adj"].iloc[0] == pytest.approx(173_381_000.0) and len(fixes) == 1


def test_rescue_never_inflates_a_small_hot_stock():
    # really 400k shares, $20m float: hot microcap with 500%/day turnover stays as reported
    out, fixes = _rescue_case(400_000.0, 2.0e7)
    assert out["shares_adj"].iloc[0] == 400_000.0 and not fixes


def test_fractional_small_count_is_thousands():
    c = shares.choose(_fv(eso=673_556.656))
    assert c.loc[0, "shares_reported"] == pytest.approx(673_556_656.0)
    c = shares.choose(_fv(eso=64_562_265.5))  # large fractional count (DRIP shares) is real
    assert c.loc[0, "shares_reported"] == 64_562_265.5


def test_units_fix_leaves_plausible_segments_alone():
    """Tiny company recapitalised into a large one: both pieces trade normally, nothing is rescaled."""
    d = pd.bdate_range("2023-01-02", "2025-12-31")
    vol = np.where(d < "2024-06-01", 3_000, 2_000_000)  # tiny shell trades little, the new company a lot
    px = pd.DataFrame({"ticker": "QXO", "date": d, "close_raw": 12.0, "volume": vol, "split_factor": 1.0})
    filed = pd.to_datetime(["2023-05-09", "2023-11-14", "2024-05-07", "2024-08-14", "2025-05-08"])
    vals = [5.26e6, 5.26e6, 5.32e6, 5.18e9, 5.19e9]  # ~1000x apart on purpose
    ev = pd.DataFrame({"cik": 3, "accn": list("abcde"), "filed": filed, "as_of": filed - pd.Timedelta(days=5),
                       "shares_reported": vals, "shares_adj": vals, "note": "", "kind": "eso"})
    ev["effective"] = shares.effective_dates(ev["filed"], d)
    out, fixes = shares.fix_units_errors(ev, px, pd.DataFrame({"cik": [3], "ticker": ["QXO"]}))
    assert not fixes and out["shares_adj"].tolist() == vals

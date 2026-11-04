"""Earnings dates from 8-K Item 2.02: parsing, release timing and the session that reacts."""
from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

from stockdata import earnings

# 2024-07-04 is a holiday; 2024-07-06/07 a weekend
CAL = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-07-01", "2024-07-02", "2024-07-03",
                        "2024-07-05", "2024-07-08"])


@pytest.mark.parametrize("utc, session, timing", [
    ("2024-07-02T11:00:00Z", "2024-07-02", "before_open"),       # 07:00 New York (EDT)
    ("2024-07-02T14:00:00Z", "2024-07-02", "during_session"),    # 10:00
    ("2024-07-02T19:59:59Z", "2024-07-02", "during_session"),    # 15:59:59
    ("2024-07-02T20:05:00Z", "2024-07-03", "after_close"),       # 16:05
    ("2024-07-03T20:30:00Z", "2024-07-05", "after_close"),       # day before the holiday
    ("2024-07-04T12:00:00Z", "2024-07-05", "non_trading_day"),   # on the holiday
    ("2024-07-06T13:00:00Z", "2024-07-08", "non_trading_day"),   # Saturday
    ("2024-07-03T02:30:00Z", "2024-07-03", "after_close"),       # 22:30 New York on 07-02
    ("2024-01-03T20:30:00Z", "2024-01-03", "during_session"),    # winter: 15:30 EST
    ("2024-01-03T21:30:00Z", "2024-01-04", "after_close"),       # winter: 16:30 EST
    ("2024-07-08T21:00:00Z", "2024-07-09", "after_close"),       # past the calendar: next weekday
])
def test_reaction_session(utc, session, timing):
    s, t = earnings.reaction_session(pd.Series(pd.to_datetime([utc], utc=True)), CAL)
    assert s.iloc[0] == pd.Timestamp(session) and t.iloc[0] == timing


def _block(rows):
    keys = ["form", "items", "filingDate", "acceptanceDateTime", "accessionNumber"]
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


def test_rows_keep_only_original_8k_with_item_202():
    b = _block([
        ("8-K", "2.02,9.01", "2020-02-03", "2020-02-03T21:05:00.000Z", "a1"),
        ("8-K/A", "2.02", "2020-02-05", "2020-02-05T21:05:00.000Z", "a2"),
        ("8-K", "5.02", "2020-03-01", "2020-03-01T21:05:00.000Z", "a3"),
        ("10-Q", "", "2020-05-01", "2020-05-01T21:05:00.000Z", "a4"),
        ("8-K", "2.02", "2010-02-03", "2010-02-03T21:05:00.000Z", "a5"),  # before SINCE
        ("8-K", "12.02", "2020-06-01", "2020-06-01T21:05:00.000Z", "a6"),  # not item 2.02
    ])
    assert [r[1] for r in earnings._rows(b, 7)] == ["a1"]


def test_build_reads_older_files_and_predecessors(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKDATA_DIR", str(tmp_path))
    from stockdata.config import paths
    p = paths().ensure()
    ev = lambda d, a: ("8-K", "2.02,9.01", d, f"{d}T21:05:00.000Z", a)  # noqa: E731  (16:05 / 17:05 NY)
    subs = {
        # OLD (cik 1): the company's filer before a holding-company reorganisation into cik 2
        "CIK0000000001.json": {"filings": {"recent": _block([ev("2019-02-04", "o1"), ev("2019-05-06", "o2"),
                                                             ev("2019-08-05", "o3")]), "files": []}},
        "CIK0000000002.json": {"filings": {"recent": _block([ev("2019-08-05", "n0"), ev("2019-11-04", "n1")]),
                                           "files": []}},
        # cik 3: recent block holds only new filings, the earnings 8-K is in an older file
        "CIK0000000003.json": {"filings": {"recent": _block([("4", "", "2024-01-02", "2024-01-02T21:05:00.000Z",
                                                              "x")]),
                                           "files": [{"name": "CIK0000000003-submissions-001.json",
                                                      "filingFrom": "2016-01-01", "filingTo": "2023-12-29"}]}},
        "CIK0000000003-submissions-001.json": _block([ev("2018-04-30", "t1")]),
    }
    with zipfile.ZipFile(p.raw_sec / "submissions.zip", "w") as z:
        for name, d in subs.items():
            z.writestr(name, json.dumps(d))
    pd.DataFrame({"cik": [2], "predecessor_cik": [1], "name": ["New"], "predecessor_name": ["Old"],
                  "gap_days": [0.0], "evidence": ["public float"]}).to_csv(p.reports / "successor_links.csv",
                                                                           index=False)
    sec = pd.DataFrame({"ticker": ["NEW", "NEWB", "THREE", "DROP"], "keep": [True, True, True, False],
                        "cik": [2, 2, 3, 3]})
    cal = pd.bdate_range("2018-01-01", "2020-01-31")
    out = earnings.build(sec, cal, workers=1)
    new = out[out["ticker"] == "NEW"]
    # predecessor releases before the new CIK's first one, then the new CIK's own
    assert new["accession"].tolist() == ["o1", "o2", "n0", "n1"]
    assert new["session"].tolist() == [pd.Timestamp(d) for d in ("2019-02-05", "2019-05-07", "2019-08-06",
                                                                  "2019-11-05")]
    assert (out[out["ticker"] == "NEWB"]["accession"].tolist() == new["accession"].tolist())  # second class
    assert out[out["ticker"] == "THREE"]["accession"].tolist() == ["t1"]
    assert "DROP" not in set(out["ticker"])
    assert earnings.load(["THREE"])["session"].iloc[0] == pd.Timestamp("2018-05-01")

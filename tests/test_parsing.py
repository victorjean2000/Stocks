"""Security classification, ADS ratio parsing, split factors from volumes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockdata.adr import parse_ratio
from stockdata.alpaca import _snap, split_factor_from_volumes
from stockdata.sectors import sic_to_sector
from stockdata.universe import classify

OP = {"sic": 3571, "n_periodic_forms": 40, "n_fund_forms": 0}


@pytest.mark.parametrize("symbol,name,exchange,sec,expected", [
    ("AAPL", "Apple Inc. Common Stock", "NASDAQ", OP, "common"),
    ("BRK.B", "BERKSHIRE HATHAWAY Class B", "NYSE", OP, "common"),
    ("BAC.PRL", "Bank Of America 7.25% Non-Cumulative Perpetual Convertible Preferred Stock", "NYSE", OP, "preferred"),
    ("MTNE.WS", "Mytheresa warrants", "NYSE", None, "warrant"),
    ("EVAC.U", "EVe Acquisition Units", "NYSE", None, "unit"),
    ("SOUL.RT", "Soulpower rights", "NYSE", None, "right"),
    ("TLACW", "Three Lions Acquisition Corp. Warrants", "NASDAQ", None, "warrant"),
    ("CATLR", "Catalyst Acquisition Corp. Rights", "NASDAQ", None, "right"),
    ("TLACU", "Three Lions Acquisition Corp. Units", "NASDAQ", None, "unit"),
    ("SPY", "State Street SPDR S&P 500 ETF Trust", "ARCA", None, "etf_fund"),
    ("QQQ", "Invesco QQQ Trust, Series 1", "NASDAQ", None, "etf_fund"),
    ("GLD", "SPDR Gold Trust, SPDR Gold Shares", "ARCA", {"sic": 6221, "n_periodic_forms": 9, "n_fund_forms": 0},
     "etf_fund"),
    ("SOJC", "The Southern Company Series 2017B 5.25% Junior Subordinated Notes due December 1, 2077", "NYSE", OP,
     "debt"),
    ("ET", "Energy Transfer LP Common Units", "NYSE", OP, "partnership"),
    ("ARCC", "Ares Capital Corporation Common Stock", "NASDAQ", {"sic": None, "n_periodic_forms": 50,
                                                                "n_fund_forms": 0}, "common"),
    ("MSDL", "Morgan Stanley Direct Lending Fund", "NYSE", {"sic": None, "n_periodic_forms": 27, "n_fund_forms": 0},
     "common"),  # a BDC, files 10-K
    ("PDI", "PIMCO Dynamic Income Fund", "NYSE", {"sic": None, "n_periodic_forms": 0, "n_fund_forms": 12},
     "etf_fund"),
    ("VNO", "Vornado Realty Trust", "NYSE", {"sic": 6798, "n_periodic_forms": 60, "n_fund_forms": 0}, "common"),
    ("PVL", "Permianville Royalty Trust", "NYSE", {"sic": 1311, "n_periodic_forms": 59, "n_fund_forms": 0},
     "trust_units"),
    ("BABA", "Alibaba Group Holding Limited American Depositary Shares, each represents eight Ordinary Shares",
     "NYSE", OP, "adr"),
    ("XYZ", "Some OTC Co", "OTC", None, "otc"),
])
def test_classify(symbol, name, exchange, sec, expected):
    assert classify(symbol, name, exchange, "", sec)[0] == expected


@pytest.mark.parametrize("text,ratio", [
    ("American Depositary Shares, each representing five common shares", 5),
    ("ADSs, each of which represents 1/10th of one ordinary share", 0.1),
    ("American Depositary Shares, each representing one-half of one Ordinary Share", 0.5),
    ("two ADSs represent one ordinary share", 0.5),
    ("each ADS represents five (5) common shares", 5),
    ("American Depositary Shares, each representing thirty (30) Common Shares", 30),
])
def test_parse_ratio(text, ratio):
    assert parse_ratio(text)[0] == pytest.approx(ratio)


def test_split_factor_from_volumes_handles_rounding_and_zero_volume():
    rng = np.random.default_rng(0)
    raw = rng.integers(10_000, 5_000_000, 300).astype(float)
    raw[50] = 0       # no trades that day
    raw[60] = 300     # tiny volume, unreliable ratio
    true = np.r_[np.full(100, 40.0), np.full(100, 10.0), np.full(100, 1.0)]
    adj = np.round(raw * true)
    adj[60] = 11_999  # rounding noise on the tiny day
    sf = split_factor_from_volumes(pd.Series(adj), pd.Series(raw))
    assert np.array_equal(sf, true)


def test_split_factor_reverse_split():
    raw = np.full(200, 1_000_000.0)
    true = np.r_[np.full(120, 1 / 8), np.ones(80)]
    sf = split_factor_from_volumes(pd.Series(np.round(raw * true)), pd.Series(raw))
    assert np.allclose(sf, true)


def test_snap():
    assert _snap(3.99985) == 4 and _snap(0.12502) == 0.125 and _snap(1.0526) == pytest.approx(1 / 0.95)


@pytest.mark.parametrize("sic,cik,sector", [
    (3674, None, "Information Technology"), (2834, None, "Health Care"), (6022, None, "Financials"),
    (6798, None, "Real Estate"), (4911, None, "Utilities"), (1311, None, "Energy"), (5331, None, "Consumer Staples"),
    (7370, 1326801, "Communication Services"), (3711, None, "Consumer Discretionary"), (811, None, "Materials"),
    (None, None, None),
])
def test_sector_map(sic, cik, sector):
    assert sic_to_sector(sic, cik) == sector


@pytest.mark.parametrize("new,old,match", [
    ("ExxonMobil Holdings Corp", "EXXON MOBIL CORP", True),
    ("Bunge Global SA", "Bunge Ltd", True),
    ("AngloGold Ashanti plc", "ANGLOGOLD ASHANTI LTD", True),
    ("BlackRock, Inc.", "BlackRock Finance, Inc.", True),
    ("ExxonMobil Holdings Corp", "DONNELLEY FINANCIAL, LLC", False),
    ("Meta Platforms, Inc.", "Metallus Inc.", False),
])
def test_successor_name_match(new, old, match):
    from stockdata.successors import _names_match
    assert _names_match(new, old) is match

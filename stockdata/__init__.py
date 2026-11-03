"""Free daily dataset for backtesting a US momentum strategy.

    from stockdata import load_panel
    df = load_panel("2020-01-01", "2024-12-31")                 # all rows
    uni = load_panel("2020-01-01", universe_only=True)          # tradable universe only

See DATA.md for sources, schema and refresh instructions.
"""
from __future__ import annotations

import pandas as pd

from .config import paths
from .panel import load_panel

__all__ = ["load_panel", "load_securities", "load_prices", "load_share_events", "data_status"]


def load_securities() -> pd.DataFrame:
    """Security master: one row per Alpaca asset with type, CIK, SIC, sector, drop reason."""
    return pd.read_parquet(paths().universe / "securities.parquet")


def load_prices(tickers=None) -> pd.DataFrame:
    """Raw price store (all kept tickers, all dates)."""
    filters = [("ticker", "in", list(tickers))] if tickers is not None else None
    return pd.read_parquet(paths().prices / "daily.parquet", filters=filters)


def load_share_events() -> pd.DataFrame:
    """Point-in-time share counts: one row per filing used (cik, filed, effective, shares_adj...)."""
    return pd.read_parquet(paths().sec / "share_events.parquet")


def data_status() -> dict:
    import json
    p = paths()
    st = json.loads(p.state_file.read_text()) if p.state_file.exists() else {}
    st["panel_exists"] = (p.panel / "panel.parquet").exists()
    sync = p.root / "sync_status.json"
    st["sync"] = json.loads(sync.read_text()) if sync.exists() else None
    st.pop("coverage", None)
    return st

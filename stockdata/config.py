"""Central settings: paths, dates, universe thresholds.

Everything that a backtest might want to tweak lives here so the rest of the
code can stay free of magic numbers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# History window for prices. Shares are read from earlier filings too, so the
# value known on 2016-01-04 comes from a 2015 filing.
PRICE_START = "2016-01-01"
FILINGS_START = "2014-01-01"

# Alpaca "exchange" values we treat as major US exchanges. OTC is excluded.
MAJOR_EXCHANGES = ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "NYSEARCA", "TXSE")

# Universe rules, applied point-in-time on each date.
MIN_MARKET_CAP = 3e9
MIN_PRICE = 10.0
MIN_ADV = 25e6
ADV_WINDOW = 50

# Security types that may enter the universe flag. ADRs are kept in the panel
# (prices + best-effort market cap) but excluded from the flag by default,
# because "US-listed common stock" usually means ordinary shares.
UNIVERSE_SECURITY_TYPES = ("common",)

# A filing becomes usable on the first trading day AFTER its filing date. EDGAR
# filing dates do not say whether the filing landed before or after the close,
# so a one-day lag guarantees nothing is used before the market could see it.
FILING_LAG_TRADING_DAYS = 1

# Shares older than this (from the last usable filing) are treated as unknown.
MAX_SHARES_STALENESS_DAYS = 550


def data_dir() -> Path:
    """Root of all data. Override with STOCKDATA_DIR (tests do this)."""
    return Path(os.environ.get("STOCKDATA_DIR", REPO_ROOT / "data"))


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw_sec(self) -> Path:
        return self.root / "raw" / "sec"

    @property
    def raw_fsds(self) -> Path:
        return self.root / "raw" / "sec" / "fsds"

    @property
    def raw_alpaca(self) -> Path:
        return self.root / "raw" / "alpaca"

    @property
    def universe(self) -> Path:
        return self.root / "universe"

    @property
    def prices(self) -> Path:
        return self.root / "prices"

    @property
    def sec(self) -> Path:
        return self.root / "sec"

    @property
    def panel(self) -> Path:
        return self.root / "panel"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    def ensure(self) -> "Paths":
        for p in (self.raw_sec, self.raw_fsds, self.raw_alpaca, self.universe, self.prices,
                  self.sec, self.panel, self.reports, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def paths() -> Paths:
    return Paths(data_dir())

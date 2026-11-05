"""Minimal client for one Alpaca PAPER trading account. The live trading API is never used."""
from __future__ import annotations

import time

import requests

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class BrokerError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


class Paper:
    """A paper account, reached with that account's own API key pair."""

    def __init__(self, key: str, secret: str):
        self.base = PAPER_URL
        self.session = requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                                     "Accept": "application/json"})

    def _call(self, method: str, url: str, retry: bool = True, **kw):
        delay, last = 1.0, ""
        for _ in range(5 if retry else 1):
            try:
                r = self.session.request(method, url, timeout=30, **kw)
            except requests.RequestException as e:
                last = type(e).__name__
            else:
                if r.status_code < 500 and r.status_code != 429:
                    if not r.ok:
                        raise BrokerError(f"{method} {url.split('.markets')[-1]}: {r.status_code} {r.text[:300]}",
                                          r.status_code)
                    return r.json() if r.content else None
                last = f"{r.status_code} {r.text[:200]}"
            if retry:
                time.sleep(delay)
                delay *= 2
        raise BrokerError(f"{method} {url.split('.markets')[-1]} failed: {last}")

    def _get(self, path: str, **params):
        return self._call("GET", self.base + path, params=params or None)

    def account(self) -> dict:
        return self._get("/v2/account")

    def positions(self) -> list[dict]:
        return self._get("/v2/positions")

    def orders(self, status: str = "open", after: str | None = None, limit: int = 500) -> list[dict]:
        params = {"status": status, "limit": limit, "direction": "desc"}
        if after:
            params["after"] = after
        return self._get("/v2/orders", **params)

    def calendar(self, start: str, end: str) -> list[dict]:
        return self._get("/v2/calendar", start=start, end=end)

    def clock(self) -> dict:
        return self._get("/v2/clock")

    def asset(self, symbol: str) -> dict:
        return self._get(f"/v2/assets/{symbol}")

    def equity_history(self, period: str = "1A") -> "list[tuple[int, float]]":
        """(unix time, equity) at each daily close, from the account's portfolio history."""
        h = self._get("/v2/account/portfolio/history", period=period, timeframe="1D")
        return [(t, float(e)) for t, e in zip(h.get("timestamp") or [], h.get("equity") or []) if e]

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """Latest IEX bar close (real time on the free plan), for sizing whole-share orders."""
        if not symbols:
            return {}
        bars = self._call("GET", f"{DATA_URL}/v2/stocks/bars/latest",
                          params={"symbols": ",".join(symbols), "feed": "iex"}).get("bars", {})
        return {s: float(b["c"]) for s, b in bars.items()}

    def cancel(self, order_id: str) -> None:
        self._call("DELETE", f"{self.base}/v2/orders/{order_id}")

    def order_by_client_id(self, cid: str) -> dict | None:
        try:
            return self._get("/v2/orders:by_client_order_id", client_order_id=cid)
        except BrokerError as e:
            if e.status == 404:
                return None
            raise

    def submit(self, symbol: str, side: str, cid: str, qty: str | None = None,
               notional: float | None = None) -> dict:
        """Market order, good for the day: placed after the close it waits for the next open."""
        body = {"symbol": symbol, "side": side, "type": "market", "time_in_force": "day", "client_order_id": cid}
        if (qty is None) == (notional is None):
            raise ValueError("give exactly one of qty and notional")
        if qty is not None:
            body["qty"] = qty
        else:
            body["notional"] = f"{notional:.2f}"
        try:
            return self._call("POST", f"{self.base}/v2/orders", retry=False, json=body)
        except BrokerError as e:
            if e.status is not None and e.status < 500:
                raise  # rejected: nothing was placed
            found = self.order_by_client_id(cid)  # a lost reply may hide an accepted order
            if found:
                return found
            raise

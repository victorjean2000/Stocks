"""Orders that move an account from its holdings to target weights (pure functions, no I/O)."""
from __future__ import annotations

import math
from dataclasses import dataclass

CASH_BUFFER = 0.005  # keep 0.5% in cash so fills at the open do not need margin
MIN_TRADE = 0.002    # skip trades under 0.2% of the account (and under $10)
TOP_UP = 0.02        # idle cash above 2% of the account is invested in the targets


@dataclass
class Holding:
    symbol: str
    qty: str  # as the broker reports it, so a full exit sells exactly this
    value: float
    price: float


@dataclass
class Order:
    symbol: str
    side: str  # buy | sell
    qty: str | None = None
    notional: float | None = None
    why: str = ""

    def value(self, holdings: dict[str, Holding], prices: dict[str, float]) -> float:
        if self.notional is not None:
            return self.notional
        h = holdings.get(self.symbol)
        px = h.price if h else prices.get(self.symbol, 0.0)
        return float(self.qty) * px


def _shares(x: float, fractional: bool) -> str | None:
    q = math.floor(x * 1e6) / 1e6 if fractional else float(math.floor(x))
    if q <= 0:
        return None
    return f"{q:.6f}".rstrip("0").rstrip(".")


def _buy(symbol: str, amount: float, fractional: bool, prices: dict[str, float], why: str) -> Order | None:
    if fractional:
        cents = math.floor(amount * 100) / 100
        return Order(symbol, "buy", notional=cents, why=why) if cents >= 1 else None
    px = prices.get(symbol)
    if not px:
        return None
    q = _shares(amount / px, False)
    return Order(symbol, "buy", qty=q, why=why) if q else None


def plan_orders(weights: dict[str, float], holdings: dict[str, Holding], equity: float, cash: float,
                full: bool, fractionable: dict[str, bool] | None = None,
                prices: dict[str, float] | None = None) -> list[Order]:
    """full=True: trade every target to its weight (a rebalance). full=False: keep weights as they
    drifted and only fix the composition: sell what is not a target, buy missing targets at their
    weight, and invest idle cash once it is above TOP_UP. Never sells more than is held."""
    fractionable = fractionable or {}
    prices = prices or {}
    invest = equity * (1 - CASH_BUFFER)
    floor = max(10.0, MIN_TRADE * equity)
    orders: list[Order | None] = []
    for s, h in holdings.items():
        if s not in weights and float(h.qty) > 0:
            orders.append(Order(s, "sell", qty=h.qty, why="not in target"))
    for s, w in weights.items():
        h = holdings.get(s)
        if not full and h is not None:
            continue
        d = w * invest - (h.value if h else 0.0)
        if abs(d) < floor:
            continue
        if d > 0:
            orders.append(_buy(s, d, fractionable.get(s, False), prices, "rebalance" if full else "missing"))
        else:
            q = _shares(min(-d / h.price, float(h.qty)), fractionable.get(s, False))
            if q:
                orders.append(Order(s, "sell", qty=q, why="rebalance"))
    out = [o for o in orders if o is not None]
    if not full:
        sells = sum(o.value(holdings, prices) for o in out if o.side == "sell")
        buys = sum(o.value(holdings, prices) for o in out if o.side == "buy")
        spare = cash + sells - buys - CASH_BUFFER * equity
        if spare > max(floor, TOP_UP * equity):
            total = sum(weights.values())
            for s, w in weights.items():
                o = _buy(s, spare * w / total, fractionable.get(s, False), prices, "idle cash")
                if o and o.value(holdings, prices) >= floor:
                    out.append(o)
    return sorted(out, key=lambda o: o.side != "sell")

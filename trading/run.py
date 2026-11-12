"""Run each strategy against its own paper account: plan from holdings and target, then place orders."""
from __future__ import annotations

import logging
import os
import time
import uuid

import pandas as pd

from .broker import BrokerError, Paper
from .plan import Holding, plan_orders
from .targets import TARGETS

log = logging.getLogger(__name__)

# Environment variables holding each strategy's paper account keys, in order of preference.
ACCOUNTS = {
    "momentum": [("ALPACA_MOMENTUM_KEY_ID", "ALPACA_MOMENTUM_SECRET_KEY"), ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY")],
    "lqq3": [("ALPACA_LQQ3_KEY_ID", "ALPACA_LQQ3_SECRET_KEY")],
}
DESCRIPTION = {
    "momentum": "Top 10 of the 500 largest US stocks by 6-1 month return, equal weight, rebalanced after each "
                "month end",
    "lqq3": "TQQQ (3x Nasdaq-100, US stand-in for LQQ3) while QQQ closes above its 200-day average, otherwise BIL "
            "(T-bills)",
}


def account_for(name: str) -> Paper | None:
    for key_var, secret_var in ACCOUNTS[name]:
        key, secret = os.environ.get(key_var), os.environ.get(secret_var)
        if key and secret:
            return Paper(key, secret)
    return None


def holdings_of(paper: Paper) -> dict[str, Holding]:
    out = {}
    for p in paper.positions():
        if float(p["qty"]) < 0:
            raise RuntimeError(f"short position in {p['symbol']}: this account is not managed by the strategy")
        out[p["symbol"]] = Holding(p["symbol"], p["qty"], float(p["market_value"]), float(p["current_price"]))
    return out


def acted_on(paper: Paper, tag: str, since: pd.Timestamp) -> bool:
    """Has a rebalance order for this decision been (at least partly) filled?"""
    after = (since - pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    return any((o.get("client_order_id") or "").startswith(f"{tag}-rebal") and float(o.get("filled_qty") or 0) > 0
               for o in paper.orders("closed", after=after))


def cancel_own(paper: Paper, name: str) -> int:
    mine = [o for o in paper.orders("open") if (o.get("client_order_id") or "").startswith(f"{name}-")]
    for o in mine:
        paper.cancel(o["id"])
    for _ in range(20):  # cancels are asynchronous
        if not any((o.get("client_order_id") or "").startswith(f"{name}-") for o in paper.orders("open")):
            break
        time.sleep(1)
    return len(mine)


def account_return(paper: Paper, start: str) -> dict:
    """The account's return since `start` (its last daily equity before then), in dollars and in pounds."""
    from backtest.lqq3 import sterling
    hist = paper.equity_history()
    acct = paper.account()
    t0 = pd.Timestamp(start, tz="America/New_York").timestamp()
    before = [e for t, e in hist if t < t0]
    base = before[-1] if before else (hist[0][1] if hist else float(acct["last_equity"]))
    usd = float(acct["equity"]) / base - 1
    fx, _ = sterling(str((pd.Timestamp(start) - pd.Timedelta(days=10)).date()))
    fx0 = fx[fx.index < pd.Timestamp(start)].iloc[-1]
    gbp = (1 + usd) / (fx.iloc[-1] / fx0) - 1
    return {"in dollars": f"{usd:+.2%}", "in pounds": f"{gbp:+.2%}"}


def lqq3_extras(paper: Paper) -> dict:
    """For the lqq3 account: what real LQQ3 did in pounds on the same signals, and the account's own return."""
    from backtest.lqq3 import SHADOW_START, shadow
    out = {}
    for label, fn in (("LQQ3 in a sterling ISA (same signals, real LQQ3 prices)", shadow),
                      (f"this account (TQQQ) since {SHADOW_START}", lambda: account_return(paper, SHADOW_START))):
        try:
            out[label] = fn()
        except Exception as e:  # tracking must never block trading
            out[label] = f"unavailable: {type(e).__name__}: {e}"
    return out


def owner_of(paper: Paper, name: str) -> str | None:
    """Another strategy whose orders this account holds (its client_order_id prefix), if any. Catches keys
    that point at the wrong account, which the same-account check cannot see when one strategy runs alone."""
    others = [n for n in ACCOUNTS if n != name]
    for o in paper.orders("all", limit=500):
        cid = o.get("client_order_id") or ""
        for other in others:
            if cid.startswith(f"{other}-"):
                return other
    return None


def run_one(name: str, paper: Paper, dry_run: bool) -> dict:
    acct = paper.account()
    if acct.get("status") != "ACTIVE" or acct.get("trading_blocked") or acct.get("account_blocked"):
        raise RuntimeError(f"account not tradable (status {acct.get('status')})")
    target = TARGETS[name](paper)
    cancelled = 0 if dry_run else cancel_own(paper, name)
    holdings = holdings_of(paper)
    acct = paper.account()
    equity, cash = float(acct["equity"]), float(acct["cash"])
    full = target.once and not acted_on(paper, target.tag, target.decided)
    fractionable = {s: bool(paper.asset(s).get("fractionable")) for s in target.weights}
    prices = paper.latest_prices([s for s, f in fractionable.items() if not f])
    orders = plan_orders(target.weights, holdings, equity, cash, full, fractionable, prices)
    clock = paper.clock()
    placed, errors = [], []
    for o in orders:
        cid = f"{target.tag}-{'rebal' if full else 'fix'}-{o.symbol}-{o.side}-{uuid.uuid4().hex[:8]}"
        if dry_run:
            continue
        try:
            r = paper.submit(o.symbol, o.side, cid, qty=o.qty, notional=o.notional)
            placed.append(r.get("status"))
        except BrokerError as e:
            errors.append(f"{o.side} {o.symbol}: {e}")
    pct = lambda v: f"{v / equity:.1%}" if equity else "n/a"  # noqa: E731
    return {
        "strategy": name, "rule": DESCRIPTION[name], "decided at close of": str(target.decided.date()),
        "data through": str(target.data_through.date()), "mode": "rebalance" if full else "keep matched",
        "target": {s: f"{w:.1%}" for s, w in target.weights.items()},
        "held before": {s: pct(h.value) for s, h in sorted(holdings.items())},
        "orders": [f"{o.side} {o.symbol} {pct(o.value(holdings, prices))} ({o.why})" for o in orders],
        "placed": len(placed) if not dry_run else "dry run", "errors": errors,
        "cancelled earlier open orders": cancelled,
        "market": "open: orders fill now" if clock.get("is_open") else f"closed: orders queue for {clock.get('next_open')}",
        **{k: v for k, v in target.info.items()},
        **(lqq3_extras(paper) if name == "lqq3" else {}),
    }


def run(names: list[str], dry_run: bool = False) -> list[dict]:
    papers = {n: account_for(n) for n in names}
    seen: dict[str, str] = {}
    results = []
    for n, paper in papers.items():
        if paper is None:
            vars_ = " / ".join(" + ".join(p) for p in ACCOUNTS[n])
            results.append({"strategy": n, "status": f"skipped: no keys ({vars_})"})
            continue
        try:
            acct_id = paper.account()["id"]
            if acct_id in seen:
                raise RuntimeError(f"same paper account as {seen[acct_id]}: each strategy needs its own account")
            seen[acct_id] = n
            other = owner_of(paper, n)
            if other:
                raise RuntimeError(f"this account holds {other} orders, so it is the {other} account: check which "
                                   f"keys {n} uses ({' / '.join(' + '.join(p) for p in ACCOUNTS[n])})")
            results.append(run_one(n, paper, dry_run))
        except Exception as e:  # one strategy failing must not stop the other
            log.exception("%s failed", n)
            results.append({"strategy": n, "status": f"error: {e}"})
    return results

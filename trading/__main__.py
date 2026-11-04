"""CLI: python -m trading {run,status} [--strategy momentum|qld|all] [--dry-run]."""
from __future__ import annotations

import argparse
import json
import logging

from .run import ACCOUNTS, DESCRIPTION, account_for, run
from .targets import TARGETS


def status(names: list[str]) -> list[dict]:
    out = []
    for n in names:
        paper = account_for(n)
        if paper is None:
            out.append({"strategy": n, "status": "no keys configured"})
            continue
        acct = paper.account()
        eq = float(acct["equity"])
        rows = {p["symbol"]: {"qty": p["qty"], "value": round(float(p["market_value"]), 2),
                              "weight": f"{float(p['market_value']) / eq:.1%}",
                              "gain": f"{float(p['unrealized_plpc']):+.1%}"} for p in paper.positions()}
        try:
            t = TARGETS[n](paper)
            target = {"weights": {s: f"{w:.1%}" for s, w in t.weights.items()}, "decided at close of": str(t.decided.date())}
        except Exception as e:  # show the account even if the data is not ready
            target = f"unavailable: {e}"
        out.append({"strategy": n, "rule": DESCRIPTION[n], "equity": round(eq, 2),
                    "cash": round(float(acct["cash"]), 2), "positions": rows, "target": target,
                    "open orders": [f"{o['side']} {o['symbol']} {o.get('qty') or '$' + str(o.get('notional'))} "
                                    f"({o['status']})" for o in paper.orders("open")]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trading")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd, hlp in (("run", "cancel this tool's open orders and place today's orders"),
                     ("status", "holdings, target and open orders for each account")):
        p = sub.add_parser(cmd, help=hlp)
        p.add_argument("--strategy", choices=["all", *ACCOUNTS], default="all")
        if cmd == "run":
            p.add_argument("--dry-run", action="store_true", help="plan only, place nothing")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    names = list(ACCOUNTS) if a.strategy == "all" else [a.strategy]
    if a.cmd == "status":
        print(json.dumps(status(names), indent=2))
        return 0
    res = run(names, dry_run=a.dry_run)
    print(json.dumps(res, indent=2))
    failed = [r for r in res if str(r.get("status", "")).startswith("error") or r.get("errors")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

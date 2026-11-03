"""CLI: python -m stockdata {update,status,rebuild,validate,snapshot,sync}."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys

from .config import paths


def _logging(name: str) -> None:
    p = paths().ensure()
    logfile = p.logs / f"{name}_{dt.datetime.now():%Y%m%d}.log"
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(logfile)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stockdata")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update", help="incremental update (full rebuild with --full)")
    u.add_argument("--full", action="store_true", help="re-download all price history")
    u.add_argument("--sec", choices=["auto", "always", "never"], default="auto",
                   help="refresh SEC data: auto = if older than 20h (default)")
    u.add_argument("--skip-prices", action="store_true")
    sub.add_parser("status", help="show what data is present and how fresh it is")
    sub.add_parser("rebuild", help="rebuild shares/panel/reports from local files (no downloads)")
    sub.add_parser("validate", help="re-run validation on the current panel")
    sn = sub.add_parser("snapshot", help="pack or restore a data snapshot")
    sn.add_argument("action", choices=["pack", "restore"])
    sn.add_argument("path", nargs="?", help="archive path (pack: output, restore: input file or URL)")
    sn.add_argument("--require-key", action="store_true", help="pack: fail unless STOCKDATA_SNAPSHOT_KEY is set")
    sy = sub.add_parser("sync", help="SessionStart entry point: restore snapshot if needed, then update")
    sy.add_argument("--no-update", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "status":
        from . import data_status
        print(json.dumps(data_status(), indent=2, default=str))
        return 0
    _logging(a.cmd)
    if a.cmd == "update":
        from .pipeline import update
        st = update(full=a.full, sec_refresh=a.sec, skip_prices=a.skip_prices)
        print(json.dumps({k: st[k] for k in ("updated_at", "last_session", "total_s", "timings_s")}, indent=2))
    elif a.cmd == "rebuild":
        from .pipeline import rebuild_derived
        rebuild_derived()
    elif a.cmd == "validate":
        import pandas as pd
        from . import prices, universe, validate
        from .panel import load_panel
        p = paths()
        validate.run(universe.load(), prices.load_store(), pd.read_parquet(p.sec / "share_events.parquet"),
                     load_panel())
    elif a.cmd == "snapshot":
        from . import snapshot
        if a.action == "pack":
            print(snapshot.pack(a.path, require_key=a.require_key))
        else:
            snapshot.restore(a.path or snapshot.default_url())
    elif a.cmd == "sync":
        from . import snapshot
        return snapshot.sync(update=not a.no_update)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

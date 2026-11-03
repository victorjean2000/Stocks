"""Snapshots for fast session start, and the `sync` entry point used by the hook.

A snapshot is a tar of everything except the 3 GB SEC bulk zips (which are
re-downloaded when SEC data is refreshed). If STOCKDATA_SNAPSHOT_KEY is set
it is encrypted with AES-256 (openssl, PBKDF2). The repository is public and
Alpaca market data may not be redistributed, so a published snapshot must be
encrypted; `pack(require_key=True)` enforces that (the CI workflow uses it).

sync():
  1. if no panel exists locally, try to restore the latest snapshot
     (STOCKDATA_SNAPSHOT_URL, default: this repo's `data-snapshot` release);
  2. run an incremental update (or a full build when nothing could be restored).
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import logging
import os
import subprocess
import tarfile
import time
from pathlib import Path

import requests

from .config import REPO_ROOT, paths

log = logging.getLogger(__name__)

RELEASE_TAG = "data-snapshot"
ASSET = "stockdata-snapshot.tar.enc"
KEY_ENV = "STOCKDATA_SNAPSHOT_KEY"
INCLUDE = ["universe", "prices", "sec", "panel", "reports", "state.json", "raw/sec/fsds",
           "raw/sec/company_tickers_exchange.json", "raw/alpaca"]
FRESH_HOURS = 6


def _openssl(args: list[str]) -> None:
    subprocess.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-md", "sha256",
                    "-pass", f"env:{KEY_ENV}", *args], check=True)


def pack(out: str | None = None, require_key: bool = False) -> str:
    root = paths().root
    key = os.environ.get(KEY_ENV)
    if require_key and not key:
        raise RuntimeError(f"{KEY_ENV} is not set; refusing to write an unencrypted snapshot")
    out_path = Path(out) if out else root / ("snapshot.tar.enc" if key else "snapshot.tar")
    tar_path = out_path.with_suffix(".tar.tmp") if key else out_path
    with tarfile.open(tar_path, "w") as tar:
        for rel in INCLUDE:
            src = root / rel
            if src.exists():
                tar.add(src, arcname=rel, filter=lambda ti: None if ti.name.endswith((".zip", ".part")) else ti)
    if key:
        _openssl(["-salt", "-in", str(tar_path), "-out", str(out_path)])
        tar_path.unlink()
    log.info("snapshot written: %s (%.0f MB)", out_path, out_path.stat().st_size / 1e6)
    return str(out_path)


def restore(src: str) -> None:
    """Restore from a local file or an https URL (.tar or encrypted .tar.enc)."""
    root = paths().ensure().root
    local = Path(src)
    tmp_files = []
    if src.startswith("http"):
        local = root / "raw" / Path(src.split("?")[0]).name
        log.info("downloading snapshot %s", src)
        with requests.get(src, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(local, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp_files.append(local)
    tar_path = local
    if local.name.endswith(".enc"):
        if not os.environ.get(KEY_ENV):
            raise RuntimeError(f"snapshot is encrypted but {KEY_ENV} is not set")
        tar_path = local.with_suffix(".tar.tmp")
        _openssl(["-d", "-in", str(local), "-out", str(tar_path)])
        tmp_files.append(tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(root, filter="data")
    for f in tmp_files:
        f.unlink(missing_ok=True)
    log.info("snapshot restored into %s", root)


def default_url() -> str | None:
    if os.environ.get("STOCKDATA_SNAPSHOT_URL"):
        return os.environ["STOCKDATA_SNAPSHOT_URL"]
    try:
        remote = subprocess.run(["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None
    remote = remote.removesuffix(".git")
    if "github.com" not in remote:
        return None
    slug = remote.split("github.com")[-1].lstrip(":/")
    slug = "/".join(slug.split("/")[-2:])
    return f"https://github.com/{slug}/releases/download/{RELEASE_TAG}/{ASSET}"


def _status(**kw) -> None:
    p = paths().ensure().root / "sync_status.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw)
    p.write_text(json.dumps(cur, indent=2, default=str))


def sync(update: bool = True) -> int:
    from . import pipeline
    root = paths().ensure().root
    lock = open(root / ".sync.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("another sync is running; nothing to do")
        return 0
    started = dt.datetime.now(dt.timezone.utc)
    _status(state="running", started_at=started.isoformat(timespec="seconds"), pid=os.getpid(), error=None)
    try:
        panel_file = paths().panel / "panel.parquet"
        restored = False
        if not panel_file.exists():
            url = default_url()
            if url and os.environ.get(KEY_ENV):
                try:
                    restore(url)
                    restored = True
                except Exception as e:
                    log.warning("snapshot restore failed (%s); building from scratch", str(e)[:200])
            else:
                log.info("no snapshot key/url configured; building from scratch (about 25 minutes)")
            _status(restored_snapshot=restored)
        st = pipeline._state()
        fresh = False
        if st.get("updated_at"):
            age = started - dt.datetime.fromisoformat(st["updated_at"])
            fresh = age < dt.timedelta(hours=FRESH_HOURS)
        if update and not (fresh and panel_file.exists()):
            t = time.time()
            pipeline.update(full=False)
            _status(update_seconds=round(time.time() - t))
        _status(state="ready", finished_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        return 0
    except Exception as e:
        log.exception("sync failed")
        _status(state="failed", error=str(e)[:500])
        return 1
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

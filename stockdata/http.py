"""HTTP helpers: a thread-safe rate limiter and a GET with retries.

Secrets live only in session headers. Nothing here logs headers, so keys never
reach logs or tracebacks.
"""
from __future__ import annotations

import logging
import random
import threading
import time

import requests

log = logging.getLogger(__name__)


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status
        self.body = body
        self.url = url


class RateLimiter:
    """Spaces request starts evenly: at most `per_second` requests per second."""

    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0
        self._pause_until = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next, self._pause_until)
            self._next = start + self.interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def pause(self, seconds: float) -> None:
        """Hold every thread back, e.g. after a 429."""
        with self._lock:
            self._pause_until = max(self._pause_until, time.monotonic() + seconds)


RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def get(session: requests.Session, url: str, params: dict | None = None,
        limiter: RateLimiter | None = None, max_retries: int = 8, timeout: float = 90,
        stream: bool = False) -> requests.Response:
    """GET with retries on network errors, 429 and 5xx. Raises HttpError otherwise."""
    last_err = ""
    for attempt in range(max_retries + 1):
        if limiter is not None:
            limiter.wait()
        try:
            r = session.get(url, params=params, timeout=timeout, stream=stream)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            last_err = type(e).__name__
            wait = min(60.0, 2.0 ** attempt) + random.random()
            log.warning("network error %s on %s, retry %d in %.1fs", last_err, url, attempt + 1, wait)
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r
        if r.status_code in RETRY_STATUS and attempt < max_retries:
            wait = _retry_after(r, attempt)
            if r.status_code == 429 and limiter is not None:
                limiter.pause(wait)
            log.warning("HTTP %d on %s, retry %d in %.1fs", r.status_code, url, attempt + 1, wait)
            time.sleep(wait)
            continue
        raise HttpError(r.status_code, r.text, url)
    raise HttpError(-1, f"gave up after retries ({last_err})", url)


def _retry_after(r: requests.Response, attempt: int) -> float:
    ra = r.headers.get("Retry-After")
    if ra:
        try:
            return max(1.0, float(ra))
        except ValueError:
            pass
    reset = r.headers.get("X-Ratelimit-Reset")
    if r.status_code == 429 and reset:
        try:
            return min(65.0, max(1.0, float(reset) - time.time() + 0.5))
        except ValueError:
            pass
    return min(60.0, 2.0 ** attempt) + random.random()

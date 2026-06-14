"""Stack Exchange API rate-limit and error handling."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api.stackexchange.com/2.3"

# Anonymous quota: 300 requests / day; throttle errors return backoff seconds.
DEFAULT_MIN_INTERVAL = 1.0
DEFAULT_MAX_RETRIES = 6
DEFAULT_QUOTA_FLOOR = 5  # pause when remaining quota drops below this


class RateLimitExhausted(Exception):
    """Raised when API quota is exhausted and crawl should stop gracefully."""

    def __init__(self, message: str, quota_remaining: int = 0):
        super().__init__(message)
        self.quota_remaining = quota_remaining


class ApiRequestError(Exception):
    """Non-retryable API error (e.g. invalid parameters)."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class QuotaStatus:
    remaining: Optional[int] = None
    max_quota: Optional[int] = None
    backoff_seconds: int = 0
    last_request_at: float = 0.0
    total_requests: int = 0
    total_throttled: int = 0

    def to_dict(self) -> dict:
        return {
            "remaining": self.remaining,
            "max_quota": self.max_quota,
            "backoff_seconds": self.backoff_seconds,
            "total_requests": self.total_requests,
            "total_throttled": self.total_throttled,
        }


@dataclass
class StackExchangeClient:
    """
    HTTP client with Stack Exchange-specific limit handling.

    Handles:
    - JSON `backoff` field (mandatory wait before next request)
    - HTTP 429 + Retry-After header
    - `quota_remaining` proactive slowdown
    - Exponential backoff on 5xx / network errors
    - Graceful stop via RateLimitExhausted when quota hits 0
    """

    min_interval: float = DEFAULT_MIN_INTERVAL
    max_retries: int = DEFAULT_MAX_RETRIES
    quota_floor: int = DEFAULT_QUOTA_FLOOR
    session: requests.Session = field(default_factory=requests.Session)
    quota: QuotaStatus = field(default_factory=QuotaStatus)

    def _parse_quota(self, data: dict) -> None:
        if "quota_remaining" in data:
            self.quota.remaining = int(data["quota_remaining"])
        if "quota_max" in data:
            self.quota.max_quota = int(data["quota_max"])
        if data.get("backoff"):
            self.quota.backoff_seconds = max(self.quota.backoff_seconds, int(data["backoff"]))

    def _wait_before_request(self) -> None:
        now = time.time()
        elapsed = now - self.quota.last_request_at
        wait = max(self.quota.backoff_seconds, self.min_interval - elapsed)
        if wait > 0:
            logger.debug(f"API throttle wait: {wait:.1f}s (backoff={self.quota.backoff_seconds})")
            time.sleep(wait)
        self.quota.backoff_seconds = 0

    def _wait_retry(self, attempt: int, retry_after: Optional[int] = None) -> None:
        self.quota.total_throttled += 1
        if retry_after and retry_after > 0:
            wait = min(retry_after, 300)
        else:
            wait = min(2 ** attempt * 5, 120)
        logger.warning(f"API rate limited — sleeping {wait}s (attempt {attempt + 1}/{self.max_retries})")
        time.sleep(wait)

    def _check_quota_floor(self) -> None:
        if self.quota.remaining is not None and self.quota.remaining <= 0:
            raise RateLimitExhausted(
                "Stack Exchange daily quota exhausted (quota_remaining=0). "
                "Resume crawl tomorrow or add an API key.",
                quota_remaining=0,
            )
        if self.quota.remaining is not None and self.quota.remaining <= self.quota_floor:
            extra = (self.quota_floor - self.quota.remaining + 1) * 2
            logger.info(
                f"Low quota ({self.quota.remaining} left) — extra sleep {extra}s"
            )
            time.sleep(extra)

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> dict:
        """
        GET request with full rate-limit handling.
        Returns parsed JSON body on success.
        """
        url = f"{API_BASE}/{path.lstrip('/')}"
        params = dict(params or {})

        for attempt in range(self.max_retries):
            self._check_quota_floor()
            self._wait_before_request()

            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                logger.warning(f"Network error on {path}: {exc}")
                self._wait_retry(attempt)
                continue

            self.quota.last_request_at = time.time()
            self.quota.total_requests += 1

            # Parse JSON when possible (even on errors SE often returns JSON)
            data: dict = {}
            try:
                data = resp.json()
                self._parse_quota(data)
            except ValueError:
                pass

            if resp.status_code == 429:
                retry_after = None
                if resp.headers.get("Retry-After"):
                    try:
                        retry_after = int(resp.headers["Retry-After"])
                    except ValueError:
                        pass
                if data.get("backoff"):
                    self.quota.backoff_seconds = int(data["backoff"])
                self._wait_retry(attempt, retry_after=retry_after)
                continue

            if resp.status_code == 400:
                err_msg = data.get("error_message", resp.text[:200])
                raise ApiRequestError(400, f"Bad request for {path}: {err_msg}")

            if resp.status_code >= 500:
                logger.warning(f"Server error {resp.status_code} on {path}")
                self._wait_retry(attempt)
                continue

            if not resp.ok:
                err_msg = data.get("error_message", resp.text[:200])
                raise ApiRequestError(resp.status_code, err_msg)

            return data

        raise RateLimitExhausted(
            f"Max retries ({self.max_retries}) exceeded for {path}. "
            "API may be throttled — resume later.",
            quota_remaining=self.quota.remaining or 0,
        )

    def get_optional(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[dict]:
        """Like get() but returns None on 400 / skip instead of raising."""
        try:
            return self.get(path, params)
        except ApiRequestError as exc:
            if exc.status_code == 400:
                logger.debug(f"Skipping {path}: {exc}")
                return None
            raise

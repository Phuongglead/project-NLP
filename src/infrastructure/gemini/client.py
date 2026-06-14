from __future__ import annotations

import os
import re
import time
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.shared.utils.io_utils import get_logger

logger = get_logger(__name__)

load_dotenv()

_key_index = 0
_client_cache = None
_key_pool: List[str] | None = None
_bad_keys: set[str] = set()


def _parse_key_pool() -> List[str]:
    keys: List[str] = []
    multi = os.getenv("GEMINI_API_KEYS", "")
    if multi.strip():
        for part in re.split(r"[,;\s]+", multi.strip()):
            part = part.strip()
            if part:
                keys.append(part)
    for env_name in ("GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY"):
        single = os.getenv(env_name, "").strip()
        if single and single not in keys:
            keys.insert(0, single)
    return keys


def get_key_pool() -> List[str]:
    global _key_pool
    if _key_pool is None:
        _key_pool = _parse_key_pool()
    return _key_pool


def active_key_count() -> int:
    return len(_active_keys())


def _invalidate_client() -> None:
    global _client_cache
    _client_cache = None


def _active_keys() -> List[str]:
    return [k for k in get_key_pool() if k not in _bad_keys]


def rotate_api_key() -> bool:
    """Advance to next non-blocked API key. Returns False if none left."""
    global _key_index
    pool = _active_keys()
    if len(pool) <= 1:
        return False
    current = pool[_key_index % len(pool)]
    idx = pool.index(current)
    _key_index = (idx + 1) % len(pool)
    _invalidate_client()
    logger.warning("Rotated to Gemini API key %s/%s", _key_index + 1, len(pool))
    return True


def _mark_key_bad(exc: Exception) -> None:
    global _key_index
    pool = _active_keys()
    if not pool:
        return
    key = pool[_key_index % len(pool)]
    _bad_keys.add(key)
    _invalidate_client()
    logger.error("Disabled Gemini API key %s/%s: %s", _key_index + 1, len(pool), exc)


def get_gemini_client():
    global _client_cache, _key_index
    pool = _active_keys()
    if not pool:
        raise EnvironmentError(
            "No usable Gemini API keys left. Check GEMINI_API_KEYS / GOOGLE_GEMINI_API_KEY."
        )
    if _client_cache is not None:
        return _client_cache
    api_key = pool[_key_index % len(pool)]
    _client_cache = genai.Client(api_key=api_key)
    return _client_cache


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "429" in str(exc)
        or "resource_exhausted" in msg
        or "quota" in msg
        or "rate limit" in msg
    )


def _is_auth_denied(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "403" in str(exc) or "permission_denied" in msg or "api key not valid" in msg


def _is_daily_quota(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "perday" in msg or "per_day" in msg or "perdayperproject" in msg


def _wait_seconds_for_rate_limit(exc: Exception, cycle: int, base_delay: float) -> float:
    match = re.search(r"retry in ([\d.]+)s", str(exc), re.IGNORECASE)
    api_delay = float(match.group(1)) if match else base_delay
    api_delay = max(api_delay, 5.0)

    if _is_daily_quota(exc):
        # Free-tier daily caps need long cool-down between retries.
        return max(api_delay, min(3600.0, 300.0 * cycle))

    if cycle <= 1:
        return max(api_delay, 30.0)
    return max(api_delay, min(900.0, 60.0 * cycle))


def generate_with_retry(
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 0,
    base_retry_delay: float = 2.0,
    temperature: float = 0.7,
) -> Optional[str]:
    """
    Call Gemini with retries. max_retries=0 means keep waiting/retrying on 429
    until a response is returned (no fallback).
    """
    global _key_index
    attempt = 0
    rate_limit_cycles = 0
    other_errors = 0

    while True:
        attempt += 1
        if max_retries > 0 and attempt > max_retries:
            logger.warning("Gemini exhausted %s attempts — giving up", max_retries)
            return None
        try:
            client = get_gemini_client()
        except EnvironmentError as exc:
            logger.error("%s", exc)
            return None
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                ),
            )
            if response and getattr(response, "text", None):
                return response.text.strip()
            logger.warning("Gemini returned empty response.")
            return None
        except Exception as exc:
            if _is_auth_denied(exc):
                _mark_key_bad(exc)
                if _active_keys():
                    continue
                logger.exception("All Gemini API keys denied or invalid")
                return None

            if _is_rate_limit(exc):
                active = _active_keys()
                if len(active) > 1:
                    prev_idx = _key_index
                    rotate_api_key()
                    if _key_index != prev_idx:
                        time.sleep(2.0)
                        continue

                rate_limit_cycles += 1
                wait = _wait_seconds_for_rate_limit(exc, rate_limit_cycles, base_retry_delay)
                logger.warning(
                    "Gemini rate limit on all %s keys — waiting %.0fs (cycle %s, attempt %s)",
                    len(active),
                    wait,
                    rate_limit_cycles,
                    attempt,
                )
                time.sleep(wait)
                _key_index = 0
                _invalidate_client()
                continue

            other_errors += 1
            if max_retries > 0 and other_errors >= max_retries:
                logger.exception("Gemini failed after %s non-quota errors", max_retries)
                return None
            delay = base_retry_delay * (2 ** min(other_errors - 1, 6))
            logger.warning(
                "Gemini error (attempt %s): %s — sleeping %.1fs",
                attempt,
                exc,
                delay,
            )
            time.sleep(delay)

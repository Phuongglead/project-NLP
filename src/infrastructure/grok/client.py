"""xAI Grok API client (OpenAI-compatible chat completions)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from dotenv import load_dotenv

from src.shared.utils.io_utils import get_logger, load_config

logger = get_logger(__name__)
load_dotenv()

GROK_API_URL = "https://api.x.ai/v1/chat/completions"


def get_grok_api_key() -> str:
    return os.getenv("GROK_API_KEY", "").strip()


def generate_grok(
    prompt: str,
    model_name: str | None = None,
    temperature: float = 0.7,
    max_retries: int = 3,
    base_retry_delay: float = 4.0,
) -> Optional[str]:
    api_key = get_grok_api_key()
    if not api_key:
        return None

    cfg = load_config().get("grok", {})
    model = model_name or cfg.get("model_name", "grok-2-latest")

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode("utf-8")

    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            GROK_API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if text:
                return text
            logger.warning("Grok returned empty response.")
            return None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if attempt >= max_retries:
                logger.error("Grok failed after %s attempts: %s %s", max_retries, exc.code, body[:200])
                return None
            delay = base_retry_delay * (2 ** (attempt - 1))
            if exc.code == 429:
                logger.warning("Grok rate limit (attempt %s/%s) — sleeping %.1fs", attempt, max_retries, delay)
            else:
                logger.warning("Grok HTTP %s (attempt %s/%s): %s", exc.code, attempt, max_retries, body[:200])
            time.sleep(delay)
        except Exception as exc:
            if attempt >= max_retries:
                logger.exception("Grok failed after %s attempts", max_retries)
                return None
            delay = base_retry_delay * (2 ** (attempt - 1))
            logger.warning("Grok error (attempt %s/%s): %s", attempt, max_retries, exc)
            time.sleep(delay)
    return None

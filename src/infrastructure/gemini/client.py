from __future__ import annotations
import os
import time
from typing import Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.shared.utils.io_utils import get_logger
logger = get_logger(__name__)
_client_cache = None

load_dotenv()

def get_gemini_client():
    global _client_cache
    if _client_cache is not None:
        return _client_cache
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_GEMINI_API_KEY (or GEMINI_API_KEY) environment variable is not set."
        )
    _client_cache = genai.Client(api_key=api_key)
    return _client_cache

def generate_with_retry(
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 3,
    base_retry_delay: float = 2.0,
    temperature: float = 0.7,
) -> Optional[str]:
    client = get_gemini_client()
    for attempt in range(1, max_retries + 1):
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
            error_message = str(exc)
            is_rate_limit = (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
                or "quota" in error_message.lower()
            )
            if attempt >= max_retries:
                logger.exception(
                    "Gemini failed after %s attempts",
                    max_retries,
                )
                return None
            delay = base_retry_delay * (2 ** (attempt - 1))
            if is_rate_limit:
                logger.warning("Gemini rate limit hit " "(attempt %s/%s). " "Sleeping %.1fs",
                    attempt, max_retries, delay,)
            else:
                logger.warning("Gemini error " "(attempt %s/%s): %s", attempt, max_retries, error_message,)
            time.sleep(delay)
    return None
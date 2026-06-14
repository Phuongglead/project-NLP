"""Unified LLM generation: Gemini primary, Grok fallback."""

from __future__ import annotations

from typing import Optional, Tuple

from src.infrastructure.gemini.client import generate_with_retry, get_key_pool, active_key_count
from src.infrastructure.grok.client import generate_grok, get_grok_api_key
from src.shared.exceptions import LlmKeyUnavailableError
from src.shared.utils.io_utils import get_logger, load_config

logger = get_logger(__name__)


def _keys_exhausted() -> bool:
    return len(get_key_pool()) > 0 and active_key_count() == 0


def generate_with_fallback(
    prompt: str,
    model_name: str,
    temperature: float = 0.7,
    gemini_max_retries: int | None = None,
    base_retry_delay: float = 4.0,
) -> Tuple[Optional[str], str]:
    """
    Try Gemini first, then Grok if configured.
    Returns (text, provider) where provider is 'gemini', 'grok', or ''.
    Raises LlmKeyUnavailableError when all API keys are invalid or missing.
    """
    cfg = load_config()
    gen_cfg = cfg.get("generator", {})
    grok_cfg = cfg.get("grok", {})

    if not get_key_pool() and not get_grok_api_key():
        raise LlmKeyUnavailableError("No Gemini or Grok API keys configured.")

    if gemini_max_retries is None:
        if get_grok_api_key() and grok_cfg.get("fallback_enabled", True):
            gemini_max_retries = gen_cfg.get("gemini_max_retries_before_grok", 24)
        else:
            gemini_max_retries = gen_cfg.get("max_retries", 0)

    text = generate_with_retry(
        prompt=prompt,
        model_name=model_name,
        temperature=temperature,
        max_retries=gemini_max_retries,
        base_retry_delay=base_retry_delay,
    )
    if text:
        return text, "gemini"

    if _keys_exhausted():
        raise LlmKeyUnavailableError("All Gemini API keys were rejected.")

    if not get_grok_api_key() or not grok_cfg.get("fallback_enabled", True):
        return None, ""

    logger.warning("Gemini unavailable — falling back to Grok API.")
    text = generate_grok(
        prompt=prompt,
        model_name=grok_cfg.get("model_name"),
        temperature=temperature,
        max_retries=grok_cfg.get("max_retries", 3),
        base_retry_delay=base_retry_delay,
    )
    if text:
        return text, "grok"

    return None, ""

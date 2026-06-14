"""LLM layer exceptions."""

from __future__ import annotations


class LlmKeyUnavailableError(RuntimeError):
    """Raised when Gemini/Grok API keys are missing or rejected."""

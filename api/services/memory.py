from __future__ import annotations

from typing import Any, Dict, Optional


class MemoryStore:
    """In-memory CV session store for uploaded files."""

    def __init__(self) -> None:
        self._cv_store: Dict[str, Dict[str, Any]] = {}

    def store_cv(self, session_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._cv_store[session_id] = {"text": text, "metadata": metadata or {}}

    def get_cv(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._cv_store.get(session_id)


_GLOBAL_MEMORY: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    global _GLOBAL_MEMORY
    if _GLOBAL_MEMORY is None:
        _GLOBAL_MEMORY = MemoryStore()
    return _GLOBAL_MEMORY

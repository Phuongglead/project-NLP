"""ESCO-based CV keyword expansion (CV side only)."""

from __future__ import annotations

import json
import os
import re
from typing import List, Set

from src.shared.utils.io_utils import get_logger, load_config

logger = get_logger(__name__)

_esco_map: dict | None = None


def _normalize_key(term: str) -> str:
    return re.sub(r"\s+", " ", term.lower().strip())


def _load_esco_map() -> dict:
    global _esco_map
    if _esco_map is not None:
        return _esco_map
    cfg = load_config()["rag"]
    path = cfg.get("esco_skill_map_path", "data/esco_skill_map.json")
    if not os.path.isabs(path):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        path = os.path.join(root, path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            _esco_map = json.load(f)
    except FileNotFoundError:
        logger.warning(f"ESCO skill map not found at {path}; expansion disabled.")
        _esco_map = {}
    return _esco_map


def expand_cv_keywords(cv_skills: List[str], max_expanded: int = 30) -> List[str]:
    """
    Expand CV-side skill keywords using offline ESCO map (1-hop).
    Does NOT expand corpus record keywords.
    """
    esco = _load_esco_map()
    expanded: List[str] = []
    seen: Set[str] = set()

    def _add(term: str) -> None:
        key = _normalize_key(term)
        if key and key not in seen:
            seen.add(key)
            expanded.append(term.strip())

    for skill in cv_skills:
        _add(skill)
        entry = esco.get(_normalize_key(skill))
        if not entry:
            # Partial match: e.g. "CI/CD" -> "cicd"
            compact = _normalize_key(skill).replace("/", "").replace("-", "")
            entry = esco.get(compact)
        if entry:
            for group in ("broader", "narrower", "related"):
                for related in entry.get(group, []):
                    _add(related)
                    if len(expanded) >= max_expanded:
                        return expanded

    return expanded[:max_expanded]

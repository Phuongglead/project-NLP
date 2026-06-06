"""
shared/utils/io_utils.py
Utilities for JSONL file I/O, config loading, logging setup, and formatting.
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import yaml


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ── Config loading ────────────────────────────────────────────────────────────

_config_cache: Dict[str, Any] = {}


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    if config_path in _config_cache:
        return _config_cache[config_path]
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _config_cache[config_path] = cfg
    return cfg


def load_prompts(prompts_path: str = "config/prompts.yaml") -> Dict[str, Any]:
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── JSONL I/O ─────────────────────────────────────────────────────────────────

def write_jsonl(records: List[Dict], path: str, mode: str = "w") -> None:
    """Write a list of dicts to a JSONL file (one JSON object per line)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: Dict, path: str) -> None:
    """Append a single dict as a new line in a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Generator[Dict, None, None]:
    """Lazily yield dicts from a JSONL file, one per line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str) -> List[Dict]:
    """Load all records from a JSONL file into a list."""
    return list(read_jsonl(path))


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_skill_entities_for_prompt(skill_entities: List[Dict]) -> str:
    """
    Format skill entity dicts into a human-readable prompt string.
    Example output:
        - Kubernetes [SKILL]
        - microservices architecture [KNOWLEDGE]
    """
    if not skill_entities:
        return "(no skills extracted)"
    lines = []
    for e in skill_entities:
        entity = e.get("entity", "")
        etype = e.get("type", "SKILL")
        lines.append(f"  - {entity} [{etype}]")
    return "\n".join(lines)


def extract_skill_names(skill_entities: List[Dict]) -> List[str]:
    """Return just the entity name strings from a list of skill entity dicts."""
    return [e["entity"] for e in skill_entities if "entity" in e]


def format_skill_names_for_prompt(skill_entities: List[Dict]) -> str:
    names = extract_skill_names(skill_entities)
    if not names:
        return "(none)"
    return ", ".join(names)


# ── File helpers ──────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def file_exists(path: str) -> bool:
    return Path(path).exists()


def count_jsonl_lines(path: str) -> int:
    """Fast line count for JSONL files."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count

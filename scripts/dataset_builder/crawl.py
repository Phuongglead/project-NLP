"""Stack Exchange API crawler for technical Q&A."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from scripts.dataset_builder.api_limits import (
    ApiRequestError,
    RateLimitExhausted,
    StackExchangeClient,
)

DEFAULT_TAGS = [
    "python", "fastapi", "kubernetes", "docker", "machine-learning",
    "sql", "javascript", "reactjs", "devops", "microservices",
]

# Single-tag rotations avoid 400 errors from invalid multi-tag combos.
TAG_ROTATIONS = [
    ["python"],
    ["kubernetes"],
    ["docker"],
    ["javascript"],
    ["reactjs"],
    ["sql"],
    ["postgresql"],
    ["machine-learning"],
    ["pytorch"],
    ["fastapi"],
    ["django"],
    ["spring-boot"],
    ["golang"],
    ["aws"],
    ["microservices"],
    ["devops"],
    ["node.js"],
    ["mongodb"],
    ["tensorflow"],
    ["security"],
]

# Shared client — persists quota state across calls in one crawl session.
_client: Optional[StackExchangeClient] = None


def get_client() -> StackExchangeClient:
    global _client
    if _client is None:
        min_interval = float(os.environ.get("SO_CRAWL_MIN_INTERVAL", "1.0"))
        max_retries = 6
        quota_floor = 5
        try:
            import sys
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if root not in sys.path:
                sys.path.insert(0, root)
            from src.shared.utils.io_utils import load_config
            cfg = load_config().get("crawl", {})
            min_interval = float(cfg.get("min_interval_seconds", min_interval))
            max_retries = int(cfg.get("max_retries", max_retries))
            quota_floor = int(cfg.get("quota_floor", quota_floor))
        except Exception:
            pass
        _client = StackExchangeClient(
            min_interval=min_interval,
            max_retries=max_retries,
            quota_floor=quota_floor,
        )
    return _client


def reset_client() -> None:
    global _client
    _client = None


def fetch_questions_page(
    page: int,
    tags: List[str],
    page_size: int = 100,
    site: str = "stackoverflow",
) -> tuple[List[Dict], bool]:
    """
    Fetch one page of questions. Returns (items, has_more).
    Raises RateLimitExhausted when quota is gone.
    """
    tag_str = ";".join(tags)
    params = {
        "order": "desc",
        "sort": "votes",
        "tagged": tag_str,
        "site": site,
        "pagesize": page_size,
        "page": page,
        "filter": "withbody",
    }
    client = get_client()
    try:
        data = client.get("questions", params)
    except ApiRequestError:
        return [], False

    items = data.get("items", [])
    has_more = bool(data.get("has_more"))
    return items, has_more


def fetch_questions(
    tags: List[str] = None,
    pages: int = 1,
    page_size: int = 30,
    site: str = "stackoverflow",
) -> List[Dict]:
    """Fetch questions with accepted answers from Stack Exchange API."""
    tags = tags or DEFAULT_TAGS
    all_items: List[Dict] = []

    for page in range(1, pages + 1):
        items, has_more = fetch_questions_page(
            page, tags, page_size=page_size, site=site
        )
        all_items.extend(items)
        if not has_more:
            break

    return all_items


def fetch_accepted_answer(
    question: Dict,
    site: str = "stackoverflow",
) -> Optional[Dict]:
    """Fetch the accepted answer for a question. Returns None on skip/error."""
    if not question.get("accepted_answer_id"):
        return None

    answer_id = question["accepted_answer_id"]
    params = {"site": site, "filter": "withbody"}
    client = get_client()

    try:
        data = client.get_optional(f"answers/{answer_id}", params)
    except RateLimitExhausted:
        raise
    except ApiRequestError:
        return None

    if not data:
        return None
    items = data.get("items", [])
    return items[0] if items else None


def quota_snapshot() -> dict:
    """Current quota stats for logging / crawl state."""
    return get_client().quota.to_dict()

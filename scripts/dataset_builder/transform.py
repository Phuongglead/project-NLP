"""Transform Stack Overflow Q/A pairs into knowledge corpus records."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.shared.corpus.schema import (
    build_index_text,
    dedup_hash,
    extract_keywords_from_text,
    normalize_topic,
)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _infer_skill(tags: List[str]) -> str:
    if not tags:
        return "General"
    return normalize_topic(tags[0].replace("-", " "))


def _infer_topic(question_title: str, tags: List[str]) -> str:
    if len(tags) > 1:
        return normalize_topic(tags[1].replace("-", " "))
    words = question_title.split()[:4]
    return " ".join(words) if words else "General"


def so_to_knowledge_record(
    question: Dict,
    answer: Dict,
    ner_fn=None,
) -> Optional[Dict]:
    """Map Stack Overflow question + accepted answer to extended schema."""
    q_body = _strip_html(question.get("body", ""))
    q_title = question.get("title", "")
    a_body = _strip_html(answer.get("body", ""))
    if not a_body or len(a_body) < 80:
        return None

    tags = question.get("tags", [])
    skill = _infer_skill(tags)
    topic = _infer_topic(q_title, tags)
    definition = a_body[:500]
    details = a_body[500:1500] if len(a_body) > 500 else ""
    reference_answer = f"{definition} {details}".strip()

    keywords = extract_keywords_from_text(reference_answer, ner_fn)
    for extra in (skill, topic):
        if extra.lower() not in {k.lower() for k in keywords}:
            keywords.insert(0, extra)

    record = {
        "id": str(uuid.uuid4()),
        "skill": skill,
        "topic": topic,
        "sub_topics": [q_title[:120]] if q_title else [],
        "reference_answer": reference_answer,
        "knowledge": {
            "definition": definition,
            "details": details,
            "stack_overflow_example": {
                "question": q_title,
                "answer": a_body[:2000],
            },
        },
        "keywords": keywords,
        "question": None,
        "interview_usage": {
            "why_important": f"Common interview topic for {skill}.",
            "question_patterns": [
                f"How would you explain {topic} in the context of {skill}?",
                f"What are practical trade-offs when applying {topic}?",
            ],
        },
        "source": {
            "dataset": "StackOverflow",
            "tags": tags,
            "url": question.get("link", ""),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    record["index_text"] = build_index_text(record)
    record["_dedup_hash"] = dedup_hash(record)
    return record

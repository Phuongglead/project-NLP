"""Knowledge corpus schema helpers and record normalization."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_topic(topic: str) -> str:
    return topic.replace("_", " ").strip().title()


def build_index_text(record: Dict[str, Any]) -> str:
    """Build FAISS embedding text from a knowledge record."""
    skill = record.get("skill") or normalize_topic(record.get("topic", ""))
    topic = record.get("topic", "")
    if isinstance(topic, str) and topic == skill:
        topic_label = normalize_topic(topic)
    else:
        topic_label = str(topic)
    keywords = record.get("keywords") or []
    kw_str = ", ".join(keywords) if keywords else ""
    answer = record.get("reference_answer", "")
    sub_topics = record.get("sub_topics") or []
    sub_str = ", ".join(sub_topics) if sub_topics else ""
    parts = [f"skill: {skill}", f"topic: {topic_label}"]
    if kw_str:
        parts.append(f"keywords: {kw_str}")
    if sub_str:
        parts.append(f"sub_topics: {sub_str}")
    parts.append(f"answer: {answer}")
    return " | ".join(parts)


def record_keywords_set(record: Dict[str, Any]) -> set:
    """Corpus-side keyword set (not ESCO-expanded)."""
    keys = set()
    for k in record.get("keywords") or []:
        if k:
            keys.add(k.lower().strip())
    for field in ("skill", "topic"):
        val = record.get(field)
        if val:
            keys.add(str(val).lower().strip())
    for st in record.get("sub_topics") or []:
        if st:
            keys.add(str(st).lower().strip())
    return keys


def dedup_hash(record: Dict[str, Any]) -> str:
    """Stable hash for deduplication."""
    url = (record.get("source") or {}).get("url", "")
    if url:
        return hashlib.sha256(url.encode()).hexdigest()
    text = record.get("reference_answer", "").strip().lower()
    return hashlib.sha256(text.encode()).hexdigest()


def extract_keywords_from_text(text: str, ner_fn=None) -> List[str]:
    """Extract keywords via NER if available, else heuristic tokenization."""
    keywords: List[str] = []
    if ner_fn and text:
        try:
            entities = ner_fn(text)
            keywords = list({e.get("entity", e.entity if hasattr(e, "entity") else "") for e in entities})
            keywords = [k for k in keywords if k]
        except Exception:
            keywords = []
    if not keywords and text:
        # Fallback: capitalized phrases and known tech tokens
        keywords = re.findall(r"\b[A-Z][a-zA-Z0-9+#./-]{1,}\b", text)
        keywords = list(dict.fromkeys(keywords))[:15]
    return keywords


def legacy_to_knowledge_record(
    legacy: Dict[str, Any],
    ner_fn=None,
    source_dataset: str = "legacy",
) -> Dict[str, Any]:
    """Convert old reference_answers.jsonl record to extended schema."""
    topic = legacy.get("topic", "general")
    skill = normalize_topic(topic)
    reference_answer = legacy.get("reference_answer", "")
    keywords = extract_keywords_from_text(reference_answer, ner_fn)
    for extra in (skill, topic):
        if extra and extra.lower() not in {k.lower() for k in keywords}:
            keywords.insert(0, extra)

    record: Dict[str, Any] = {
        "id": legacy.get("id") or dedup_hash({"reference_answer": reference_answer}),
        "skill": skill,
        "topic": normalize_topic(topic) if isinstance(topic, str) else str(topic),
        "sub_topics": [],
        "reference_answer": reference_answer,
        "knowledge": {
            "definition": reference_answer,
            "details": "",
            "stack_overflow_example": {"question": "", "answer": ""},
        },
        "keywords": keywords,
        "question": legacy.get("question"),
        "interview_usage": {
            "why_important": f"Core knowledge for {skill} technical interviews.",
            "question_patterns": [
                f"How would you explain {skill} in a production system?",
                f"What are trade-offs when using {skill}?",
            ],
        },
        "source": {
            "dataset": source_dataset,
            "tags": [topic] if topic else [],
            "url": legacy.get("source", {}).get("url", "") if isinstance(legacy.get("source"), dict) else "",
        },
        "created_at": legacy.get("created_at") or _now_iso(),
        "updated_at": _now_iso(),
    }
    record["index_text"] = build_index_text(record)
    return record


def normalize_knowledge_record(record: Dict[str, Any], ner_fn=None) -> Dict[str, Any]:
    """Ensure record has all required fields."""
    if not record.get("reference_answer") and record.get("knowledge"):
        k = record["knowledge"]
        definition = k.get("definition", "")
        details = k.get("details", "")
        record["reference_answer"] = f"{definition} {details}".strip()

    if not record.get("skill") and record.get("topic"):
        record["skill"] = normalize_topic(record["topic"])

    if not record.get("keywords"):
        record["keywords"] = extract_keywords_from_text(record.get("reference_answer", ""), ner_fn)

    if "question" not in record:
        record["question"] = None

    if not record.get("index_text"):
        record["index_text"] = build_index_text(record)

    if not record.get("id"):
        record["id"] = dedup_hash(record)

    record.setdefault("created_at", _now_iso())
    record["updated_at"] = _now_iso()
    return record


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        pass
    return records


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

"""Helpers to build evaluation JSONL records for reports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from src.shared.contracts.schemas import RetrievalHit, SkillEntity


def hit_to_eval_dict(hit: RetrievalHit) -> dict:
    rec = hit.record
    return {
        "corpus_id": hit.corpus_id,
        "match_score": hit.match_score,
        "semantic_distance": hit.semantic_distance,
        "skill": rec.get("skill"),
        "topic": rec.get("topic"),
        "reference_answer": rec.get("reference_answer", ""),
        "cached_question": rec.get("question"),
        "question_source": hit.question_source,
    }


def build_batch_eval_record(
    eval_id: str,
    cv_id: str,
    category: str,
    cv_text: str,
    job_description: str,
    skills: List[SkillEntity],
    rag_hits: List[RetrievalHit],
    generated_question: str,
    question_source: str,
    selected_corpus_id: str,
    evaluation: dict,
) -> dict:
    hit_dicts = [hit_to_eval_dict(h) for h in rag_hits]
    selected = next(
        (h for h in hit_dicts if h["corpus_id"] == selected_corpus_id),
        hit_dicts[0] if hit_dicts else {},
    )
    return {
        "eval_id": eval_id,
        "cv_id": cv_id,
        "category": category,
        "cv_text": cv_text,
        "job_description": job_description,
        "skills": [s.to_dict() for s in skills],
        "rag_hits": hit_dicts,
        "selected_hit": selected,
        "generated_question": generated_question,
        "question_source": question_source,
        "corpus_id": selected_corpus_id,
        "evaluation": evaluation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

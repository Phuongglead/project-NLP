"""Hybrid scoring and MMR diversity selection for RAG retrieval."""

from __future__ import annotations

from typing import Dict, List, Set

from src.shared.corpus.schema import record_keywords_set
from src.shared.contracts.schemas import RetrievalHit
from src.shared.utils.io_utils import load_config


def _normalize_kw_set(keywords: List[str]) -> Set[str]:
    return {k.lower().strip() for k in keywords if k}


def keyword_overlap_score(query_keywords: List[str], record: Dict) -> float:
    query_set = _normalize_kw_set(query_keywords)
    if not query_set:
        return 0.0
    record_set = record_keywords_set(record)
    overlap = len(query_set & record_set)
    return overlap / len(query_set)


def semantic_score_from_distance(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))


def hybrid_score(distance: float, query_keywords: List[str], record: Dict) -> float:
    cfg = load_config()["rag"]
    w_sem = cfg.get("hybrid_semantic_weight", 0.6)
    w_kw = cfg.get("hybrid_keyword_weight", 0.4)
    sem = semantic_score_from_distance(distance)
    kw = keyword_overlap_score(query_keywords, record)
    return w_sem * sem + w_kw * kw


def jaccard_similarity(record_a: Dict, record_b: Dict) -> float:
    set_a = record_keywords_set(record_a)
    set_b = record_keywords_set(record_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def topic_key(record: Dict) -> str:
    skill = (record.get("skill") or "").lower().strip()
    topic = (record.get("topic") or "").lower().strip()
    return f"{skill}::{topic}"


def mmr_select(
    candidates: List[Dict],
    top_k: int,
    exclude_ids: List[str] | None = None,
) -> List[RetrievalHit]:
    """
    Maximal Marginal Relevance selection on hybrid-scored candidates.
    Each candidate dict must have: record, distance, hybrid_score, idx.
    """
    cfg = load_config()["rag"]
    lam = cfg.get("mmr_lambda", 0.7)
    exclude = set(exclude_ids or [])
    pool = [c for c in candidates if c["record"].get("id") not in exclude]
    selected: List[RetrievalHit] = []
    selected_records: List[Dict] = []
    used_topics: Set[str] = set()

    while pool and len(selected) < top_k:
        best = None
        best_mmr = -1.0
        for cand in pool:
            tk = topic_key(cand["record"])
            if tk in used_topics:
                continue
            if selected_records:
                max_sim = max(
                    jaccard_similarity(cand["record"], s) for s in selected_records
                )
            else:
                max_sim = 0.0
            mmr = lam * cand["hybrid_score"] - (1.0 - lam) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best = cand

        if best is None:
            # Relax topic uniqueness for remaining slots
            for cand in pool:
                if selected_records:
                    max_sim = max(
                        jaccard_similarity(cand["record"], s) for s in selected_records
                    )
                else:
                    max_sim = 0.0
                mmr = lam * cand["hybrid_score"] - (1.0 - lam) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best = cand
            if best is None:
                break

        record = best["record"]
        question = record.get("question")
        hit = RetrievalHit(
            corpus_id=record.get("id", ""),
            record=record,
            match_score=round(best["hybrid_score"], 4),
            semantic_distance=round(best["distance"], 4),
            question_source="cached" if question else "pending",
        )
        selected.append(hit)
        selected_records.append(record)
        used_topics.add(topic_key(record))
        pool = [c for c in pool if c["record"].get("id") != record.get("id")]

    return selected

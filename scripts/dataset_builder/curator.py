#!/usr/bin/env python3
"""
Promote highly-rated user feedback into corpus question column.
Runs ALCE gate only (no NLI, no SHAP).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.core.xai_evaluator.xai_module import compute_alce_scores
from src.shared.contracts.schemas import SkillEntity
from src.shared.corpus.schema import load_jsonl, write_jsonl
from src.shared.utils.io_utils import get_logger, load_config

logger = get_logger("curator")


def _load_feedback(path: str) -> list:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _best_feedback_per_corpus(feedback_records: list, min_rating: int) -> dict:
    """Return corpus_id -> best feedback entry with rating >= min_rating."""
    best = {}
    for fb in feedback_records:
        if fb.get("rating", 0) < min_rating:
            continue
        cid = fb.get("corpus_id")
        if not cid:
            continue
        prev = best.get(cid)
        if not prev or fb["rating"] > prev["rating"]:
            best[cid] = fb
    return best


def run_curator(
    corpus_path: str,
    feedback_path: str,
    min_rating: int = 4,
    min_citation_precision: float = 0.5,
    cv_skills: list = None,
) -> int:
    """
    Promote feedback questions into corpus records where question is null.
    Uses ALCE with provided cv_skills or empty list (precision on question text only).
    """
    corpus = load_jsonl(corpus_path)
    if not corpus:
        logger.warning(f"No corpus at {corpus_path}")
        return 0

    by_id = {r["id"]: r for r in corpus}
    feedback = _load_feedback(feedback_path)
    candidates = _best_feedback_per_corpus(feedback, min_rating)

    skills = [SkillEntity(entity=s, type="SKILL", start=0, end=1) for s in (cv_skills or [])]
    promoted = 0

    for corpus_id, fb in candidates.items():
        record = by_id.get(corpus_id)
        if not record:
            continue
        if record.get("question"):
            continue

        question = fb.get("generated_question", "").strip()
        if not question:
            continue

        if skills:
            alce = compute_alce_scores(question, skills)
            if alce["citation_precision"] < min_citation_precision:
                logger.info(f"Skip {corpus_id}: ALCE precision {alce['citation_precision']:.2f}")
                continue

        record["question"] = question
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        record["feedback_source"] = fb.get("feedback_id")
        promoted += 1
        logger.info(f"Promoted question for {corpus_id} (rating={fb['rating']})")

    if promoted:
        write_jsonl(corpus, corpus_path)
    logger.info(f"Curator promoted {promoted} questions.")
    return promoted


def main():
    parser = argparse.ArgumentParser(description="Promote feedback to corpus question column")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--feedback", default=None)
    args = parser.parse_args()

    cfg = load_config()
    rag = cfg["rag"]
    xai = cfg["xai"]
    corpus = args.corpus or rag.get("staging_corpus_path", "data/knowledge_corpus.staging.jsonl")
    feedback = args.feedback or rag.get("feedback_path", "data/feedback.jsonl")
    if not os.path.isabs(corpus):
        corpus = os.path.join(ROOT, corpus)
    if not os.path.isabs(feedback):
        feedback = os.path.join(ROOT, feedback)

    n = run_curator(
        corpus,
        feedback,
        min_rating=xai.get("curator_min_rating", 4),
        min_citation_precision=xai.get("curator_min_citation_precision", 0.5),
    )
    print(f"Promoted {n} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

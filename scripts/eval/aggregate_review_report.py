#!/usr/bin/env python3
"""Aggregate REVIEW_MODE human ratings into report summary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from statistics import mean, median

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.shared.corpus.schema import load_jsonl
from src.shared.utils.io_utils import load_config


def aggregate_review(records: list) -> dict:
    ratings = [r for r in records if r.get("event") == "rating"]
    if not ratings:
        return {"n_ratings": 0, "message": "No rating events in review log."}

    scores = [r["rating"] for r in ratings]
    by_session: dict = defaultdict(list)
    for r in ratings:
        by_session[r.get("cv_session_id", "unknown")].append(r["rating"])

    per_cv = {
        sid: {
            "n_ratings": len(vals),
            "mean_rating": round(mean(vals), 2),
            "ratings": vals,
        }
        for sid, vals in by_session.items()
    }

    # Per-question detail for LaTeX
    per_question = []
    for r in ratings:
        per_question.append({
            "cv_session_id": r.get("cv_session_id"),
            "holdout_cv_id": r.get("holdout_cv_id"),
            "corpus_id": r.get("corpus_id"),
            "question": r.get("generated_question", "")[:200],
            "rating": r.get("rating"),
        })

    dist = Counter(scores)
    return {
        "n_ratings": len(scores),
        "n_sessions": len(by_session),
        "mean_rating": round(mean(scores), 2),
        "median_rating": round(median(scores), 2),
        "pct_rating_ge_4": round(100 * sum(1 for s in scores if s >= 4) / len(scores), 2),
        "rating_distribution": dict(sorted(dist.items())),
        "per_cv_session": per_cv,
        "per_question": per_question,
        "expected_protocol": "3 CVs x 5 questions = 15 ratings",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config().get("evaluation", {})
    input_path = args.input or cfg.get("review_output", "data/review_eval.jsonl")
    output_path = args.output or cfg.get("review_summary", "outputs/eval/review_summary.json")
    if not os.path.isabs(input_path):
        input_path = os.path.join(ROOT, input_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(ROOT, output_path)

    records = load_jsonl(input_path)
    summary = aggregate_review(records)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Review summary -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

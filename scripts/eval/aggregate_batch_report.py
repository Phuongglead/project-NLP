#!/usr/bin/env python3
"""Aggregate batch eval JSONL into report-ready summary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from statistics import mean, median, stdev

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.shared.corpus.schema import load_jsonl
from src.shared.utils.io_utils import load_config


def aggregate(records: list, cv_threshold: float = 0.40) -> dict:
    if not records:
        return {"n_evaluated": 0}

    precisions = [r["evaluation"]["citation_precision"] for r in records]
    recalls = [r["evaluation"]["citation_recall"] for r in records]
    shap_cv = [r["evaluation"]["shap_cv_ratio"] for r in records if r["evaluation"].get("shap_cv_ratio", 0) > 0]
    shap_ans = [r["evaluation"].get("shap_answer_ratio", 0) for r in records if r["evaluation"].get("shap_answer_ratio", 0) > 0]
    match_scores = [r["selected_hit"].get("match_score", 0) for r in records if r.get("selected_hit")]
    n_skills = [len(r.get("skills", [])) for r in records]

    cached = sum(1 for r in records if r.get("question_source") == "cached")
    generated = len(records) - cached

    topics = Counter(r["selected_hit"].get("topic", "?") for r in records if r.get("selected_hit"))
    skills_cited = Counter()
    for r in records:
        for s in r.get("evaluation", {}).get("cited_skills", []):
            skills_cited[s] += 1
        # derive from question vs skills
        q = r.get("generated_question", "").lower()
        for sk in r.get("skills", []):
            if sk.get("entity", "").lower() in q:
                skills_cited[sk["entity"]] += 1

    def _stats(vals):
        if not vals:
            return {"mean": 0, "median": 0, "std": 0}
        return {
            "mean": round(mean(vals), 4),
            "median": round(median(vals), 4),
            "std": round(stdev(vals), 4) if len(vals) > 1 else 0.0,
        }

    return {
        "n_evaluated": len(records),
        "n_skills_avg": round(mean(n_skills), 2) if n_skills else 0,
        "alce": {
            "precision": _stats(precisions),
            "recall": _stats(recalls),
            "pct_precision_ge_0_5": round(100 * sum(1 for p in precisions if p >= 0.5) / len(precisions), 2),
            "pct_recall_ge_0_5": round(100 * sum(1 for r in recalls if r >= 0.5) / len(recalls), 2),
        },
        "shap": {
            "cv_ratio": _stats(shap_cv),
            "answer_ratio": _stats(shap_ans),
            "pct_cv_above_threshold": round(
                100 * sum(1 for v in shap_cv if v >= cv_threshold) / len(shap_cv), 2
            ) if shap_cv else 0,
            "pct_answer_above_threshold": round(
                100 * sum(1 for v in shap_ans if v >= cv_threshold) / len(shap_ans), 2
            ) if shap_ans else 0,
            "cv_contribution_threshold": cv_threshold,
        },
        "rag": {
            "match_score": _stats(match_scores),
            "pct_cached_questions": round(100 * cached / len(records), 2),
            "pct_generated_questions": round(100 * generated / len(records), 2),
        },
        "top_topics_retrieved": topics.most_common(10),
        "top_skills_cited_in_questions": skills_cited.most_common(15),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config()
    eval_cfg = cfg.get("evaluation", {})
    xai_cfg = cfg.get("xai", {})
    input_path = args.input or eval_cfg.get("batch_output", "outputs/eval/batch_records.jsonl")
    output_path = args.output or eval_cfg.get("batch_summary", "outputs/eval/batch_summary.json")
    if not os.path.isabs(input_path):
        input_path = os.path.join(ROOT, input_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(ROOT, output_path)

    records = load_jsonl(input_path)
    summary = aggregate(
        records,
        cv_threshold=xai_cfg.get("cv_contribution_threshold", 0.40),
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Summary written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

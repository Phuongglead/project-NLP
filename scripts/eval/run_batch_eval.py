#!/usr/bin/env python3
"""Batch ALCE + SHAP evaluation on Kaggle IT resumes."""

from __future__ import annotations

import argparse
import os
import sys

from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.core.NER.ner_module import skill_extract
from src.core.rag_retriever.rag_module import retrieve_candidates
from src.core.xai_evaluator.xai_module import evaluate_question
from src.evaluation.record_builder import build_batch_eval_record
from src.pipeline.runner import run_pipeline_multi
from src.shared.contracts.schemas import GeneratorOutput, SkillEntity
from src.shared.corpus.schema import load_jsonl
from src.shared.utils.io_utils import append_jsonl, get_logger, load_config

logger = get_logger("run_batch_eval")


def run_batch_eval(
    input_path: str,
    output_path: str,
    n: int = None,
    shap_nsamples: int = None,
    rag_top_k: int = 5,
    skip_existing: bool = True,
) -> int:
    cfg = load_config()
    eval_cfg = cfg.get("evaluation", {})
    shap_n = shap_nsamples or eval_cfg.get("shap_nsamples_batch", 50)
    rag_k = rag_top_k or eval_cfg.get("rag_top_k", 5)

    samples = load_jsonl(input_path)
    if n:
        samples = samples[:n]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    done_ids = set()
    if skip_existing and os.path.exists(output_path):
        for rec in load_jsonl(output_path):
            done_ids.add(rec.get("cv_id"))

    pending = [
        (i, s) for i, s in enumerate(samples)
        if s.get("cv_id", f"cv_{i:04d}") not in done_ids
    ]
    processed = 0
    pbar = tqdm(
        pending,
        total=len(pending),
        desc="Batch eval",
        unit="cv",
        dynamic_ncols=True,
    )
    for i, sample in pbar:
        cv_id = sample.get("cv_id", f"cv_{i:04d}")
        cv_text = sample.get("cv_text", "")
        job_description = sample.get(
            "job_description",
            eval_cfg.get("default_job_description", "Information-Technology role."),
        )
        category = sample.get("category", "Information-Technology")
        eval_id = f"it_{i:04d}"

        pbar.set_postfix(cv=cv_id, done=processed, refresh=False)
        logger.info(f"[{i+1}/{len(samples)}] Evaluating {cv_id}...")

        raw_skills = skill_extract(cv_text)
        skills = [SkillEntity.from_dict(s) if isinstance(s, dict) else s for s in raw_skills]
        rag_hits = retrieve_candidates(cv_text, job_description, skills, top_k=rag_k)

        results = run_pipeline_multi(
            cv_text=cv_text,
            job_description=job_description,
            num_questions=1,
            force_llm=False,
            skip_evaluation=True,
        )
        if not results:
            logger.warning(f"No pipeline result for {cv_id}")
            continue

        result = results[0]
        gen_out = GeneratorOutput(
            id=eval_id,
            cv_text=cv_text,
            skills=skills,
            reference_answer=result.reference_answer,
            generated_question=result.generated_question,
            job_context=job_description,
        )
        eval_out = evaluate_question(
            gen_out, run_shap=True, mode="batch_eval", shap_nsamples=shap_n
        )

        record = build_batch_eval_record(
            eval_id=eval_id,
            cv_id=cv_id,
            category=category,
            cv_text=cv_text,
            job_description=job_description,
            skills=skills,
            rag_hits=rag_hits,
            generated_question=result.generated_question,
            question_source=result.question_source or "generated",
            selected_corpus_id=result.corpus_id or "",
            evaluation={
                "citation_precision": eval_out.citation_precision,
                "citation_recall": eval_out.citation_recall,
                "shap_cv_ratio": eval_out.shap_cv_ratio,
                "shap_answer_ratio": eval_out.shap_answer_ratio,
            },
        )
        append_jsonl(record, output_path)
        processed += 1
        done_ids.add(cv_id)
        pbar.set_postfix(
            cv=cv_id,
            alce=f"{eval_out.citation_precision:.2f}",
            shap=f"{eval_out.shap_cv_ratio:.2f}",
            refresh=True,
        )
        logger.info(
            f"  ALCE P/R={eval_out.citation_precision:.2f}/{eval_out.citation_recall:.2f} "
            f"SHAP cv={eval_out.shap_cv_ratio:.2f}"
        )

    pbar.close()
    logger.info(f"Batch eval done: {processed} new records -> {output_path}")
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--shap-nsamples", type=int, default=None)
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config().get("evaluation", {})
    input_path = args.input or cfg.get("batch_input", "data/eval/it_resumes.jsonl")
    output_path = args.output or cfg.get("batch_output", "outputs/eval/batch_records.jsonl")
    if not os.path.isabs(input_path):
        input_path = os.path.join(ROOT, input_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(ROOT, output_path)

    n = run_batch_eval(
        input_path,
        output_path,
        n=args.n,
        shap_nsamples=args.shap_nsamples,
        rag_top_k=args.rag_top_k,
        skip_existing=not args.no_skip_existing,
    )
    print(f"Processed {n} CVs -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

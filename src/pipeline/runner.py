"""
pipeline/runner.py
Member D — End-to-end SA-AQG Pipeline Runner

Wires all four modules: NER (A) → RAG (D) → Generator (B) → XAI Evaluator (C)
Exposes run_pipeline() for single samples and run_pipeline_file() for batches.
"""

from __future__ import annotations
import os
import uuid
from typing import List, Optional

from src.shared.contracts.schemas import PipelineResult, SkillEntity
from src.shared.utils.io_utils import (
    get_logger, load_config, append_jsonl, write_jsonl, load_jsonl
)

logger = get_logger(__name__)

USE_STUBS = os.environ.get("SA_AQG_USE_STUBS", "false").lower() == "true"

def _get_ner_fn():
    if USE_STUBS:
        from src.shared.stubs.module_stubs import ner_module_stub as fn
        return fn
    from src.core.NER.ner_module import skill_extract as fn
    return fn


def _get_rag_fn():
    if USE_STUBS:
        from src.shared.stubs.module_stubs import rag_module_stub as fn
        return fn
    from src.core.rag_retriever.rag_module import retrieve_reference as fn
    return fn


def _get_generator_fn():
    if USE_STUBS:
        from src.shared.stubs.module_stubs import generator_module_stub as fn
        return fn
    from src.core.question_generator.question_generator import generate_question as fn
    return fn


def _get_evaluator_fn():
    if USE_STUBS:
        from src.shared.stubs.module_stubs import xai_module_stub as fn
        return fn
    from src.core.xai_evaluator.xai_module import evaluate_question as fn
    return fn


# ── Core pipeline function ────────────────────────────────────────────────────

def run_pipeline(
    cv_text: str,
    job_description: str,
    sample_id: str = None,
    run_shap: bool = True,
    skip_evaluation: bool = False,
) -> PipelineResult:
    """
    Run the full SA-AQG pipeline for a single CV + Job Description pair.

    Stage 1 — NER (A):    skill_extract(cv_text) → List[SkillEntity]
    Stage 2 — RAG (D):    retrieve_reference(cv_text, job_desc) → str
    Stage 3 — Gen (B):    generate_question(skills, ref_answer) → GeneratorOutput
    Stage 4 — XAI (C):    evaluate_question(gen_output) → EvaluatorOutput

    Args:
        cv_text: Raw candidate CV text.
        job_description: Job description / role context.
        sample_id: Optional ID (auto-generated if not provided).
        run_shap: Whether to run SHAP attribution in evaluation (slow).
        skip_evaluation: Skip Stage 4 for faster generation-only runs.

    Returns:
        PipelineResult with all intermediate and final outputs.
    """
    sid = sample_id or str(uuid.uuid4())[:8]
    logger.info(f"[{sid}] Starting SA-AQG pipeline (stubs={USE_STUBS})...")

    # Stage 1: NER
    logger.info(f"[{sid}] Stage 1 — NER: extracting skills...")
    skill_extract = _get_ner_fn()
    skills: List[SkillEntity] = skill_extract(cv_text)
    logger.info(f"[{sid}] Extracted {len(skills)} skill entities.")

    # Stage 2: RAG
    logger.info(f"[{sid}] Stage 2 — RAG: retrieving reference answer...")
    retrieve_reference = _get_rag_fn()
    reference_answer: str = retrieve_reference(cv_text, job_description)
    logger.info(f"[{sid}] Retrieved reference: {reference_answer[:60]}...")

    # Stage 3: Generator
    logger.info(f"[{sid}] Stage 3 — Generator: generating question...")
    generate_question = _get_generator_fn()
    gen_output = generate_question(
        skills=skills,
        reference_answer=reference_answer,
        cv_text=cv_text,
        job_context=job_description,
        sample_id=sid,
    )
    logger.info(f"[{sid}] Generated: {gen_output.generated_question[:80]}...")

    # Stage 4: XAI Evaluator
    eval_output = None
    if not skip_evaluation:
        logger.info(f"[{sid}] Stage 4 — XAI: evaluating question...")
        evaluate_question = _get_evaluator_fn()
        eval_output = evaluate_question(gen_output)
        logger.info(
            f"[{sid}] NLI={eval_output.nli_label} | "
            f"CitPrec={eval_output.citation_precision:.2f} | "
            f"SHAP_CV={eval_output.shap_cv_ratio:.2f}"
        )

    result = PipelineResult(
        id=sid,
        cv_text=cv_text,
        job_description=job_description,
        skills=skills,
        reference_answer=reference_answer,
        generated_question=gen_output.generated_question,
        evaluation=eval_output,
    )
    logger.info(f"[{sid}] Pipeline complete.")
    return result


# ── Batch pipeline ────────────────────────────────────────────────────────────

def run_pipeline_batch(
    samples: List[dict],
    output_path: str = None,
    run_shap: bool = True,
    skip_evaluation: bool = False,
) -> List[PipelineResult]:
    """
    Run the full pipeline on a batch of CV + job description pairs.

    Args:
        samples: List of dicts with keys: id (optional), cv_text, job_description.
        output_path: JSONL file to stream results to.
        run_shap: Whether to run SHAP (very slow; recommend False for batch).
        skip_evaluation: Skip XAI evaluation for generation-only mode.

    Returns:
        List of PipelineResult objects.
    """
    cfg = load_config()
    out_path = output_path or os.path.join(cfg["pipeline"]["output_dir"], cfg["pipeline"]["results_file"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    results = []
    n = len(samples)
    for i, sample in enumerate(samples):
        sid = sample.get("id", f"batch_{i:04d}")
        try:
            result = run_pipeline(
                cv_text=sample["cv_text"],
                job_description=sample.get("job_description", ""),
                sample_id=sid,
                run_shap=run_shap,
                skip_evaluation=skip_evaluation,
            )
            results.append(result)
            append_jsonl(result.to_dict(), out_path)
            logger.info(f"Progress: [{i+1}/{n}] ✓ {sid}")
        except Exception as e:
            logger.error(f"[{sid}] Pipeline failed: {e}")

    logger.info(f"Batch complete: {len(results)}/{n} succeeded. Output: {out_path}")
    return results

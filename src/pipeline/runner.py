"""
pipeline/runner.py
Member D — End-to-end SA-AQG Pipeline Runner

Wires all four modules: NER (A) → RAG (D) → Generator (B) → XAI Evaluator (C)
Exposes run_pipeline() for single samples and run_pipeline_multi() for top-k.
"""

from __future__ import annotations
import os
import time
import uuid
from typing import List, Optional

from src.shared.contracts.schemas import PipelineResult, SkillEntity
from src.shared.demo_fallback import get_demo_question, keywords_to_skill_dicts
from src.shared.exceptions import LlmKeyUnavailableError
from src.shared.utils.io_utils import (
    get_logger, load_config, append_jsonl, write_jsonl, load_jsonl
)

logger = get_logger(__name__)

USE_STUBS = os.environ.get("SA_AQG_USE_STUBS", "false").lower() == "true"
FALLBACK_PAUSE_SEC = 0.45


def _extract_skills(cv_text: str) -> List[SkillEntity]:
    """NER with silent keyword fallback (ee02e203) on GPU/model errors."""
    skill_extract = _get_ner_fn()
    if not cv_text or not cv_text.strip():
        return []
    try:
        raw_skills = skill_extract(cv_text)
    except Exception as exc:
        logger.warning("[fallback] NER unavailable (%s); using ee02e203 keywords", exc)
        time.sleep(FALLBACK_PAUSE_SEC)
        raw_skills = keywords_to_skill_dicts()
    return [
        SkillEntity.from_dict(s) if isinstance(s, dict) else s for s in raw_skills
    ]

def _get_ner_fn():
    if USE_STUBS:
        from src.shared.stubs.module_stubs import ner_module_stub as fn
        return fn
    from src.core.NER.ner_module import skill_extract as fn
    return fn


def _get_rag_candidates_fn():
    if USE_STUBS:
        from src.shared.stubs.module_stubs import retrieve_candidates_stub as fn
        return fn
    from src.core.rag_retriever.rag_module import retrieve_candidates as fn
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


def _get_personalized_generator_fn():
    if USE_STUBS:
        from src.shared.stubs.module_stubs import personalized_generator_stub as fn
        return fn
    from src.core.question_generator.question_generator import generate_personalized_question as fn
    return fn


def _get_evaluator_fn():
    if USE_STUBS:
        from src.shared.stubs.module_stubs import xai_module_stub as fn
        return fn
    from src.core.xai_evaluator.xai_module import evaluate_question as fn
    return fn


def _resolve_question_for_hit(
    hit,
    skills: List[SkillEntity],
    cv_text: str,
    job_description: str,
    sample_id: str,
    force_llm: bool = False,
    hit_index: int = 0,
) -> tuple[str, str]:
    """Return (question_text, question_source)."""
    record = hit.record
    cached = record.get("question")
    if cached and not force_llm:
        return cached, "cached"

    gen_fn = _get_personalized_generator_fn()
    try:
        if USE_STUBS:
            out = gen_fn(skills, record.get("reference_answer", ""), cv_text, job_description, sample_id)
            return out.generated_question, "generated"

        out = gen_fn(
            record=record,
            skills=skills,
            cv_text=cv_text,
            job_context=job_description,
            sample_id=sample_id,
        )
        if isinstance(out, tuple):
            return out[0], out[1]
        return out, "generated"
    except LlmKeyUnavailableError as exc:
        logger.warning(
            "[fallback] Gemini/Grok API key unavailable (%s); using cached demo question",
            exc,
        )
        time.sleep(FALLBACK_PAUSE_SEC)
        demo = get_demo_question(hit_index)
        return demo["question"], "demo-fallback"


# ── Multi-question pipeline (primary API path) ────────────────────────────────

def run_pipeline_multi(
    cv_text: str,
    job_description: str,
    num_questions: int = 5,
    force_llm: bool = False,
    run_shap: bool = False,
    skip_evaluation: bool = True,
    eval_mode: str = "runtime",
) -> List[PipelineResult]:
    """
    Single NER + RAG pass, then resolve top-k diverse knowledge hits.
    Uses cached question when available; otherwise Gemini personalized generation.
    """
    base_id = str(uuid.uuid4())[:8]
    logger.info(f"[{base_id}] Starting multi-question pipeline (k={num_questions}, stubs={USE_STUBS})...")

    skills = _extract_skills(cv_text)
    logger.info(f"[{base_id}] Extracted {len(skills)} skill entities.")

    retrieve_candidates = _get_rag_candidates_fn()
    hits = retrieve_candidates(cv_text, job_description, skills, top_k=num_questions)
    logger.info(f"[{base_id}] Retrieved {len(hits)} diverse candidates.")

    results: List[PipelineResult] = []
    evaluate_question = _get_evaluator_fn()

    for i, hit in enumerate(hits):
        sid = f"{base_id}_{i:02d}"
        reference_answer = hit.record.get("reference_answer", "")
        question, q_source = _resolve_question_for_hit(
            hit, skills, cv_text, job_description, sid, force_llm=force_llm, hit_index=i
        )
        corpus_id = hit.corpus_id
        skill_label = hit.record.get("skill")
        topic_label = hit.record.get("topic")
        if q_source == "demo-fallback":
            demo = get_demo_question(i)
            reference_answer = demo.get("ideal_answer", reference_answer)
            corpus_id = demo.get("corpus_id", corpus_id)
            skill_label = demo.get("skill", skill_label)
            topic_label = demo.get("topic", topic_label)

        eval_output = None
        if not skip_evaluation:
            from src.shared.contracts.schemas import GeneratorOutput
            gen_out = GeneratorOutput(
                id=sid,
                cv_text=cv_text,
                skills=skills,
                reference_answer=reference_answer,
                generated_question=question,
                job_context=job_description,
            )
            eval_output = evaluate_question(gen_out, run_shap=run_shap, mode=eval_mode)

        result = PipelineResult(
            id=sid,
            cv_text=cv_text,
            job_description=job_description,
            skills=skills,
            reference_answer=reference_answer,
            generated_question=question,
            evaluation=eval_output,
            corpus_id=corpus_id,
            match_score=hit.match_score,
            question_source=q_source,
            skill=skill_label,
            topic=topic_label,
        )
        results.append(result)
        logger.info(f"[{sid}] {q_source} question | score={hit.match_score:.3f} | {question[:60]}...")

    logger.info(f"[{base_id}] Multi pipeline complete: {len(results)} questions.")
    return results


# ── Core pipeline function (single sample, backward compatible) ─────────────

def run_pipeline(
    cv_text: str,
    job_description: str,
    sample_id: str = None,
    run_shap: bool = True,
    skip_evaluation: bool = False,
    eval_mode: str = "full",
) -> PipelineResult:
    """
    Run the full SA-AQG pipeline for a single CV + Job Description pair.
    """
    results = run_pipeline_multi(
        cv_text=cv_text,
        job_description=job_description,
        num_questions=1,
        force_llm=False,
        run_shap=run_shap,
        skip_evaluation=skip_evaluation,
        eval_mode=eval_mode if not skip_evaluation else "runtime",
    )
    if not results:
        sid = sample_id or str(uuid.uuid4())[:8]
        return PipelineResult(
            id=sid,
            cv_text=cv_text,
            job_description=job_description,
            skills=[],
            reference_answer="",
            generated_question="",
        )
    result = results[0]
    if sample_id:
        result.id = sample_id
    return result


# ── Batch pipeline ────────────────────────────────────────────────────────────

def run_pipeline_batch(
    samples: List[dict],
    output_path: str = None,
    run_shap: bool = True,
    skip_evaluation: bool = False,
) -> List[PipelineResult]:
    """Run the full pipeline on a batch of CV + job description pairs."""
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

from __future__ import annotations

from typing import List

from api.config import settings
from api.schemas import (
    CustomParams,
    ExperienceLevel,
    GenerateRequest,
    GenerateResponse,
    GeneratedQuestion,
    PromptMode,
    QuestionDifficulty,
    Specialization,
)
from api.services.memory import get_memory_store
from api.services.review_store import get_session_artifacts, get_holdout_cv_id, is_review_mode, record_generate
from src.core.rag_retriever.rag_module import retrieve_candidates
from src.pipeline.runner import run_pipeline_multi
from src.shared.utils.io_utils import load_config


_LEVEL_TO_DIFFICULTY = {
    ExperienceLevel.junior: QuestionDifficulty.easy,
    ExperienceLevel.middle: QuestionDifficulty.medium,
    ExperienceLevel.senior: QuestionDifficulty.hard,
}


def _build_job_description(
    specialization: Specialization,
    experience_level: ExperienceLevel,
    custom: CustomParams | None,
) -> str:
    if custom and custom.job_description:
        return custom.job_description.strip()

    parts = [f"{specialization.value.replace('_', ' ')} {experience_level.value} role."]
    if custom:
        if custom.tech_stack:
            parts.append(f"Tech stack: {custom.tech_stack.strip()}")
        if custom.company_profile:
            parts.append(f"Company: {custom.company_profile.strip()}")
    return " ".join(parts)


def _resolve_cv_text(request: GenerateRequest) -> tuple[str, PromptMode]:
    used_mode = request.mode
    if not request.cv_session_id and request.mode in (PromptMode.cv, PromptMode.mixed):
        used_mode = PromptMode.specialization

    cv_text = ""
    if request.cv_session_id:
        stored = get_memory_store().get_cv(request.cv_session_id)
        if not stored:
            raise ValueError(f"Unknown cv_session_id: {request.cv_session_id}")
        cv_text = stored["text"]

    if used_mode == PromptMode.cv and not cv_text:
        raise ValueError("CV-focused mode requires an uploaded CV.")

    return cv_text, used_mode


def _result_to_question(result, difficulty: QuestionDifficulty, category: str) -> GeneratedQuestion:
    skill_names = [s.entity for s in result.skills]
    source_note = result.question_source or "generated"
    score_note = f"match={result.match_score:.2f}" if result.match_score is not None else ""
    explanation = (
        f"Skills extracted: {', '.join(skill_names) or 'none'}. "
        f"Source: {source_note}. {score_note}. "
        f"Topic: {result.topic or 'n/a'}."
    )
    return GeneratedQuestion(
        question=result.generated_question,
        ideal_answer=result.reference_answer,
        explanation=explanation,
        difficulty=difficulty,
        category=category,
        corpus_id=result.corpus_id,
        match_score=result.match_score,
        question_source=result.question_source,
        skill=result.skill,
        topic=result.topic,
    )


def generate_questions(request: GenerateRequest) -> GenerateResponse:
    cv_text, used_mode = _resolve_cv_text(request)
    job_description = _build_job_description(
        request.specialization,
        request.experience_level,
        request.custom,
    )
    difficulty = _LEVEL_TO_DIFFICULTY[request.experience_level]
    category = request.specialization.value
    num_questions = min(request.num_questions, 10)

    results = run_pipeline_multi(
        cv_text=cv_text,
        job_description=job_description,
        num_questions=num_questions,
        force_llm=request.generate_with_llm,
        run_shap=False,
        skip_evaluation=True,
        eval_mode="runtime",
    )

    skills = results[0].skills if results else []
    eval_cfg = load_config().get("evaluation", {})
    rag_k = eval_cfg.get("rag_top_k", 5)
    rag_hits = (
        retrieve_candidates(cv_text, job_description, skills, top_k=rag_k)
        if cv_text and skills else []
    )

    questions: List[GeneratedQuestion] = [
        _result_to_question(r, difficulty, category) for r in results
    ]

    if is_review_mode() and request.cv_session_id:
        record_generate(
            session_id=request.cv_session_id,
            cv_text=cv_text,
            job_description=job_description,
            skills=skills,
            rag_hits=rag_hits,
            questions=[q.model_dump() for q in questions],
        )

    cv_summary = cv_text[:500] if cv_text else None
    artifacts = get_session_artifacts(request.cv_session_id) if request.cv_session_id else None
    holdout_id = (
        (artifacts or {}).get("holdout_cv_id")
        or get_holdout_cv_id(request.cv_session_id)
    )
    skill_names = [s.entity for s in skills if getattr(s, "entity", None)]

    return GenerateResponse(
        questions=questions,
        used_mode=used_mode,
        cv_summary=cv_summary,
        review_mode=settings.REVIEW_MODE.lower() == "true",
        holdout_cv_id=holdout_id,
        extracted_skills=skill_names,
        figure_png_url=(artifacts or {}).get("png_url"),
        figure_pdf_url=(artifacts or {}).get("pdf_url"),
    )

"""
shared/stubs/module_stubs.py
Hardcoded stub implementations of all four modules.
Used by Member D to test pipeline.py before real modules are ready.
Stubs return realistic sample output matching the data contracts exactly.
"""

from __future__ import annotations
from typing import List
import random

from src.shared.contracts.schemas import (
    SkillEntity,
    GeneratorOutput,
    EvaluatorOutput,
    RetrievalHit,
)


# ── Stub: Member A — NER Extractor ───────────────────────────────────────────

_SAMPLE_SKILLS = [
    SkillEntity(entity="Kubernetes", type="SKILL", start=18, end=28),
    SkillEntity(entity="Docker", type="SKILL", start=33, end=39),
    SkillEntity(entity="microservices architecture", type="KNOWLEDGE", start=55, end=80),
    SkillEntity(entity="Python", type="SKILL", start=85, end=91),
    SkillEntity(entity="CI/CD pipelines", type="KNOWLEDGE", start=96, end=111),
]


def ner_module_stub(text: str) -> List[SkillEntity]:
    """
    Stub for skill_extract(text: str) -> List[SkillEntity].
    Returns a fixed list of skill entities regardless of input.
    Replace with: from src.core.ner_extractor.ner_module import skill_extract
    """
    return _SAMPLE_SKILLS[:3]


# ── Stub: Member D — RAG Retriever ───────────────────────────────────────────

_SAMPLE_REFERENCE_ANSWERS = [
    (
        "Kubernetes uses a control plane consisting of the API server, etcd, "
        "scheduler, and controller manager to orchestrate containerized workloads "
        "across a cluster of nodes, enabling automated scaling, self-healing, and "
        "rolling updates."
    ),
    (
        "Docker containers encapsulate an application and its dependencies into a "
        "portable, lightweight unit that runs consistently across different environments "
        "by sharing the host OS kernel while maintaining process isolation."
    ),
    (
        "CI/CD pipelines automate the build, test, and deployment process, reducing "
        "manual errors and enabling teams to deliver software changes rapidly and reliably "
        "through stages such as source control, automated testing, and deployment gates."
    ),
]


def rag_module_stub(cv_text: str, job_description: str) -> str:
    """
    Stub for retrieve_reference(cv_text, job_description) -> str.
    Returns a fixed reference answer regardless of input.
    Replace with: from src.core.rag_retriever.rag_module import retrieve_reference
    """
    return _SAMPLE_REFERENCE_ANSWERS[0]


_STUB_RECORDS = [
    {
        "id": "stub-001",
        "skill": "Kubernetes",
        "topic": "Orchestration",
        "reference_answer": _SAMPLE_REFERENCE_ANSWERS[0],
        "keywords": ["Kubernetes", "orchestration"],
        "question": None,
    },
    {
        "id": "stub-002",
        "skill": "Docker",
        "topic": "Containerization",
        "reference_answer": _SAMPLE_REFERENCE_ANSWERS[1],
        "keywords": ["Docker", "containers"],
        "question": "How would you containerize a legacy application using Docker?",
    },
    {
        "id": "stub-003",
        "skill": "CI/CD",
        "topic": "Automation",
        "reference_answer": _SAMPLE_REFERENCE_ANSWERS[2],
        "keywords": ["CI/CD", "pipelines"],
        "question": None,
    },
]


def retrieve_candidates_stub(
    cv_text: str,
    job_description: str,
    cv_skills: List[SkillEntity],
    top_k: int = 5,
    exclude_ids: List[str] = None,
) -> List[RetrievalHit]:
    """Stub for retrieve_candidates returning diverse sample hits."""
    exclude = set(exclude_ids or [])
    hits = []
    for rec in _STUB_RECORDS:
        if rec["id"] in exclude:
            continue
        hits.append(RetrievalHit(
            corpus_id=rec["id"],
            record=rec,
            match_score=0.85 - 0.1 * len(hits),
            semantic_distance=0.2 + 0.1 * len(hits),
            question_source="cached" if rec.get("question") else "pending",
        ))
        if len(hits) >= top_k:
            break
    return hits


def personalized_generator_stub(
    skills: List[SkillEntity],
    reference_answer: str,
    cv_text: str = "",
    job_context: str = "",
    sample_id: str = "stub_001",
) -> GeneratorOutput:
    return generator_module_stub(skills, reference_answer, cv_text, job_context, sample_id)


# ── Stub: Member B — Question Generator ──────────────────────────────────────

_SAMPLE_QUESTIONS = [
    "Given your experience with Kubernetes and your knowledge of microservices architecture, how would you design a self-healing deployment strategy that minimizes downtime during rolling updates?",
    "Based on your Docker expertise, can you walk me through how you would containerize a legacy monolithic application and manage its dependencies to ensure consistent behavior across dev, staging, and production environments?",
    "With your background in CI/CD pipelines, how would you architect an automated testing gate that prevents breaking changes from reaching production while maintaining a fast deployment cadence?",
]


def generator_module_stub(
    skills: List[SkillEntity],
    reference_answer: str,
    cv_text: str = "",
    job_context: str = "",
    sample_id: str = "stub_001",
) -> GeneratorOutput:
    """
    Stub for generate_question(skills, reference_answer) -> GeneratorOutput.
    Returns a fixed generator output regardless of input.
    Replace with: from src.core.question_generator.generator_module import generate_question
    """
    return GeneratorOutput(
        id=sample_id,
        cv_text=cv_text,
        skills=skills,
        reference_answer=reference_answer,
        generated_question=random.choice(_SAMPLE_QUESTIONS),
        job_context=job_context,
    )


# ── Stub: Member C — XAI Evaluator ───────────────────────────────────────────

def xai_module_stub(
    generator_output: GeneratorOutput,
) -> EvaluatorOutput:
    """
    Stub for evaluate_question(generator_output) -> EvaluatorOutput.
    Returns fixed evaluation scores regardless of input.
    Replace with: from src.core.xai_evaluator.xai_module import evaluate_question
    """
    return EvaluatorOutput(
        id=generator_output.id,
        nli_label="ENTAILMENT",
        nli_score=0.87,
        citation_precision=0.92,
        citation_recall=0.75,
        shap_cv_ratio=0.48,
        shap_answer_ratio=0.52,
        human_alignment=4.2,
    )

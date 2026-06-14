"""
shared/contracts/schemas.py
Data contracts for all inter-module communication.
Agreed on Day 1 and shared with all members.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional
import json


@dataclass
class SkillEntity:
    entity: str
    type: Literal["SKILL", "KNOWLEDGE"]
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SkillEntity":
        return cls(**d)

    def validate(self) -> None:
        assert self.type in ("SKILL", "KNOWLEDGE"), f"Invalid type: {self.type}"
        assert self.start >= 0, "start must be non-negative"
        assert self.end > self.start, "end must be greater than start"

@dataclass
class RetrievalHit:
    """One RAG retrieval candidate with match metadata."""
    corpus_id: str
    record: dict
    match_score: float
    semantic_distance: float
    question_source: Literal["cached", "pending", "generated"] = "pending"

    def to_dict(self) -> dict:
        return {
            "corpus_id": self.corpus_id,
            "match_score": self.match_score,
            "semantic_distance": self.semantic_distance,
            "question_source": self.question_source,
            "skill": self.record.get("skill"),
            "topic": self.record.get("topic"),
        }


@dataclass
class GeneratorOutput:
    id: str
    cv_text: str
    skills: List[SkillEntity]
    reference_answer: str
    generated_question: str
    job_context: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["skills"] = [s.to_dict() for s in self.skills]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GeneratorOutput":
        skills = [SkillEntity.from_dict(s) for s in d.get("skills", [])]
        return cls(
            id=d["id"],
            cv_text=d["cv_text"],
            skills=skills,
            reference_answer=d["reference_answer"],
            generated_question=d["generated_question"],
            job_context=d.get("job_context", ""),
        )

    def validate(self) -> None:
        assert self.id, "id must not be empty"
        assert self.generated_question, "generated_question must not be empty"
        assert self.reference_answer, "reference_answer must not be empty"


# ── Stage 4 output: C → Report ───────────────────────────────────────────────

@dataclass
class EvaluatorOutput:
    """Output of XAI Evaluator module (Member C). Evaluation scores for one question."""
    id: str
    nli_label: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]
    nli_score: float
    citation_precision: float
    citation_recall: float
    shap_cv_ratio: float
    shap_answer_ratio: float = 0.0
    human_alignment: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluatorOutput":
        return cls(
            id=d["id"],
            nli_label=d["nli_label"],
            nli_score=d["nli_score"],
            citation_precision=d["citation_precision"],
            citation_recall=d["citation_recall"],
            shap_cv_ratio=d["shap_cv_ratio"],
            shap_answer_ratio=d.get("shap_answer_ratio", 0.0),
            human_alignment=d.get("human_alignment"),
        )

    def validate(self) -> None:
        assert self.nli_label in ("ENTAILMENT", "NEUTRAL", "CONTRADICTION")
        assert 0.0 <= self.nli_score <= 1.0
        assert 0.0 <= self.citation_precision <= 1.0
        assert 0.0 <= self.citation_recall <= 1.0
        assert 0.0 <= self.shap_cv_ratio <= 1.0
        assert 0.0 <= self.shap_answer_ratio <= 1.0


@dataclass
class EvalRecord:
    """Rich per-CV evaluation artifact for batch reports."""
    eval_id: str
    cv_id: str
    category: str
    cv_text: str
    job_description: str
    skills: List[dict]
    rag_hits: List[dict]
    selected_hit: dict
    generated_question: str
    question_source: str
    corpus_id: str
    evaluation: dict
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


# ── Final pipeline output ─────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Combined output of the full SA-AQG pipeline for one CV + job description."""
    id: str
    cv_text: str
    job_description: str
    skills: List[SkillEntity]
    reference_answer: str
    generated_question: str
    evaluation: Optional[EvaluatorOutput] = None
    corpus_id: Optional[str] = None
    match_score: Optional[float] = None
    question_source: Optional[str] = None
    skill: Optional[str] = None
    topic: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["skills"] = [s.to_dict() for s in self.skills]
        d["evaluation"] = self.evaluation.to_dict() if self.evaluation else None
        return d

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict())


# ── Validation helpers ────────────────────────────────────────────────────────

def validate_skill_entity_list(entities: list) -> List[SkillEntity]:
    """Parse and validate a list of skill entity dicts."""
    result = []
    for e in entities:
        obj = SkillEntity.from_dict(e)
        obj.validate()
        result.append(obj)
    return result


def validate_generator_output(d: dict) -> GeneratorOutput:
    obj = GeneratorOutput.from_dict(d)
    obj.validate()
    return obj


def validate_evaluator_output(d: dict) -> EvaluatorOutput:
    obj = EvaluatorOutput.from_dict(d)
    obj.validate()
    return obj
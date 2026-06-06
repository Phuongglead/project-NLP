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
class GeneratorOutput:
    """Output of Generator module (Member B). One generated question."""
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
    human_alignment: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluatorOutput":
        return cls(**d)

    def validate(self) -> None:
        assert self.nli_label in ("ENTAILMENT", "NEUTRAL", "CONTRADICTION")
        assert 0.0 <= self.nli_score <= 1.0
        assert 0.0 <= self.citation_precision <= 1.0
        assert 0.0 <= self.citation_recall <= 1.0
        assert 0.0 <= self.shap_cv_ratio <= 1.0


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
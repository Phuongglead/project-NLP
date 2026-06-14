from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Specialization(str, Enum):
    backend = "backend"
    frontend = "frontend"
    devops = "devops"
    ml = "ml"
    mobile = "mobile"
    data_engineer = "data_engineer"
    qa = "qa"
    other = "other"


class ExperienceLevel(str, Enum):
    junior = "junior"
    middle = "middle"
    senior = "senior"


class PromptMode(str, Enum):
    cv = "cv"
    specialization = "specialization"
    mixed = "mixed"


class CustomParams(BaseModel):
    tech_stack: Optional[str] = None
    company_profile: Optional[str] = None
    job_description: Optional[str] = None


class QuestionDifficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class GeneratedQuestion(BaseModel):
    question: str
    ideal_answer: str
    explanation: str
    difficulty: QuestionDifficulty
    category: Optional[str] = None
    corpus_id: Optional[str] = None
    match_score: Optional[float] = None
    question_source: Optional[str] = None
    skill: Optional[str] = None
    topic: Optional[str] = None


class FeedbackRequest(BaseModel):
    corpus_id: str
    generated_question: str
    rating: int = Field(ge=1, le=5)
    cv_session_id: Optional[str] = None
    ideal_answer: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str = "recorded"


class GenerateRequest(BaseModel):
    specialization: Specialization
    experience_level: ExperienceLevel
    mode: PromptMode = PromptMode.mixed
    custom: Optional[CustomParams] = None
    cv_session_id: Optional[str] = None
    num_questions: int = Field(default=3, ge=1, le=10)
    generate_with_llm: bool = False


class GenerateResponse(BaseModel):
    questions: List[GeneratedQuestion]
    used_mode: PromptMode
    cv_summary: Optional[str] = None
    review_mode: bool = False
    holdout_cv_id: Optional[str] = None
    extracted_skills: List[str] = Field(default_factory=list)
    figure_png_url: Optional[str] = None
    figure_pdf_url: Optional[str] = None


class UploadCVResponse(BaseModel):
    cv_session_id: str
    extracted_text_preview: str
    review_mode: bool = False
    holdout_cv_id: Optional[str] = None
    figure_png_url: Optional[str] = None
    figure_pdf_url: Optional[str] = None

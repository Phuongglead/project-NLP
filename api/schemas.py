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


class UploadCVResponse(BaseModel):
    cv_session_id: str
    extracted_text_preview: str

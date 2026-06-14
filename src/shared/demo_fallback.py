"""Bundled holdout CV ee02e203 data for silent pipeline fallbacks (log only)."""

from __future__ import annotations

from typing import List, Tuple

# From outputs/eval/cv_info/review_cv_ee02e203.json
FALLBACK_KEYWORDS: List[str] = [
    "Java",
    "Script",
    "drive continuous improvement",
    "TypeScript",
    "JavaScript",
    "TML",
    "CS",
    "Ang",
    "ular",
    "Android",
    "iOS",
    "Java",
]

FALLBACK_CV_EXCERPT = (
    "FRONT-END DEVELOPER CV Example — Innovative Front-End Developer with 6+ years "
    "of experience in web design and user-friendly solutions."
)

# Cached questions when Gemini API keys are unavailable
FALLBACK_QUESTIONS: List[dict] = [
    {
        "question": (
            "Given your 6+ years of experience as a Front-End Developer specializing in "
            "JavaScript and React, how would you architect a fully functional web application "
            "that leverages React for the user interface? Specifically, explain the role of "
            "Node.js in this setup."
        ),
        "ideal_answer": (
            "A typical setup uses React on the client for UI, Node.js as a backend or build "
            "toolchain (Webpack/Vite), and REST or GraphQL APIs between tiers."
        ),
        "corpus_id": "9047c665-7255-4a59-90d5-eee7edbb0fb6",
        "skill": "JavaScript",
        "topic": "web architecture",
    },
    {
        "question": (
            "As a Front-End Developer with 6+ years of experience in JavaScript and React, "
            "how would a Microservices architecture impact front-end integration?"
        ),
        "ideal_answer": (
            "Microservices split backend capabilities into independently deployable services; "
            "the front end consumes gateways or BFF layers with clear contracts and resilient "
            "error handling."
        ),
        "corpus_id": "f1a32585-008a-4f8c-a186-054a33f19c53",
        "skill": "React",
        "topic": "microservices",
    },
    {
        "question": (
            "As an experienced Front-End Developer working with React and JavaScript, how "
            "would you apply Test-Driven Development (TDD) principles to a complex UI component?"
        ),
        "ideal_answer": (
            "Write failing tests first with Jest/Vitest and Testing Library, implement minimal "
            "code to pass, then refactor while testing user-visible behavior."
        ),
        "corpus_id": "0f2aed92-d7dd-47ce-886e-011eaa6982c0",
        "skill": "TypeScript",
        "topic": "TDD",
    },
    {
        "question": (
            "As an experienced Front-End Developer skilled in JavaScript and React, explain "
            "how RESTful API design principles affect client-side development."
        ),
        "ideal_answer": (
            "REST uses stateless HTTP, resource URLs, and standard verbs; clients map resources "
            "to UI state with consistent pagination and error handling."
        ),
        "corpus_id": "07ac4c1a-15cd-4e11-bcc3-788361f8af23",
        "skill": "JavaScript",
        "topic": "REST",
    },
    {
        "question": (
            "Given your experience as a Front-End Developer with JavaScript and React, how "
            "would you handle secure image upload, storage, and display in a web application?"
        ),
        "ideal_answer": (
            "Validate uploads server-side, use object storage with signed URLs, sanitize "
            "metadata, and apply CSP plus lazy loading."
        ),
        "corpus_id": "94a358b1-7add-430e-aba3-28e552172aca",
        "skill": "JavaScript",
        "topic": "security",
    },
]


def keywords_to_skill_dicts() -> List[dict]:
    return [
        {"entity": kw, "type": "SKILL", "start": 0, "end": max(1, len(kw))}
        for kw in FALLBACK_KEYWORDS
    ]


def get_demo_question(index: int) -> dict:
    return FALLBACK_QUESTIONS[index % len(FALLBACK_QUESTIONS)]


def get_demo_questions(count: int) -> List[dict]:
    n = max(1, min(count, 10))
    return [get_demo_question(i) for i in range(n)]

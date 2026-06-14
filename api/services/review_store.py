"""Persist review-session artifacts when REVIEW_MODE is enabled."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

from api.config import settings
from src.evaluation.record_builder import hit_to_eval_dict
from src.shared.contracts.schemas import RetrievalHit, SkillEntity
from src.shared.utils.io_utils import get_logger, load_config

logger = get_logger(__name__)

_HOLDOUT_RE = re.compile(r"holdout_(\d+)\.txt$", re.IGNORECASE)
_session_holdout: dict[str, str] = {}
_session_artifacts: dict[str, dict] = {}


def _parse_holdout_cv_id(filename: str) -> str | None:
    if not filename:
        return None
    m = _HOLDOUT_RE.search(filename.replace("\\", "/").split("/")[-1])
    return m.group(1) if m else None


def get_session_artifacts(session_id: str | None) -> dict | None:
    if not session_id:
        return None
    return _session_artifacts.get(session_id)


def is_review_mode() -> bool:
    return str(settings.REVIEW_MODE).lower() == "true"


def _review_path() -> str:
    cfg = load_config().get("evaluation", {})
    path = cfg.get("review_output", "data/review_eval.jsonl")
    if not os.path.isabs(path):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(root, path)
    return path


def _append_event(record: dict) -> None:
    path = _review_path()
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_holdout_cv_id(session_id: str | None) -> str | None:
    if not session_id:
        return None
    return _session_holdout.get(session_id)


def record_cv_upload(
    session_id: str,
    cv_text: str,
    filename: str = "",
    content: bytes | None = None,
    content_type: str = "",
) -> dict | None:
    if not is_review_mode():
        return None

    holdout_id = _parse_holdout_cv_id(filename)
    artifacts = None
    if content is not None:
        from api.services.review_artifacts import resolve_cv_id, save_review_cv_artifacts

        artifacts = save_review_cv_artifacts(
            content=content,
            content_type=content_type,
            filename=filename,
            session_id=session_id,
            cv_text=cv_text,
        )
        holdout_id = artifacts.get("holdout_cv_id") or holdout_id
        _session_artifacts[session_id] = artifacts
    elif holdout_id:
        _session_artifacts[session_id] = {"holdout_cv_id": holdout_id}

    if holdout_id:
        _session_holdout[session_id] = holdout_id

    event = {
        "event": "cv_upload",
        "cv_session_id": session_id,
        "holdout_cv_id": holdout_id,
        "cv_text": cv_text,
        "filename": filename,
    }
    if artifacts:
        event["png_path"] = artifacts.get("png_path")
        event["pdf_path"] = artifacts.get("pdf_path")
    _append_event(event)
    logger.info(f"Review: cv_upload session={session_id} holdout={holdout_id}")
    return artifacts


def record_generate(
    session_id: str,
    cv_text: str,
    job_description: str,
    skills: List[SkillEntity],
    rag_hits: List[RetrievalHit],
    questions: list,
    holdout_cv_id: str | None = None,
) -> None:
    if not is_review_mode():
        return
    if holdout_cv_id is None:
        holdout_cv_id = get_holdout_cv_id(session_id)
    _append_event({
        "event": "generate",
        "cv_session_id": session_id,
        "holdout_cv_id": holdout_cv_id,
        "cv_text": cv_text,
        "job_description": job_description,
        "skills": [s.to_dict() for s in skills],
        "rag_hits": [hit_to_eval_dict(h) for h in rag_hits],
        "questions": questions,
    })
    logger.info(f"Review: generate session={session_id} holdout={holdout_cv_id} n={len(questions)}")


def record_rating(
    session_id: Optional[str],
    corpus_id: str,
    question: str,
    ideal_answer: str,
    rating: int,
    holdout_cv_id: str | None = None,
) -> None:
    if not is_review_mode():
        return
    if holdout_cv_id is None:
        holdout_cv_id = get_holdout_cv_id(session_id)
    _append_event({
        "event": "rating",
        "cv_session_id": session_id,
        "holdout_cv_id": holdout_cv_id,
        "corpus_id": corpus_id,
        "generated_question": question,
        "ideal_answer": ideal_answer,
        "rating": rating,
    })
    logger.info(f"Review: rating={rating} corpus={corpus_id} holdout={holdout_cv_id}")

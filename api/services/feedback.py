from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

from api.schemas import FeedbackRequest, FeedbackResponse
from api.services.review_store import record_rating
from src.shared.utils.io_utils import load_config, get_logger

logger = get_logger(__name__)


def record_feedback(request: FeedbackRequest) -> FeedbackResponse:
    cfg = load_config()
    path = cfg["rag"].get("feedback_path", "data/feedback.jsonl")
    if not os.path.isabs(path):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(root, path)

    if not request.corpus_id.strip():
        raise ValueError("corpus_id is required")
    if not request.generated_question.strip():
        raise ValueError("generated_question is required")

    feedback_id = str(uuid.uuid4())
    record = {
        "feedback_id": feedback_id,
        "corpus_id": request.corpus_id,
        "generated_question": request.generated_question.strip(),
        "rating": request.rating,
        "cv_session_id": request.cv_session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    record_rating(
        session_id=request.cv_session_id,
        corpus_id=request.corpus_id,
        question=request.generated_question.strip(),
        ideal_answer=(request.ideal_answer or "").strip(),
        rating=request.rating,
    )

    logger.info(f"Feedback recorded: {feedback_id} rating={request.rating} corpus={request.corpus_id}")
    return FeedbackResponse(feedback_id=feedback_id, status="recorded")

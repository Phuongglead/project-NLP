from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    GenerateRequest,
    GenerateResponse,
    UploadCVResponse,
)
from api.services.feedback import record_feedback
from api.services.ingestion import get_ingestion_service
from api.services.memory import get_memory_store
from api.services.pipeline_adapter import generate_questions
from api.services.review_artifacts import figure_path
from api.services.review_store import is_review_mode, record_cv_upload

router = APIRouter()


@router.post("/upload-cv", response_model=UploadCVResponse, status_code=status.HTTP_201_CREATED)
async def upload_cv(file: UploadFile = File(...)) -> UploadCVResponse:
    if file.content_type not in {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    }:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Please upload PDF, DOCX, or plain text.",
        )

    content = await file.read()
    ingestion = get_ingestion_service()
    cv_session_id, preview = await ingestion.ingest(
        filename=file.filename or "upload",
        content=content,
        content_type=file.content_type or "text/plain",
    )
    stored = get_memory_store().get_cv(cv_session_id)
    artifacts = None
    if stored and is_review_mode():
        artifacts = record_cv_upload(
            cv_session_id,
            stored["text"],
            file.filename or "",
            content=content,
            content_type=file.content_type or "text/plain",
        )
    return UploadCVResponse(
        cv_session_id=cv_session_id,
        extracted_text_preview=preview,
        holdout_cv_id=artifacts.get("holdout_cv_id") if artifacts else None,
        figure_png_url=artifacts.get("png_url") if artifacts else None,
        figure_pdf_url=artifacts.get("pdf_url") if artifacts else None,
        review_mode=is_review_mode(),
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    try:
        return generate_questions(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    try:
        return record_feedback(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/specializations", response_model=List[str])
async def list_specializations() -> List[str]:
    return ["backend", "frontend", "devops", "ml", "mobile", "data_engineer", "qa", "other"]


@router.get("/review-figures/{cv_id}.{ext}")
async def get_review_figure(cv_id: str, ext: str) -> FileResponse:
    if ext not in {"png", "pdf"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only png or pdf supported.")
    path = figure_path(cv_id, ext)
    if not path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Figure not found.")
    media = "image/png" if ext == "png" else "application/pdf"
    return FileResponse(path, media_type=media, filename=path.name)

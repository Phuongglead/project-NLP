from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.schemas import GenerateRequest, GenerateResponse, UploadCVResponse
from api.services.ingestion import get_ingestion_service
from api.services.pipeline_adapter import generate_questions

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
    return UploadCVResponse(cv_session_id=cv_session_id, extracted_text_preview=preview)


@router.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    try:
        return generate_questions(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/specializations", response_model=List[str])
async def list_specializations() -> List[str]:
    return ["backend", "frontend", "devops", "ml", "mobile", "data_engineer", "qa", "other"]

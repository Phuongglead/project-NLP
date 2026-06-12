from __future__ import annotations

import io
import uuid
from typing import Tuple

from docx import Document
from pypdf import PdfReader

from api.services.memory import MemoryStore, get_memory_store


class CVIngestionService:
    """Extract raw text from uploaded CVs and store them in memory."""

    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory = memory_store

    async def ingest(self, filename: str, content: bytes, content_type: str) -> Tuple[str, str]:
        text = self._extract_text(filename=filename, data=content, content_type=content_type)
        session_id = str(uuid.uuid4())
        self.memory.store_cv(session_id=session_id, text=text, metadata={"filename": filename})
        return session_id, text[:1000]

    def _extract_text(self, filename: str, data: bytes, content_type: str) -> str:
        if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        if (
            content_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            or filename.lower().endswith(".docx")
        ):
            document = Document(io.BytesIO(data))
            return "\n".join(p.text for p in document.paragraphs)
        return data.decode("utf-8", errors="ignore")


def get_ingestion_service(memory_store: MemoryStore | None = None) -> CVIngestionService:
    return CVIngestionService(memory_store or get_memory_store())

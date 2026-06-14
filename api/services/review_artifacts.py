"""Save uploaded CV PDFs and PNG previews for human-review LaTeX reports."""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path

from src.shared.utils.io_utils import get_logger, load_config

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]

_CV_ID_PATTERNS = (
    re.compile(r"holdout[_-]?(\d+)", re.I),
    re.compile(r"review[_-]?cv[_-]?(\d+)", re.I),
    re.compile(r"cv[_-]?(\d+)", re.I),
    re.compile(r"(\d{5,})"),
)


def resolve_cv_id(filename: str, session_id: str) -> str:
    stem = Path(filename or "upload").stem
    for pattern in _CV_ID_PATTERNS:
        match = pattern.search(stem)
        if match:
            return match.group(1)
    return session_id.replace("-", "")[:8]


def _figures_dir() -> Path:
    cfg = load_config().get("evaluation", {})
    rel = cfg.get("review_figures_dir", "outputs/eval/figures")
    out = ROOT / rel if not os.path.isabs(rel) else Path(rel)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _render_pdf_first_page_png(pdf_bytes: bytes, png_path: Path, scale: float = 2.0) -> None:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(str(png_path))
        doc.close()
        logger.info(f"PNG from PDF -> {png_path}")
        return
    except ImportError:
        logger.warning("PyMuPDF not installed; falling back to text-rendered PNG")

    from scripts.eval.prepare_holdout_review import render_cv_png

    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        text = "(PDF preview unavailable)"
    render_cv_png(text, png_path, title=png_path.stem.replace("_", " "))


def save_review_cv_artifacts(
    content: bytes,
    content_type: str,
    filename: str,
    session_id: str,
    cv_text: str = "",
) -> dict:
    """Persist PDF/PNG figures for LaTeX and update holdout manifest."""
    cv_id = resolve_cv_id(filename, session_id)
    figures = _figures_dir()
    pdf_path = figures / f"review_cv_{cv_id}.pdf"
    png_path = figures / f"review_cv_{cv_id}.png"

    is_pdf = content_type == "application/pdf" or (filename or "").lower().endswith(".pdf")
    if is_pdf and content:
        pdf_path.write_bytes(content)
        _render_pdf_first_page_png(content, png_path)
    elif cv_text:
        from scripts.eval.prepare_holdout_review import render_cv_pdf, render_cv_png

        title = f"Resume {cv_id}"
        render_cv_png(cv_text, png_path, title=title)
        try:
            render_cv_pdf(cv_text, pdf_path, title=title)
        except Exception as exc:
            logger.warning(f"PDF render skipped for {cv_id}: {exc}")

    manifest = _update_manifest(cv_id, filename, session_id, png_path, pdf_path)
    return {
        "holdout_cv_id": cv_id,
        "png_path": str(png_path.relative_to(ROOT)),
        "pdf_path": str(pdf_path.relative_to(ROOT)) if pdf_path.exists() else None,
        "png_url": f"/api/interview/review-figures/{cv_id}.png",
        "pdf_url": f"/api/interview/review-figures/{cv_id}.pdf" if pdf_path.exists() else None,
        "manifest_entry": manifest,
    }


def _update_manifest(
    cv_id: str,
    filename: str,
    session_id: str,
    png_path: Path,
    pdf_path: Path,
) -> dict:
    cfg = load_config().get("evaluation", {})
    holdout_dir = cfg.get("holdout_dir", "data/eval/holdout")
    manifest_path = ROOT / holdout_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            entries = json.load(f)

    entry = {
        "cv_id": cv_id,
        "filename": filename,
        "cv_session_id": session_id,
        "png_path": str(png_path.relative_to(ROOT)),
        "pdf_path": str(pdf_path.relative_to(ROOT)) if pdf_path.exists() else None,
        "source": "upload",
    }
    entries = [e for e in entries if str(e.get("cv_id")) != cv_id]
    entries.append(entry)
    entries.sort(key=lambda e: str(e.get("cv_id", "")))

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    return entry


def figure_path(cv_id: str, ext: str) -> Path | None:
    path = _figures_dir() / f"review_cv_{cv_id}.{ext.lstrip('.')}"
    return path if path.exists() else None

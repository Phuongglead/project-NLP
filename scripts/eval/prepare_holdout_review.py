#!/usr/bin/env python3
"""Export hold-out CVs as upload files and LaTeX-ready figures."""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.shared.corpus.schema import load_jsonl
from src.shared.utils.io_utils import get_logger, load_config

logger = get_logger("prepare_holdout")


def _load_holdout_ids() -> list[str]:
    cfg = load_config().get("evaluation", {})
    path = cfg.get("holdout_ids_path", "data/eval/holdout_cv_ids.json")
    p = ROOT / path if not os.path.isabs(path) else Path(path)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return [str(x) for x in data.get("holdout_cv_ids", [])]


def _wrap_lines(text: str, width: int = 90) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=width) or [""])
    return lines


def render_cv_png(cv_text: str, out_path: Path, title: str = "CV") -> None:
    from PIL import Image, ImageDraw, ImageFont

    lines = _wrap_lines(cv_text, width=85)
    max_lines = 55
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["... (truncated)"]

    font_size = 14
    line_height = 18
    margin = 24
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    img_h = margin * 2 + line_height * (len(lines) + 2)
    img_w = 620
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, margin), title, fill="black", font=title_font)
    y = margin + line_height * 2
    for line in lines:
        draw.text((margin, y), line, fill="black", font=font)
        y += line_height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    logger.info(f"PNG -> {out_path}")


def render_cv_pdf(cv_text: str, out_path: Path, title: str = "CV") -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, title, ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    for line in _wrap_lines(cv_text, width=95):
        safe = line.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 5, safe)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    logger.info(f"PDF -> {out_path}")


def prepare_holdout(
    resumes_path: str | None = None,
    holdout_dir: str | None = None,
    figures_dir: str | None = None,
) -> list[dict]:
    cfg = load_config().get("evaluation", {})
    resumes_path = resumes_path or cfg.get("batch_input", "data/eval/it_resumes.jsonl")
    holdout_dir = holdout_dir or "data/eval/holdout"
    figures_dir = figures_dir or "outputs/eval/figures"

    resumes_file = ROOT / resumes_path if not os.path.isabs(resumes_path) else Path(resumes_path)
    out_holdout = ROOT / holdout_dir
    out_figures = ROOT / figures_dir
    out_holdout.mkdir(parents=True, exist_ok=True)

    holdout_ids = set(_load_holdout_ids())
    by_id = {str(r["cv_id"]): r for r in load_jsonl(str(resumes_file))}

    manifest = []
    for cv_id in sorted(holdout_ids):
        rec = by_id.get(cv_id)
        if not rec:
            logger.warning(f"Hold-out CV {cv_id} not found in {resumes_file}")
            continue
        cv_text = rec.get("cv_text", "")
        txt_path = out_holdout / f"holdout_{cv_id}.txt"
        txt_path.write_text(cv_text, encoding="utf-8")

        png_path = out_figures / f"review_cv_{cv_id}.png"
        pdf_path = out_figures / f"review_cv_{cv_id}.pdf"
        title = f"IT Resume (ID {cv_id})"
        render_cv_png(cv_text, png_path, title=title)
        try:
            render_cv_pdf(cv_text, pdf_path, title=title)
        except Exception as exc:
            logger.warning(f"PDF skipped for {cv_id}: {exc}")

        manifest.append({
            "cv_id": cv_id,
            "txt_path": str(txt_path.relative_to(ROOT)),
            "png_path": str(png_path.relative_to(ROOT)),
            "pdf_path": str(pdf_path.relative_to(ROOT)),
            "job_description": rec.get("job_description", ""),
        })

    manifest_path = out_holdout / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Prepared {len(manifest)} hold-out CVs -> {out_holdout}")
    return manifest


def main():
    manifest = prepare_holdout()
    for m in manifest:
        print(f"  {m['cv_id']}: {m['txt_path']}, {m['png_path']}")
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())

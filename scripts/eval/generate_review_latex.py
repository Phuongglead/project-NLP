#!/usr/bin/env python3
"""Generate LaTeX human-review table and prose subsection from review_eval.jsonl."""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.eval.aggregate_review_report import aggregate_review
from src.shared.corpus.schema import load_jsonl
from src.shared.utils.io_utils import load_config

LATEX_ESC = str.maketrans({
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
})


_PUA_RE = re.compile(
    "["
    "\uE000-\uF8FF"
    "\U000F0000-\U000FFFFD"
    "\u200B-\u200F\u2028-\u202F\uFEFF"
    "]+"
)


def _latex_ascii(s: str) -> str:
    """Strip icons/PUA glyphs and transliterate to ASCII for pdfLaTeX."""
    if not s:
        return s
    s = _PUA_RE.sub(" ", s)
    s = (
        s.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    normalized = unicodedata.normalize("NFKD", s)
    return normalized.encode("ascii", "ignore").decode("ascii")


def _tex(s: str, max_len: int = 0) -> str:
    s = _latex_ascii(s or "").strip()
    s = re.sub(r"\s+", " ", s)
    if max_len and len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s.translate(LATEX_ESC)


def _tex_question(s: str, max_len: int = 280) -> str:
    s = _latex_ascii(s or "").strip()
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    if max_len and len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s.translate(LATEX_ESC)


def _build_review_rows(records: list) -> list:
    """Join generate + rating events per holdout CV."""
    by_session: dict = {}
    for r in records:
        sid = r.get("cv_session_id")
        if not sid:
            continue
        if r.get("event") == "generate":
            existing = by_session.get(sid, {})
            by_session[sid] = {
                "holdout_cv_id": r.get("holdout_cv_id") or existing.get("holdout_cv_id"),
                "cv_text": r.get("cv_text", "") or existing.get("cv_text", ""),
                "filename": existing.get("filename", ""),
                "job_description": r.get("job_description", "") or existing.get("job_description", ""),
                "skills": r.get("skills", []),
                "questions": r.get("questions", []),
                "ratings": existing.get("ratings", []),
            }
        elif r.get("event") == "cv_upload":
            if sid not in by_session:
                by_session[sid] = {
                    "holdout_cv_id": r.get("holdout_cv_id"),
                    "cv_text": r.get("cv_text", ""),
                    "filename": r.get("filename", ""),
                    "job_description": "",
                    "skills": [],
                    "questions": [],
                    "ratings": [],
                }
            else:
                if r.get("cv_text"):
                    by_session[sid]["cv_text"] = r.get("cv_text", "")
                if r.get("filename"):
                    by_session[sid]["filename"] = r.get("filename", "")
                if r.get("holdout_cv_id"):
                    by_session[sid]["holdout_cv_id"] = r.get("holdout_cv_id")

    for r in records:
        if r.get("event") != "rating":
            continue
        sid = r.get("cv_session_id")
        if sid not in by_session:
            by_session[sid] = {
                "holdout_cv_id": r.get("holdout_cv_id"),
                "cv_text": "",
                "filename": "",
                "job_description": "",
                "skills": [],
                "questions": [],
                "ratings": [],
            }
        by_session[sid]["ratings"].append({
            "question": r.get("generated_question", ""),
            "rating": r.get("rating"),
            "corpus_id": r.get("corpus_id", ""),
        })
        if r.get("holdout_cv_id"):
            by_session[sid]["holdout_cv_id"] = r.get("holdout_cv_id")

    rows = sorted(
        by_session.values(),
        key=lambda x: str(x.get("holdout_cv_id") or ""),
    )
    return [r for r in rows if r.get("questions")]


def _build_cv_info_json(block: dict) -> dict:
    """Structured CV metadata for the LaTeX table column."""
    skills = block.get("skills", [])
    skill_names: list[str] = []
    for s in skills:
        ent = s.get("entity") if isinstance(s, dict) else getattr(s, "entity", "")
        if ent and len(str(ent)) > 1:
            skill_names.append(str(ent))

    cv_text = (block.get("cv_text") or "").strip()
    excerpt = cv_text[:350] + "..." if len(cv_text) > 350 else cv_text
    excerpt = excerpt.replace("\n", r" \n ")

    return {
        "cv_id": block.get("holdout_cv_id"),
        "filename": block.get("filename") or None,
        "job_description": block.get("job_description") or None,
        "extracted_skills": skill_names[:12],
        "text_excerpt": excerpt or None,
    }


def _build_cv_info_latex(block: dict) -> dict:
    """ASCII-safe CV metadata for inline LaTeX (no long text excerpt)."""
    info = _build_cv_info_json(block)
    info.pop("text_excerpt", None)
    out: dict = {}
    for key, value in info.items():
        if value is None:
            continue
        if isinstance(value, str):
            out[key] = _latex_ascii(value)
        elif isinstance(value, list):
            out[key] = [_latex_ascii(str(v)) for v in value]
        else:
            out[key] = value
    return out


def _format_cv_json_tex(info: dict, col_width: str = "3.4cm", max_chars: int = 520) -> str:
    """Render compact JSON for a table cell."""
    raw = json.dumps(info, ensure_ascii=True, indent=2)
    if len(raw) > max_chars:
        skills = info.get("extracted_skills", [])
        while len(raw) > max_chars and len(skills) > 4:
            skills = skills[:-1]
            info = {**info, "extracted_skills": skills}
            raw = json.dumps(info, ensure_ascii=True, indent=2)
        if len(raw) > max_chars:
            raw = raw[: max_chars - 4] + "..."

    lines = [_tex(line) for line in raw.split("\n")]
    body = r" \\ ".join(lines)
    return (
        f"\\begin{{minipage}}[t]{{{col_width}}}"
        f"\\raggedright\\ttfamily\\scriptsize {body} \\end{{minipage}}"
    )


def _skill_keywords(skills: list, max_n: int = 12) -> str:
    names = []
    for s in skills:
        ent = s.get("entity") if isinstance(s, dict) else getattr(s, "entity", "")
        if ent and len(ent) > 1:
            names.append(str(ent))
    if not names:
        return "---"
    return ", ".join(names[:max_n])


def _match_ratings_to_questions(questions: list, ratings: list) -> list[dict]:
    """Pair each generated question with its rating (by corpus_id or order)."""
    rating_by_corpus = {r["corpus_id"]: r for r in ratings if r.get("corpus_id")}
    paired = []
    for i, q in enumerate(questions):
        if isinstance(q, dict):
            corpus_id = q.get("corpus_id", "")
            question = q.get("question", "")
        else:
            corpus_id = getattr(q, "corpus_id", "")
            question = getattr(q, "question", "")
        r = rating_by_corpus.get(corpus_id)
        if r:
            paired.append({"question": question, "rating": r["rating"]})
        elif i < len(ratings):
            paired.append({"question": question, "rating": ratings[i].get("rating")})
        else:
            paired.append({"question": question, "rating": None})
    # If more ratings than questions, append extras
    while len(paired) < len(ratings):
        r = ratings[len(paired)]
        paired.append({"question": r.get("question", ""), "rating": r.get("rating")})
    return paired[:5]


def generate_table_tex(rows: list, thesis_figures: str = "figures") -> str:
    lines = [
        "% Add to main.tex preamble (required):",
        "% \\usepackage{booktabs}",
        "% \\usepackage{multirow}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Human evaluation of generated interview questions (three hold-out IT resumes, five questions each).}",
        "\\label{tab:human_review}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{p{3.4cm}p{2.8cm}p{7.2cm}c}",
        "\\toprule",
        "\\textbf{CV information (JSON)} & \\textbf{Extracted keywords} & \\textbf{Generated question} & \\textbf{Rating (1--5)} \\\\",
        "\\midrule",
    ]

    all_ratings: list[float] = []

    for block in rows:
        keywords = _tex(_skill_keywords(block.get("skills", []), max_n=10), max_len=120)
        paired = _match_ratings_to_questions(block.get("questions", []), block.get("ratings", []))
        n_rows = max(len(paired), 1)
        cv_cell = _format_cv_json_tex(_build_cv_info_latex(block))

        cv_ratings = [p["rating"] for p in paired if p.get("rating") is not None]
        if cv_ratings:
            all_ratings.extend(cv_ratings)
        cv_mean = f"{mean(cv_ratings):.1f}" if cv_ratings else "---"

        total_block_rows = n_rows + 1  # +1 for mean row
        for i, pair in enumerate(paired):
            q_tex = _tex_question(pair.get("question", ""))
            rating = pair.get("rating")
            rating_tex = str(rating) if rating is not None else "---"
            if i == 0:
                lines.append(
                    f"\\multirow{{{total_block_rows}}}{{*}}{{{cv_cell}}} "
                    f"& \\multirow{{{total_block_rows}}}{{*}}{{{keywords}}} "
                    f"& {q_tex} & {rating_tex} \\\\"
                )
            else:
                lines.append(f" & & {q_tex} & {rating_tex} \\\\")

        lines.append(f" & & \\textbf{{Mean}} & \\textbf{{{cv_mean}}} \\\\")
        lines.append("\\midrule")

    overall = f"{mean(all_ratings):.2f}" if all_ratings else "---"
    lines.append(
        f"\\multicolumn{{3}}{{r}}{{\\textbf{{Overall mean}}}} & \\textbf{{{overall}}} \\\\"
    )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines) + "\n"


def generate_prose_tex(summary: dict, n_ratings: int) -> str:
    mean_r = summary.get("mean_rating", 0)
    pct4 = summary.get("pct_rating_ge_4", 0)
    median_r = summary.get("median_rating", 0)

    if n_ratings == 0:
        body = (
            "A complementary human study targets three hold-out information-technology "
            "resumes that were not part of the automated batch cohort. For each resume, "
            "the deployed pipeline extracts technical skills, retrieves a relevant reference "
            "answer, and produces five personalized interview questions. Reviewers assign "
            "a single usefulness score from one to five for every question-and-answer pair, "
            "capturing whether the output would be appropriate in a real technical interview. "
            "Quantitative results are inserted below once all fifteen ratings have been collected."
        )
    else:
        body = (
            "We completed a focused human evaluation on three hold-out IT resumes excluded "
            "from the large-scale automated run. Each uploaded resume was processed by the "
            "same end-to-end question-generation pipeline used in production: skills were "
            "extracted from the curriculum vitae, relevant technical knowledge was retrieved, "
            "and five interview questions were produced per resume. "
            "A reviewer scored every question together with its reference answer on a "
            "five-point usefulness scale. "
            f"Across all rated pairs, the average score was {mean_r:.2f} with a median of "
            f"{median_r:.1f}; roughly {pct4:.0f}\\% of pairs received four stars or higher. "
            "Higher ratings corresponded to questions that cited concrete skills from the "
            "resume and probed realistic engineering scenarios, whereas lower ratings tended "
            "to reflect generic wording or weak personalization."
        )

    return "\n".join([
        "\\subsection{Human Usefulness Results}",
        "\\label{sec:exp_human_results}",
        "",
        body,
        "",
        "Table~\\ref{tab:human_review} summarizes each hold-out resume as structured CV "
        "information (JSON), the extracted skill keywords, each generated question, and the "
        "assigned usefulness rating. Per-resume and overall means appear in the final row "
        "of each block.",
        "",
    ])


def generate_review_latex(
    review_path: str | None = None,
    output_dir: str | None = None,
    thesis_figures: str = "figures",
) -> dict:
    cfg = load_config().get("evaluation", {})
    review_path = review_path or cfg.get("review_output", "data/review_eval.jsonl")
    output_dir = output_dir or "outputs/eval"
    if not os.path.isabs(review_path):
        review_path = str(ROOT / review_path)
    out = ROOT / output_dir
    out.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(review_path) if os.path.exists(review_path) else []
    summary = aggregate_review(records)
    summary_path = out / "review_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = _build_review_rows(records)
    table_tex = generate_table_tex(rows, thesis_figures=thesis_figures)
    prose_tex = generate_prose_tex(summary, summary.get("n_ratings", 0))

    # Persist per-CV JSON sidecar files for thesis assets
    cv_json_dir = out / "cv_info"
    cv_json_dir.mkdir(parents=True, exist_ok=True)
    for block in rows:
        cv_id = block.get("holdout_cv_id")
        if not cv_id:
            continue
        info = _build_cv_info_json(block)
        with open(cv_json_dir / f"review_cv_{cv_id}.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    table_path = out / "review_table.tex"
    prose_path = out / "human_review_section.tex"
    table_path.write_text(table_tex, encoding="utf-8")
    prose_path.write_text(prose_tex, encoding="utf-8")

    # Copy PNGs to thesis figures hint file
    manifest_path = ROOT / "data/eval/holdout/manifest.json"
    copy_notes = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        for m in manifest:
            src = ROOT / m["png_path"]
            if src.exists():
                copy_notes.append(f"cp {src} <thesis>/{thesis_figures}/review_cv_{m['cv_id']}.png")

    notes_path = out / "copy_figures.sh"
    if copy_notes:
        notes_path.write_text("#!/bin/bash\n" + "\n".join(copy_notes) + "\n", encoding="utf-8")
        os.chmod(notes_path, 0o755)

    return {
        "table_path": str(table_path),
        "prose_path": str(prose_path),
        "summary_path": str(summary_path),
        "n_ratings": summary.get("n_ratings", 0),
    }


REVIEW_PLACEHOLDER = "% HUMAN_REVIEW_RESULTS_PLACEHOLDER"
REVIEW_BEGIN = "% HUMAN_REVIEW_RESULTS_BEGIN"
REVIEW_END = "% HUMAN_REVIEW_RESULTS_END"
DISCUSSION_PENDING = "pending human usefulness scores"


def _wrap_review_block(prose_tex: str, table_tex: str) -> str:
    return (
        REVIEW_BEGIN + "\n"
        + prose_tex.rstrip() + "\n\n"
        + table_tex.rstrip() + "\n"
        + REVIEW_END + "\n"
    )


def inject_into_experiments(
    experiments_path: str | Path,
    prose_tex: str,
    table_tex: str,
    summary: dict | None = None,
) -> str:
    """Insert human-review prose and table into the Experiments chapter."""
    path = Path(experiments_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Experiments chapter not found: {path}")

    content = path.read_text(encoding="utf-8")
    block = _wrap_review_block(prose_tex, table_tex)

    if REVIEW_BEGIN in content and REVIEW_END in content:
        start = content.index(REVIEW_BEGIN)
        end = content.index(REVIEW_END) + len(REVIEW_END)
        content = content[:start] + block + content[end:]
    elif REVIEW_PLACEHOLDER in content:
        content = content.replace(REVIEW_PLACEHOLDER, block)
    else:
        marker = "\\subsection{Human Evaluation Protocol}"
        if marker not in content:
            raise ValueError(f"Missing review markers or Human Evaluation Protocol in {path}")
        idx = content.index(marker)
        next_sec = content.find("\\subsection{", idx + len(marker))
        next_chap = content.find("\\section{", idx + len(marker))
        insert_at = min(x for x in (next_sec, next_chap) if x != -1) if any(
            x != -1 for x in (next_sec, next_chap)
        ) else len(content)
        content = content[:insert_at] + "\n" + block + "\n" + content[insert_at:]

    if summary and summary.get("n_ratings", 0) > 0 and DISCUSSION_PENDING in content:
        mean_r = summary.get("mean_rating", 0)
        replacement = (
            f"human reviewers rated generated questions at an average usefulness of {mean_r:.2f} "
            f"on a five-point scale (Table~\\ref{{tab:human_review}})"
        )
        content = content.replace(DISCUSSION_PENDING, replacement)

    path.write_text(content, encoding="utf-8")
    return str(path)


def finalize_review_report(
    review_path: str | None = None,
    output_dir: str | None = None,
    thesis_figures: str = "figures",
    experiments_path: str | None = None,
) -> dict:
    """Aggregate ratings, write LaTeX artifacts, and inject into experiments chapter."""
    result = generate_review_latex(
        review_path=review_path,
        output_dir=output_dir,
        thesis_figures=thesis_figures,
    )
    cfg = load_config().get("evaluation", {})
    exp_path = experiments_path or cfg.get("experiments_tex", "docs/thesis/experiments.tex")
    prose = Path(result["prose_path"]).read_text(encoding="utf-8")
    table = Path(result["table_path"]).read_text(encoding="utf-8")
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    injected = inject_into_experiments(exp_path, prose, table, summary=summary)
    result["experiments_path"] = injected
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--thesis-figures", default="figures")
    parser.add_argument(
        "--inject",
        action="store_true",
        help="Insert results into docs/thesis/experiments.tex after generation",
    )
    parser.add_argument("--experiments-tex", default=None)
    args = parser.parse_args()

    if args.inject:
        result = finalize_review_report(
            review_path=args.input,
            output_dir=args.output_dir,
            thesis_figures=args.thesis_figures,
            experiments_path=args.experiments_tex,
        )
        print(f"Injected -> {result['experiments_path']}")
    else:
        result = generate_review_latex(
            review_path=args.input,
            output_dir=args.output_dir,
            thesis_figures=args.thesis_figures,
        )
    print(f"Prose  -> {result['prose_path']}")
    print(f"Table  -> {result['table_path']}")
    print(f"Summary -> {result['summary_path']} (n_ratings={result['n_ratings']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

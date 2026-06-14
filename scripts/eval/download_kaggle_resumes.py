#!/usr/bin/env python3
"""Download Kaggle resume dataset and export IT-category eval JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.shared.corpus.schema import load_jsonl, write_jsonl
from src.shared.utils.io_utils import get_logger, load_config

logger = get_logger("download_kaggle")


def _setup_kaggle_credentials(creds_path: Path) -> None:
    with open(creds_path, "r", encoding="utf-8") as f:
        creds = json.load(f)
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]


def _find_resume_csv(dataset_dir: Path) -> Path | None:
    for pattern in ("*.csv", "**/*.csv"):
        for p in dataset_dir.glob(pattern):
            if p.is_file() and "resume" in p.name.lower():
                return p
    csv_files = list(dataset_dir.glob("**/*.csv"))
    return csv_files[0] if csv_files else None


def download_dataset(dataset_slug: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    creds_path = ROOT / "data" / "kaggle.json"
    if creds_path.exists():
        _setup_kaggle_credentials(creds_path)
    else:
        logger.warning("data/kaggle.json not found — expecting dataset already on disk.")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading {dataset_slug} -> {dest_dir}")
        api.dataset_download_files(dataset_slug, path=str(dest_dir), unzip=True)
    except ImportError:
        logger.warning("kaggle package not installed; pip install kaggle")
    except Exception as exc:
        logger.warning(f"Kaggle download failed ({exc}); using existing files if present.")

    csv_path = _find_resume_csv(dest_dir)
    if not csv_path:
        raise FileNotFoundError(f"No CSV found under {dest_dir}. Run with valid kaggle.json.")
    return csv_path


def export_it_resumes(
    csv_path: Path,
    output_path: Path,
    category: str,
    job_description: str,
) -> int:
    import pandas as pd

    df = pd.read_csv(csv_path)
    cat_col = "Category" if "Category" in df.columns else "category"
    id_col = "ID" if "ID" in df.columns else "id"
    text_col = "Resume_str" if "Resume_str" in df.columns else "resume_str"

    cat_norm = category.strip().upper()
    subset = df[df[cat_col].astype(str).str.strip().str.upper() == cat_norm].copy()
    records = []
    for _, row in subset.iterrows():
        cv_text = str(row.get(text_col, "") or "").strip()
        if len(cv_text) < 50:
            continue
        records.append({
            "cv_id": str(row.get(id_col, len(records))),
            "category": category,
            "cv_text": cv_text,
            "job_description": job_description,
        })

    write_jsonl(records, str(output_path))
    logger.info(f"Exported {len(records)} IT resumes -> {output_path}")
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    cfg = load_config().get("evaluation", {})
    dataset_slug = cfg.get("kaggle_dataset", "snehaanbhawal/resume-dataset")
    category = args.category or cfg.get("it_category", "Information-Technology")
    job_desc = cfg.get(
        "default_job_description",
        "Information-Technology software engineering role.",
    )
    output = args.output or cfg.get("batch_input", "data/eval/it_resumes.jsonl")
    dataset_dir = ROOT / "data" / "kaggle" / "resume-dataset"
    output_path = ROOT / output if not os.path.isabs(output) else Path(output)

    if not args.skip_download:
        csv_path = download_dataset(dataset_slug, dataset_dir)
    else:
        csv_path = _find_resume_csv(dataset_dir)
        if not csv_path:
            raise FileNotFoundError(f"No CSV in {dataset_dir}")

    n = export_it_resumes(csv_path, output_path, category, job_desc)

    holdout_path = cfg.get("holdout_ids_path", "data/eval/holdout_cv_ids.json")
    holdout_file = ROOT / holdout_path if not os.path.isabs(holdout_path) else Path(holdout_path)
    holdout_file.parent.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(str(output_path))
    holdout_ids = [r["cv_id"] for r in records[-3:]] if len(records) >= 3 else []
    with open(holdout_file, "w", encoding="utf-8") as f:
        json.dump({"holdout_cv_ids": holdout_ids, "note": "Use these 3 CVs for human REVIEW_MODE eval"}, f, indent=2)
    logger.info(f"Hold-out CV IDs ({len(holdout_ids)}) -> {holdout_file}")

    print(f"Exported {n} resumes to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

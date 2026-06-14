#!/usr/bin/env python3
"""Upload hold-out CVs and generate 5 questions each via the running API."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.shared.utils.io_utils import get_logger, load_config

logger = get_logger("run_holdout_review")

DEFAULT_API = "http://127.0.0.1:8000"


def _multipart_upload(url: str, file_path: Path) -> dict:
    boundary = "----HoldoutBoundary7MA4YWxk"
    content = file_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_holdout_review(
    api_base: str = DEFAULT_API,
    num_questions: int = 5,
    holdout_dir: str | None = None,
) -> list[dict]:
    holdout_dir = holdout_dir or "data/eval/holdout"
    base = api_base.rstrip("/")
    manifest_path = ROOT / holdout_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run eval-prepare-holdout first: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    results = []
    for entry in manifest:
        cv_id = entry["cv_id"]
        txt_path = ROOT / entry["txt_path"]
        logger.info(f"Processing hold-out CV {cv_id}...")

        upload = _multipart_upload(f"{base}/api/interview/upload-cv", txt_path)
        session_id = upload["cv_session_id"]

        gen = _post_json(
            f"{base}/api/interview/generate",
            {
                "specialization": "backend",
                "experience_level": "middle",
                "mode": "mixed",
                "cv_session_id": session_id,
                "num_questions": num_questions,
                "custom": {
                    "job_description": entry.get(
                        "job_description",
                        "Information-Technology software engineering role.",
                    ),
                },
            },
        )
        n_q = len(gen.get("questions", []))
        logger.info(f"  session={session_id} questions={n_q} review_mode={gen.get('review_mode')}")
        results.append({
            "cv_id": cv_id,
            "cv_session_id": session_id,
            "n_questions": n_q,
            "review_mode": gen.get("review_mode", False),
        })

    out_path = ROOT / "outputs/eval/holdout_sessions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nHold-out review generation complete.")
    print("Next: open the UI, upload is already done — use these session IDs if needed:")
    for r in results:
        print(f"  CV {r['cv_id']}: session {r['cv_session_id']} ({r['n_questions']} questions)")
    print("\nRate all 15 Q&A pairs in the browser (1–5 stars each).")
    print(f"Sessions saved -> {out_path}")
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=os.getenv("SA_AQG_API_BASE", DEFAULT_API))
    parser.add_argument("--num-questions", type=int, default=5)
    args = parser.parse_args()

    try:
        run_holdout_review(api_base=args.api, num_questions=args.num_questions)
    except urllib.error.URLError as exc:
        logger.error(f"API not reachable at {args.api}: {exc}")
        logger.error("Start: REVIEW_MODE=true uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

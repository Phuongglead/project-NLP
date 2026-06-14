#!/usr/bin/env python3
"""Remove template-fallback records so they can be re-evaluated with Gemini."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "outputs/eval/batch_records.jsonl"
FALLBACK_PREFIX = "Based on your experience with"

records = []
removed = []
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    q = rec.get("generated_question", "")
    if q.startswith(FALLBACK_PREFIX):
        removed.append(rec.get("cv_id"))
        continue
    records.append(line)

path.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")
print(f"Kept {len(records)} records, removed {len(removed)} fallback CVs: {removed}")

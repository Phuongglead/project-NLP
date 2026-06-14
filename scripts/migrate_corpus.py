#!/usr/bin/env python3
"""Migrate reference_answers.jsonl to knowledge_corpus.active.jsonl."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.shared.corpus.schema import (
    legacy_to_knowledge_record,
    load_jsonl,
    write_jsonl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/reference_answers.jsonl")
    parser.add_argument("--output", default="data/knowledge_corpus.active.jsonl")
    parser.add_argument("--use-ner", action="store_true", help="Run JobBERT NER for keywords")
    args = parser.parse_args()

    ner_fn = None
    if args.use_ner:
        from src.core.NER.ner_module import skill_extract
        ner_fn = skill_extract

    input_path = os.path.join(ROOT, args.input)
    output_path = os.path.join(ROOT, args.output)

    legacy_records = load_jsonl(input_path)
    if not legacy_records:
        print(f"No records found at {input_path}")
        return 1

    migrated = [legacy_to_knowledge_record(r, ner_fn=ner_fn) for r in legacy_records]
    write_jsonl(migrated, output_path)
    print(f"Migrated {len(migrated)} records -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

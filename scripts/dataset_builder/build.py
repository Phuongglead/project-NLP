#!/usr/bin/env python3
"""
Dataset builder: crawl StackOverflow → transform → append to staging JSONL.
Standalone — not imported by runtime API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.dataset_builder.api_limits import RateLimitExhausted
from scripts.dataset_builder.crawl import (
    TAG_ROTATIONS,
    fetch_accepted_answer,
    fetch_questions,
    fetch_questions_page,
    quota_snapshot,
    reset_client,
)
from scripts.dataset_builder.transform import so_to_knowledge_record
from src.shared.corpus.schema import dedup_hash, load_jsonl, write_jsonl
from src.shared.utils.io_utils import get_logger, load_config

logger = get_logger("dataset_builder")


def _existing_dedup_keys(corpus_path: str) -> set:
    keys = set()
    for rec in load_jsonl(corpus_path):
        keys.add(dedup_hash(rec))
        url = (rec.get("source") or {}).get("url", "")
        if url:
            keys.add(url)
    return keys


def run_builder(
    output_path: str,
    pages: int = 1,
    tags: list = None,
    use_ner: bool = False,
    base_corpus: str = None,
    max_records: int = None,
    max_minutes: float = None,
) -> int:
    ner_fn = None
    if use_ner:
        from src.core.NER.ner_module import skill_extract
        ner_fn = skill_extract

    records = load_jsonl(base_corpus) if base_corpus and os.path.exists(base_corpus) else []
    seen = _existing_dedup_keys(base_corpus or output_path)
    for r in records:
        seen.add(dedup_hash(r))

    start = time.time()
    deadline = start + max_minutes * 60 if max_minutes else None
    target = max_records
    added = 0

    def _should_stop() -> bool:
        if target and len(records) >= target:
            logger.info(f"Reached max_records={target}")
            return True
        if deadline and time.time() >= deadline:
            logger.info(f"Reached max_minutes={max_minutes}")
            return True
        return False

    def _try_add_question(q: dict) -> bool:
        nonlocal added
        url = q.get("link", "")
        if url in seen:
            return False
        try:
            answer = fetch_accepted_answer(q)
        except RateLimitExhausted:
            raise
        if not answer:
            return False
        record = so_to_knowledge_record(q, answer, ner_fn=ner_fn)
        if not record:
            return False
        h = record.pop("_dedup_hash", dedup_hash(record))
        if h in seen:
            return False
        records.append(record)
        seen.add(h)
        if url:
            seen.add(url)
        added += 1
        if added % 10 == 0:
            write_jsonl(records, output_path)
            logger.info(f"Checkpoint: {len(records)} total records ({added} new)")
        else:
            logger.info(f"Added [{len(records)}]: {record['skill']} / {record['topic']}")
        return True

    state_path = output_path.replace(".jsonl", "_crawl_state.json")

    def _load_state() -> dict:
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"page_by_tag": {}, "rotation_idx": 0, "quota": {}}

    def _save_state(state: dict) -> None:
        state["quota"] = quota_snapshot()
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    rate_limited = False

    try:
        # Timed / capped crawl: page through tag rotations until limit hit
        if max_records or max_minutes:
            tag_sets = [tags] if tags else TAG_ROTATIONS
            state = _load_state()
            rotation_idx = state.get("rotation_idx", 0)
            page_by_tag: dict[str, int] = state.get("page_by_tag", {})
            consecutive_empty = 0

            while not _should_stop():
                current_tags = tag_sets[rotation_idx % len(tag_sets)]
                tag_key = ";".join(current_tags)
                page = page_by_tag.get(tag_key, 1)

                try:
                    items, has_more = fetch_questions_page(page, current_tags, page_size=100)
                except RateLimitExhausted as exc:
                    logger.warning(f"Rate limit hit during question fetch: {exc}")
                    rate_limited = True
                    state["page_by_tag"] = page_by_tag
                    state["rotation_idx"] = rotation_idx
                    _save_state(state)
                    break

                if not items:
                    consecutive_empty += 1
                    rotation_idx += 1
                    state["rotation_idx"] = rotation_idx
                    _save_state(state)
                    if consecutive_empty >= len(tag_sets):
                        logger.info("Exhausted all tag rotations.")
                        break
                    continue

                consecutive_empty = 0
                for q in items:
                    if _should_stop():
                        break
                    try:
                        _try_add_question(q)
                    except RateLimitExhausted as exc:
                        logger.warning(f"Rate limit hit during answer fetch: {exc}")
                        rate_limited = True
                        break

                if rate_limited:
                    state["page_by_tag"] = page_by_tag
                    state["rotation_idx"] = rotation_idx
                    _save_state(state)
                    break

                if has_more:
                    page_by_tag[tag_key] = page + 1
                else:
                    rotation_idx += 1

                state["page_by_tag"] = page_by_tag
                state["rotation_idx"] = rotation_idx
                _save_state(state)

                if _should_stop():
                    break

        else:
            # Legacy single-batch mode
            questions = fetch_questions(tags=tags, pages=pages)
            for q in questions:
                try:
                    _try_add_question(q)
                except RateLimitExhausted as exc:
                    logger.warning(f"Rate limit hit: {exc}")
                    rate_limited = True
                    break
    finally:
        reset_client()

    write_jsonl(records, output_path)
    elapsed = time.time() - start
    stop_reason = "complete"
    if rate_limited:
        stop_reason = "rate_limited"
    elif target and len(records) >= target:
        stop_reason = f"max_records={target}"
    elif deadline and time.time() >= deadline:
        stop_reason = f"max_minutes={max_minutes}"
    logger.info(
        f"Builder done ({stop_reason}): {added} new, {len(records)} total in {elapsed:.0f}s -> {output_path}"
    )
    if rate_limited:
        logger.info(
            "Crawl paused due to API limits. Resume with the same --output/--base; "
            "state saved in *_crawl_state.json"
        )
    return added


def main():
    parser = argparse.ArgumentParser(description="Build knowledge corpus from StackOverflow")
    parser.add_argument("--output", default=None, help="Staging JSONL output path")
    parser.add_argument("--base", default=None, help="Base corpus to copy/extend")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--use-ner", action="store_true")
    parser.add_argument("--max-records", type=int, default=None, help="Stop when corpus reaches N records")
    parser.add_argument("--max-minutes", type=float, default=None, help="Stop after N minutes")
    args = parser.parse_args()

    cfg = load_config()["rag"]
    output = args.output or cfg.get("staging_corpus_path", "data/knowledge_corpus.staging.jsonl")
    base = args.base or cfg.get("reference_corpus_path", "data/knowledge_corpus.active.jsonl")
    if not os.path.isabs(output):
        output = os.path.join(ROOT, output)
    if not os.path.isabs(base):
        base = os.path.join(ROOT, base)

    # Ensure staging starts from active if staging empty/small
    if not os.path.exists(output) or os.path.getsize(output) == 0:
        if os.path.exists(base):
            import shutil
            shutil.copy(base, output)

    added = run_builder(
        output,
        pages=args.pages,
        tags=args.tags,
        use_ner=args.use_ner,
        base_corpus=output,
        max_records=args.max_records,
        max_minutes=args.max_minutes,
    )
    print(f"Added {added} records -> {output} (total lines in file)")
    total = len(load_jsonl(output))
    print(f"Total records: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# Nightly blue-green corpus update (02:00–05:00 window).
# Runtime API reads *.active files only; this script updates staging then swaps.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Use sa-aqg conda environment
if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate sa-aqg
fi

ACTIVE_CORPUS="data/knowledge_corpus.active.jsonl"
STAGING_CORPUS="data/knowledge_corpus.staging.jsonl"
ACTIVE_INDEX="models/faiss_index.active"
STAGING_INDEX="models/faiss_index.staging"
MANIFEST="data/corpus_manifest.json"
FEEDBACK="data/feedback.jsonl"

echo "[$(date -Iseconds)] Nightly corpus update started"

# 1. Copy active corpus as staging base
if [[ -f "$ACTIVE_CORPUS" ]]; then
  cp "$ACTIVE_CORPUS" "$STAGING_CORPUS"
  echo "Copied active corpus -> staging"
else
  echo "No active corpus; creating empty staging"
  touch "$STAGING_CORPUS"
fi

# 2. Curator: promote feedback questions into staging
python scripts/dataset_builder/curator.py \
  --corpus "$STAGING_CORPUS" \
  --feedback "$FEEDBACK" || echo "Curator skipped or failed (non-fatal)"

# 3. Crawl new StackOverflow records into staging
python scripts/dataset_builder/build.py \
  --output "$STAGING_CORPUS" \
  --base "$STAGING_CORPUS" \
  --pages 1 || echo "Builder skipped or failed (non-fatal)"

# 4. Build FAISS index on staging
python -c "
from src.core.rag_retriever.rag_module import build_faiss_index
build_faiss_index(corpus_path='$STAGING_CORPUS', index_path='$STAGING_INDEX')
print('Staging FAISS index built')
"

# 5. Atomic swap at end of window
RECORD_COUNT=$(wc -l < "$STAGING_CORPUS" | tr -d ' ')
SWAPPED_AT=$(date -Iseconds)

cp "$STAGING_CORPUS" "$ACTIVE_CORPUS"
cp "${STAGING_INDEX}.faiss" "${ACTIVE_INDEX}.faiss"
cp "${STAGING_INDEX}_texts.json" "${ACTIVE_INDEX}_texts.json"

python -c "
import json
manifest = {
    'version': '$SWAPPED_AT',
    'record_count': int('$RECORD_COUNT'),
    'swapped_at': '$SWAPPED_AT',
    'active_corpus': '$ACTIVE_CORPUS',
    'active_index': '$ACTIVE_INDEX',
}
with open('$MANIFEST', 'w') as f:
    json.dump(manifest, f, indent=2)
print('Manifest written')
"

echo "[$(date -Iseconds)] Swap complete: $RECORD_COUNT records"

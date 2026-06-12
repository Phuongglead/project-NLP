#!/usr/bin/env bash
# Interactive shell in bootstrap container for manual debugging.

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-rag_system-backend:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-sa-aqg-shell}"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run -it --rm --name "$CONTAINER_NAME" \
  -v "$PROJECT_ROOT:/workspace" \
  -w /workspace \
  -p 18000:8000 \
  "$BASE_IMAGE" \
  bash

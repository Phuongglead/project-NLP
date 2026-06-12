#!/usr/bin/env bash
# Bootstrap SA-AQG backend image by iterating inside a prebuilt container.
# Usage (from WSL, project root):
#   sed -i 's/\r$//' scripts/docker-bootstrap.sh   # if edited on Windows
#   chmod +x scripts/docker-bootstrap.sh
#   ./scripts/docker-bootstrap.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_IMAGE="${BASE_IMAGE:-rag_system-backend:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-sa-aqg-bootstrap}"
COMMIT_TAG="${COMMIT_TAG:-sa-aqg-backend:dev}"

echo "==> Base image: $BASE_IMAGE"
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
  echo "Base image not found. Build rag_system backend first:"
  echo "  cd ../rag_system && docker compose build backend"
  exit 1
}

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "==> Starting bootstrap container (project mounted at /workspace)"
GPU_ARGS=()
if [ "${USE_GPU:-auto}" != "false" ]; then
  if docker run --rm --gpus all "$BASE_IMAGE" true 2>/dev/null; then
    GPU_ARGS=(--gpus all -e NVIDIA_VISIBLE_DEVICES=all)
    echo "    GPU: enabled"
  else
    echo "    GPU: unavailable — install nvidia-container-toolkit. CPU-only for now."
  fi
fi
docker run -d --name "$CONTAINER_NAME" \
  "${GPU_ARGS[@]}" \
  -v "$PROJECT_ROOT:/workspace" \
  -w /workspace \
  -p 18000:8000 \
  "$BASE_IMAGE" \
  sleep infinity

install_missing() {
  docker exec "$CONTAINER_NAME" bash -lc "$1"
}

echo "==> Installing minimal deps (base image already has fastapi, torch, sentence-transformers)"
install_missing "cd /tmp && pip install --no-cache-dir -r /workspace/requirements-bootstrap.txt"

echo "==> Copy project into /app"
install_missing "rm -rf /app && mkdir -p /app && cp -r /workspace/src /workspace/api /workspace/config /workspace/main.py /workspace/test.py /app/ && cp -r /workspace/models /workspace/data /app/ 2>/dev/null || true && mkdir -p /app/best_model /app/models /app/data"

echo "==> Smoke test: stub pipeline"
install_missing "export SA_AQG_USE_STUBS=true PYTHONPATH=/app && cd /app && python test.py"

echo "==> Smoke test: API health"
install_missing "pkill -f 'uvicorn api.main' 2>/dev/null || true"
install_missing "export SA_AQG_USE_STUBS=true PYTHONPATH=/app && cd /app && nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 >/tmp/uvicorn.log 2>&1 &"
sleep 5
curl -sf http://127.0.0.1:18000/api/health || {
  echo "API health failed. Logs:"
  docker exec "$CONTAINER_NAME" cat /tmp/uvicorn.log || true
  echo "Debug: ./scripts/docker-shell.sh then docker commit $CONTAINER_NAME $COMMIT_TAG"
  exit 1
}
echo "API health OK"

echo "==> Committing image as $COMMIT_TAG"
docker commit \
  -c 'WORKDIR /app' \
  -c 'ENV PYTHONPATH=/app PYTHONUNBUFFERED=1' \
  -c 'EXPOSE 8000' \
  -c 'CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]' \
  "$CONTAINER_NAME" "$COMMIT_TAG"

docker rm -f "$CONTAINER_NAME"

echo ""
echo "Done. Image: $COMMIT_TAG"
echo "CPU:  docker compose up -d"
echo "GPU:  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d"

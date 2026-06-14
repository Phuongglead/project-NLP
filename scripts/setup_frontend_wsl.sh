#!/usr/bin/env bash
# Install frontend deps on the Linux filesystem and symlink node_modules.
# Required when the repo lives on a Windows drive (/mnt/c, /mnt/h) under WSL:
# npm native binaries (esbuild) are corrupted on drvfs mounts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND="$ROOT/frontend"
MODULES_DIR="${SA_AQG_FRONTEND_MODULES:-$HOME/.cv-iqg-frontend-modules}"

if [[ "$FRONTEND" == /mnt/* ]]; then
  echo "WSL + Windows drive detected: installing deps under $MODULES_DIR"
fi

mkdir -p "$MODULES_DIR"
cp "$FRONTEND/package.json" "$MODULES_DIR/"
if [[ -f "$FRONTEND/package-lock.json" ]]; then
  cp "$FRONTEND/package-lock.json" "$MODULES_DIR/"
fi

(
  cd "$MODULES_DIR"
  npm install --legacy-peer-deps
)

rm -rf "$FRONTEND/node_modules"
ln -sfn "$MODULES_DIR/node_modules" "$FRONTEND/node_modules"

if [[ ! -f "$FRONTEND/package-lock.json" && -f "$MODULES_DIR/package-lock.json" ]]; then
  cp "$MODULES_DIR/package-lock.json" "$FRONTEND/package-lock.json"
fi

echo "Frontend ready. Run: cd frontend && npm run dev"

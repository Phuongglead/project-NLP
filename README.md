# SA-AQG — Skill-Aware Answer-Aware Question Generation

Generate personalized technical interview questions from a CV and job description using a four-stage NLP pipeline: **NER** (JobBERT) → **RAG** (FAISS) → **Gemini** (answer-aware QG) → **XAI** evaluation (NLI, ALCE, SHAP).

For architecture and module details, see [IMPLEMENTATION_DETAIL.md](IMPLEMENTATION_DETAIL.md), [IMPLEMENTATION.md](IMPLEMENTATION.md), and [Method.md](Method.md).

---

## Remote API + server-side fallbacks

The UI can target a **remote FastAPI server** (e.g. GPU machine on your LAN).

### Configure API host and port

**File (default for all users):** edit [`frontend/public/app-config.json`](frontend/public/app-config.json):

```json
{
  "apiHost": "192.168.1.198",
  "apiPort": 1408,
  "useRelativeApi": false
}
```

| Field | Default | Meaning |
|-------|---------|---------|
| `apiHost` | `192.168.1.198` | IP or hostname of the machine running `uvicorn` |
| `apiPort` | `1408` | FastAPI port |
| `useRelativeApi` | `false` | Set `true` when using Docker/Nginx (`/api` proxy) |

**Runtime override:** open the UI → **API server settings** → set host/port → **Apply** (stored in browser `localStorage`).

**Dev proxy:** `npm run dev` proxies `/api` when `useRelativeApi` is true. For direct remote calls, keep `useRelativeApi: false`.

### Step-separated fallbacks (server, log only)

Handled in the pipeline — **nothing is shown in the UI**; check API logs for `[fallback]`:

| Step | When | Action |
|------|------|--------|
| **NER** | JobBERT/CUDA fails | Use ee02e203 holdout keywords; RAG + Gemini continue normally |
| **Gemini** | API keys missing/rejected | Use five cached holdout questions (`src/shared/demo_fallback.py`) |

The UI shows a brief loading spinner during requests; no fallback banner or error message.

---

## Remote server: demo in 5 minutes (Phase A)

**Goal:** Upload one CV → get **one** suggested interview question.

### 1. Clone and setup

```bash
git clone https://github.com/Phuongglead/project-NLP.git
cd project-NLP
git lfs install && git lfs pull
cp .env.example .env
# Edit .env: GOOGLE_GEMINI_API_KEY=your_key
```

### 2. Python environment

```bash
conda create -n sa-aqg python=3.10 -y && conda activate sa-aqg
pip install -r requirements.txt
pip install faiss-cpu sentence-transformers email-validator   # if missing
python main.py build-index
```

### 3. Demo — CLI (one command)

```bash
chmod +x scripts/run_demo.sh
./scripts/run_demo.sh                    # stub mode (no API key)
SA_AQG_USE_STUBS=false ./scripts/run_demo.sh   # real NER + Gemini
```

Uses [`data/sample_cv.txt`](data/sample_cv.txt) and prints one generated question.

### 4. Demo — API (upload CV + generate)

**Terminal 1:**

```bash
export SA_AQG_USE_STUBS=false   # or true for stub
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2:**

```bash
python scripts/test_api.py
```

Expect: `questions returned: 1` and the question text printed.

**Manual curl:**

```bash
curl -F "file=@data/sample_cv.txt" http://localhost:8000/api/interview/upload-cv
# Then POST /api/interview/generate with cv_session_id and "num_questions": 1
```

### 5. Demo — UI (optional)

```bash
# Terminal 1: API (as above)
# Terminal 2:
cd frontend && npm install --legacy-peer-deps && npm run dev
```

Open http://localhost:3000 → upload CV → **Number of questions = 1** → Generate.

### Phase A checklist

- [ ] `python main.py build-index` succeeds
- [ ] `./scripts/run_demo.sh` prints one question
- [ ] `python scripts/test_api.py` returns exactly 1 question
- [ ] UI shows one question card after generate

> **Phase B (later):** SkillSpan batch evaluation (5 / 50 samples), NLI/ALCE metrics — see plan `skillspan_eval_pipeline`.

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ | Conda `mlops` env recommended (WSL) |
| Node.js | 18+ | For local UI dev (`npm run dev`) |
| Docker | WSL mode | For containerized run; set data-root on H: if C: is full |
| Gemini API key | — | [Google AI Studio](https://aistudio.google.com/apikey) |
| Git LFS | — | Required to pull `best_model/model.safetensors` |

---

## Quick Start (Local — Recommended First)

Run from the **project root** (`project-NLP/`).

### 1. Environment

```bash
cp .env.example .env
# Edit .env — set GOOGLE_GEMINI_API_KEY=your_key
```

### 2. Install dependencies

```bash
# WSL example
conda activate mlops
pip install -r requirements.txt
```

If `faiss` or `sentence-transformers` are missing in your env:

```bash
pip install faiss-cpu sentence-transformers email-validator
```

### 3. Pull NER model weights (required for real runs)

```bash
git lfs install
git lfs pull
```

Without LFS, use stub mode (step 4a) or the API/UI with `SA_AQG_USE_STUBS=true`.

### 4a. Stub smoke test (no GPU / API)

```bash
SA_AQG_USE_STUBS=true python test.py
```

### 4b. Build RAG index

```bash
python main.py build-index
```

Creates `data/reference_answers.jsonl` (sample corpus) and `models/faiss_index.faiss`.

### 4c. CLI single run

```bash
export SA_AQG_USE_STUBS=false
python main.py run \
  --cv "5 years Kubernetes and Docker experience" \
  --job "Senior DevOps Engineer" \
  --no-shap
```

### 5. API + UI (development)

**Terminal 1 — API:**

```bash
export SA_AQG_USE_STUBS=true   # or false when model + key are ready
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — UI:**

```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
```

Open **http://localhost:3000**. By default the UI calls `http://192.168.1.198:1408/api` (see [Remote API + server-side fallbacks](#remote-api--server-side-fallbacks)). For a local API on the same machine, set `apiHost` to `localhost` in `frontend/public/app-config.json` or use the in-app **API server settings** panel.

**API smoke test:**

```bash
python scripts/test_api.py
```

---

## Docker Quick Start

**Recommended:** bootstrap from an existing GPU image, iterate inside the container, then `docker commit`. Much faster than building PyTorch from scratch.

### Step 1 — Bootstrap backend image (WSL)

Uses `rag_system-backend:latest` if present (already has sentence-transformers + FastAPI). Override base:

```bash
cd /mnt/h/Dev/University/CV-IQG/project-NLP
cp .env.example .env   # set GOOGLE_GEMINI_API_KEY, SA_AQG_USE_STUBS=true for first test

chmod +x scripts/docker-bootstrap.sh scripts/docker-shell.sh
./scripts/docker-bootstrap.sh
```

Manual debug shell (install missing packages, run tests, then commit yourself):

```bash
./scripts/docker-shell.sh
# inside container:
export SA_AQG_USE_STUBS=true PYTHONPATH=/app
pip install -r requirements-docker.txt email-validator google-genai
python test.py
uvicorn api.main:app --host 0.0.0.0 --port 8000
# from another terminal: curl http://localhost:18000/api/health
# exit and: docker commit sa-aqg-shell sa-aqg-backend:dev
```

### Step 2 — Run stack

```bash
docker compose up -d
curl http://localhost:8000/api/health
# UI: http://localhost:3000
docker compose down
```

### Optional: build from Dockerfile (later)

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml build backend
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000/api/docs |

**Build note:** Backend uses `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (GPU). First pull is ~5–6 GB; `gpus: all` in compose requires NVIDIA Container Toolkit in WSL.

**GPU check:** `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

**Stub mode in Docker:** Set `SA_AQG_USE_STUBS=true` in `.env` to run without NER weights or Gemini calls.

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `python main.py run --cv "..." --job "..."` | Single pipeline run |
| `python main.py batch --input data/eval_samples.jsonl` | Batch evaluation |
| `python main.py create-data --n 10` | Create sample eval JSONL |
| `python main.py build-index` | Build FAISS RAG index |
| `python main.py eval-nli --input outputs/results.jsonl` | NLI hallucination eval |
| `python main.py eval-xai --input outputs/results.jsonl` | XAI batch eval |

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GOOGLE_GEMINI_API_KEY` | — | Gemini API key (primary) |
| `GEMINI_API_KEY` | — | Fallback alias |
| `SA_AQG_USE_STUBS` | `false` | Use stub modules instead of real ML |
| `CORS_ALLOW_ORIGINS` | `["http://localhost:3000",...]` | Explicit FastAPI CORS origins |
| `CORS_ALLOW_ORIGIN_REGEX` | `192.168.*.*` pattern (see `api/config.py`) | LAN UI hosts, e.g. `http://192.168.1.209:3000` |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FAISS index not found` | Run `python main.py build-index` |
| `header too large` on NER load | Run `git lfs pull` for `best_model/` |
| `GOOGLE_GEMINI_API_KEY not set` | Copy `.env.example` → `.env` and set key |
| Pipeline still uses stubs | `.env` may set `SA_AQG_USE_STUBS=true` — export `false` explicitly |
| UI **Network Error** on Generate | Browser on LAN (`192.168.x.x:3000`) blocked by CORS — restart API after pull; default regex now allows `192.168.*.*`. Check server log for `OPTIONS ... 400`. |
| Docker build fills C: / daemon EOF | Use `./scripts/docker-bootstrap.sh` instead of full build |
| WSL catastrophic failure | `wsl --shutdown` in PowerShell, wait 10s, reopen Ubuntu |
| `sa-aqg-backend:dev` not found | Run `./scripts/docker-bootstrap.sh` first |
| `email-validator` import error | `pip install email-validator` |

---

## Security

- **Never commit `.env`** — it is listed in `.gitignore`
- Rotate any API key that was ever committed or shared
- Use stub mode for public demos without exposing keys

---

## Manual Steps (cannot be automated)

1. Create and set `GOOGLE_GEMINI_API_KEY` in `.env`
2. `git lfs pull` for NER weights
3. Configure WSL Docker `data-root` on H: if C: is full
4. `docker login` if pushing images to Docker Hub
5. Install Node.js 18+ for local `npm run dev`

---

## Git hygiene

`.gitignore` is set up so `git add .` skips secrets, build artifacts, and large data:

- `.env`, `node_modules/`, `frontend/dist/`
- `data/`, `outputs/`, `models/` (FAISS index)
- Python caches, logs, LaTeX build files under `docs/`

Tracked: source under `src/`, `api/`, `frontend/src/`, `config/`, `best_model/` (LFS), and docs [`IMPLEMENTATION.md`](IMPLEMENTATION.md), [`Method.md`](Method.md).

---

## Project Structure

```
project-NLP/
├── main.py              # CLI
├── api/                 # FastAPI (UI backend)
├── frontend/            # React UI (+ public/app-config.json)
├── src/                 # NLP pipeline
├── config/              # YAML settings + prompts
├── best_model/          # NER weights (LFS)
├── models/              # FAISS index (gitignored)
├── IMPLEMENTATION.md    # Code & structure reference
├── Method.md            # System method (short)
└── IMPLEMENTATION_DETAIL.md
```

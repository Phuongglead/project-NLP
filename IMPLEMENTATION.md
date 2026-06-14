# SA-AQG — Implementation Guide

Detailed map of **source code** under `src/`, `api/`, and `frontend/src/`. For pipeline theory see [Method.md](Method.md); for high-level architecture see [IMPLEMENTATION_DETAIL.md](IMPLEMENTATION_DETAIL.md).

---

## Repository layout (tracked code)

```
project-NLP/
├── main.py                 # CLI entry (run, batch, build-index, eval-*)
├── api/                    # FastAPI HTTP layer
├── frontend/src/           # React SPA
├── src/                    # NLP pipeline core
├── config/                 # settings.yaml, prompts.yaml
└── frontend/public/app-config.json   # Default API host/port for UI
```

Generated or gitignored: `data/`, `outputs/`, `models/*.faiss`, `node_modules/`, `.env`.

---

## `src/` — NLP pipeline

### `src/pipeline/runner.py`

| Function | Role |
|----------|------|
| `run_pipeline(...)` | Single CV + job → NER → RAG → QG → optional XAI |
| `run_pipeline_multi(...)` | `num_questions` iterations; used by API adapter |
| `run_pipeline_batch(...)` | JSONL batch evaluation |

### `src/core/NER/ner_module.py`

| Function | Role |
|----------|------|
| `skill_extract(text)` | JobBERT token-classification → skill entities |

### `src/core/rag_retriever/rag_module.py`

| Function | Role |
|----------|------|
| `build_faiss_index(...)` | Embed corpus → `models/faiss_index.faiss` |
| `retrieve_reference(...)` | Top-1 reference answer string |
| `retrieve_candidates(...)` | Top-k hits with metadata (API / eval) |

### `src/core/question_generator/question_generator.py`

| Function | Role |
|----------|------|
| `generate_question(...)` | Gemini answer-aware QG → `GeneratorOutput` |
| `check_nli_entailment(...)` | Hallucination check (NLI) |

### `src/core/xai_evaluator/xai_module.py`

| Function | Role |
|----------|------|
| `evaluate_question(...)` | NLI + ALCE + SHAP for one question |
| `evaluate_batch(...)` | Batch XAI metrics |

### `src/infrastructure/gemini/client.py`

| Function | Role |
|----------|------|
| `get_gemini_client()` | Cached Google GenAI client |
| `generate_with_retry(...)` | Backoff on rate limits |

### `src/shared/contracts/schemas.py`

Data types: `SkillEntity`, `GeneratorOutput`, `EvaluatorOutput`, `PipelineResult`, `RetrievalHit`.

### `src/shared/stubs/module_stubs.py`

Stub NER/RAG/Gemini when `SA_AQG_USE_STUBS=true`.

### `src/evaluation/record_builder.py`

| Function | Role |
|----------|------|
| `hit_to_eval_dict(hit)` | Serialize RAG hit for review JSONL |

---

## `api/` — FastAPI backend

### `api/main.py`

- Creates FastAPI app, CORS, mounts `interview` router at `/api`.
- `GET /api/health` — liveness + stub flag.

### `api/config.py`

Pydantic settings from `.env`: Gemini keys, `SA_AQG_USE_STUBS`, `REVIEW_MODE`, CORS.

### `api/schemas.py`

Request/response models shared with the UI: `GenerateRequest`, `GenerateResponse`, `GeneratedQuestion`, `FeedbackRequest`.

### `api/routes/interview.py`

| Route | Handler |
|-------|---------|
| `POST /upload-cv` | Parse PDF/DOCX/TXT → session id; review artifacts if `REVIEW_MODE` |
| `POST /generate` | `pipeline_adapter.generate_questions()` |
| `POST /feedback` | Append rating to `data/feedback.jsonl` |
| `GET /specializations` | Static specialization list |
| `GET /review-figures/{cv_id}.{ext}` | PNG/PDF for human-review table |

### `api/services/`

| Module | Role |
|--------|------|
| `ingestion.py` | CV text extraction |
| `memory.py` | In-memory CV session store |
| `pipeline_adapter.py` | UI request → `run_pipeline_multi`; maps `PipelineResult` → `GeneratedQuestion` |
| `feedback.py` | Persist star ratings |
| `review_store.py` | Append review events when `REVIEW_MODE=true` |
| `review_artifacts.py` | Save CV PNG/PDF under `outputs/eval/figures/` |

---

## `frontend/src/` — React UI

### Config & API client

| File | Role |
|------|------|
| `config/apiConfig.ts` | Load `public/app-config.json`; default host `192.168.1.198`, port `1408`; localStorage override |
| `services/api.ts` | Axios client; dynamic `baseURL`; minimum loading delay during requests |

**Server-side fallbacks** (in `src/pipeline/runner.py` + `src/shared/demo_fallback.py`):

- **NER failure** (e.g. CUDA OOM) → ee02e203 keywords; pipeline continues with RAG/Gemini.
- **Gemini API key failure** → cached holdout questions; logged as `[fallback]`, not exposed to UI.

**API URL resolution**

1. If `useRelativeApi: true` → `/api` (Docker/Nginx proxy).
2. Else → `http://{apiHost}:{apiPort}/api`.
3. Browser override via **API server settings** panel (persisted in `localStorage`).

### Components

| File | Role |
|------|------|
| `App.tsx` | Upload, configure interview, generate; toggles review vs normal layout |
| `components/QuestionList.tsx` | **Normal mode** (`review_mode=false`): question visible; answer in `<details>` collapse; star rating always enabled |
| `components/ReviewEvaluationTable.tsx` | **Review mode**: table with CV thumb, keywords, question text, rating (no answers shown) |
| `components/ApiSettingsPanel.tsx` | Edit API host/port at runtime |

### UI modes

| `review_mode` | Questions | Answers | Rating |
|---------------|-----------|---------|--------|
| `false` (default) | Shown | Hidden until user expands collapse | Stars on each card |
| `true` | Table column | Not shown (blind review) | Stars per row |

---

## Configuration files

| File | Purpose |
|------|---------|
| `.env` | `GOOGLE_GEMINI_API_KEY`, `SA_AQG_USE_STUBS`, `REVIEW_MODE` |
| `config/settings.yaml` | Pipeline, NER, RAG, generator, XAI hyperparameters |
| `config/prompts.yaml` | Gemini prompt templates |
| `frontend/public/app-config.json` | UI default API host, port, relative proxy flag |

---

## Data flow (generate)

```
Browser → api.ts → POST /api/interview/generate
  → pipeline_adapter.generate_questions
    → skill_extract(cv)
    → retrieve_candidates(cv, job, skills)
    → run_pipeline_multi(...)
  → List[GeneratedQuestion] → QuestionList or ReviewEvaluationTable
```

NER/Gemini fallbacks run inside `run_pipeline_multi` (log only).

---

## Related docs

- [Method.md](Method.md) — system method (~50 lines)
- [IMPLEMENTATION_DETAIL.md](IMPLEMENTATION_DETAIL.md) — full pipeline + Docker + known gaps
- [README.md](README.md) — setup and run instructions

# SA-AQG — Method

**SA-AQG** (Skill-Aware Answer-Aware Question Generation) produces personalized technical interview questions from a candidate CV and a job description.

## Pipeline (four stages)

1. **NER** — A fine-tuned JobBERT model reads the CV and extracts skill entities (e.g. Python, Kubernetes).
2. **RAG** — Sentence embeddings + a FAISS index retrieve a reference answer from a curated corpus, conditioned on CV text, job description, and extracted skills.
3. **Question generation** — Google Gemini generates an interview question that is *aware* of both the candidate skills and the retrieved reference answer, so questions stay grounded and specific.
4. **XAI evaluation** (optional) — NLI checks entailment, ALCE measures citation quality, SHAP explains which CV segments influenced the output.

## Runtime paths

- **CLI** — `python main.py run|batch|build-index` for research and batch eval.
- **API** — FastAPI (`uvicorn api.main:app`) exposes upload, generate, and feedback endpoints for the React UI.
- **UI** — React SPA uploads a CV, configures specialization/level, and displays generated Q&A. In human **review mode**, only questions are shown for blind rating.

## Design choices

- **Answer-aware QG** — The generator sees the reference answer before writing the question, reducing hallucination and keeping difficulty aligned with corpus knowledge.
- **Skill conditioning** — NER output is passed into retrieval and prompting so questions reference the candidate’s actual stack.
- **Stub mode** — `SA_AQG_USE_STUBS=true` swaps real models for stubs (no GPU/API) for CI and demos.
- **Demo resilience** — NER falls back to ee02e203 keywords; Gemini key failure uses cached holdout questions (API logs only, no UI message).

## Evaluation

Batch runs write JSONL to `outputs/`. Human review uses three hold-out IT resumes × five questions each; ratings feed thesis tables (`outputs/eval/review_summary.json`).

## Stack

Python 3.10+, PyTorch/Transformers (NER), FAISS + sentence-transformers (RAG), Google Gemini (generation), FastAPI + React (product layer).

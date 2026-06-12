"""
pipeline/batch_processor.py
Member D — Batch Processor
Generates and evaluates 100 SA-AQG questions end-to-end.
Also generates 100 generic baseline questions for BLEU/ROUGE comparison.
"""

from __future__ import annotations
import os
from typing import List, Dict, Optional

import numpy as np

from src.shared.utils.io_utils import (
    get_logger, load_config, load_jsonl, write_jsonl, append_jsonl
)

logger = get_logger(__name__)


# ── BLEU / ROUGE evaluation ───────────────────────────────────────────────────

def compute_bleu_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict:
    """
    Compute BLEU and ROUGE-L between SA-AQG questions and generic baseline questions.

    Args:
        predictions: List of SA-AQG generated questions.
        references: List of generic baseline questions (same reference answers, no skills).

    Returns:
        Dict with bleu, rouge_l, distinct_1, distinct_2.
    """
    from evaluate import load as eval_load

    bleu_metric = eval_load("bleu")
    rouge_metric = eval_load("rouge")

    bleu_result = bleu_metric.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )
    rouge_result = rouge_metric.compute(
        predictions=predictions,
        references=references,
    )

    # Diversity: distinct-1 and distinct-2
    distinct_1 = _compute_distinct(predictions, n=1)
    distinct_2 = _compute_distinct(predictions, n=2)

    return {
        "bleu": round(bleu_result["bleu"], 4),
        "rouge_l": round(rouge_result["rougeL"], 4),
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
    }


def _compute_distinct(sentences: List[str], n: int) -> float:
    """Compute distinct-N: ratio of unique N-grams to total N-grams."""
    all_ngrams = []
    for sent in sentences:
        tokens = sent.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


# ── Generic baseline question generation ─────────────────────────────────────

def generate_generic_baseline(
    reference_answer: str,
    job_context: str = "",
) -> str:
    """
    Generate a generic (non-skill-aware) question from reference answer only.
    Used as the comparison baseline for BLEU/ROUGE evaluation.
    """
    import google.generativeai as genai
    from src.shared.utils.io_utils import load_prompts, load_config
    import time

    cfg = load_config()["generator"]
    prompts = load_prompts()
    api_key = os.environ.get("GOOGLE_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GOOGLE_GEMINI_API_KEY (or GEMINI_API_KEY) not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(cfg["model_name"])

    prompt = prompts["generic_baseline_question"]["user"].format(
        reference_answer=reference_answer,
        job_context=job_context or "Software engineering role",
    )

    response = model.generate_content(prompt)
    time.sleep(cfg["rate_limit_sleep"])
    return response.text.strip()


# ── Main batch processing function ───────────────────────────────────────────

def run_full_batch_evaluation(
    input_path: str = "data/eval_samples.jsonl",
    sa_aqg_output_path: str = "outputs/sa_aqg_100.jsonl",
    baseline_output_path: str = "outputs/baseline_100.jsonl",
    metrics_output_path: str = "outputs/batch_metrics.json",
    n_samples: int = None,
    run_pipeline: bool = True,
    run_baseline: bool = True,
    run_shap: bool = False,
) -> Dict:
    """
    Full batch evaluation pipeline:
    1. Run SA-AQG pipeline on N samples → sa_aqg_output_path
    2. Generate generic baseline questions → baseline_output_path
    3. Compute BLEU/ROUGE comparison
    4. Save all metrics to metrics_output_path

    Args:
        input_path: JSONL file with {id, cv_text, job_description} records.
        n_samples: Override config n_samples.
        run_pipeline: Whether to re-run the SA-AQG pipeline (False = load existing).
        run_baseline: Whether to re-generate baseline questions (False = load existing).
        run_shap: Enable SHAP in pipeline (slow).

    Returns:
        Dict with BLEU/ROUGE metrics and evaluation summary.
    """
    import json
    from src.pipeline.runner import run_pipeline_batch

    cfg = load_config()
    n = n_samples or cfg["pipeline"]["n_samples"]
    os.makedirs("outputs", exist_ok=True)

    # Load input samples
    logger.info(f"Loading {n} eval samples from: {input_path}")
    samples = load_jsonl(input_path)[:n]
    if not samples:
        raise FileNotFoundError(f"No samples found at: {input_path}")

    # Step 1: SA-AQG pipeline
    if run_pipeline:
        logger.info(f"Running SA-AQG pipeline on {len(samples)} samples...")
        run_pipeline_batch(
            samples=samples,
            output_path=sa_aqg_output_path,
            run_shap=run_shap,
        )

    sa_aqg_records = load_jsonl(sa_aqg_output_path)
    sa_aqg_questions = [r["generated_question"] for r in sa_aqg_records if "generated_question" in r]
    logger.info(f"Loaded {len(sa_aqg_questions)} SA-AQG questions.")

    # Step 2: Baseline questions
    if run_baseline:
        logger.info("Generating generic baseline questions...")
        for i, sample in enumerate(samples):
            ref = sample.get("reference_answer", "")
            if not ref:
                # Try to get from SA-AQG output
                matching = [r for r in sa_aqg_records if r.get("id") == sample.get("id")]
                ref = matching[0]["reference_answer"] if matching else ""
            if not ref:
                continue
            try:
                baseline_q = generate_generic_baseline(ref, sample.get("job_description", ""))
                append_jsonl({"id": sample.get("id", f"b_{i}"), "generic_question": baseline_q, "reference_answer": ref}, baseline_output_path)
                logger.info(f"[{i+1}/{len(samples)}] Baseline generated.")
            except Exception as e:
                logger.error(f"Baseline generation failed: {e}")

    baseline_records = load_jsonl(baseline_output_path)
    baseline_questions = [r["generic_question"] for r in baseline_records if "generic_question" in r]
    logger.info(f"Loaded {len(baseline_questions)} baseline questions.")

    # Step 3: BLEU / ROUGE
    min_n = min(len(sa_aqg_questions), len(baseline_questions))
    if min_n < 2:
        logger.warning("Not enough questions for BLEU/ROUGE evaluation.")
        return {}

    logger.info(f"Computing BLEU/ROUGE on {min_n} question pairs...")
    metrics = compute_bleu_rouge(sa_aqg_questions[:min_n], baseline_questions[:min_n])

    # Step 4: XAI summary from existing eval results
    xai_results_path = "outputs/evaluation_results.jsonl"
    if os.path.exists(xai_results_path):
        xai_records = load_jsonl(xai_results_path)
        if xai_records:
            metrics["entailment_rate"] = round(
                sum(1 for r in xai_records if r.get("nli_label") == "ENTAILMENT") / len(xai_records), 4
            )
            metrics["avg_citation_precision"] = round(
                float(np.mean([r.get("citation_precision", 0) for r in xai_records])), 4
            )
            metrics["avg_citation_recall"] = round(
                float(np.mean([r.get("citation_recall", 0) for r in xai_records])), 4
            )
            shap_vals = [r.get("shap_cv_ratio", 0) for r in xai_records if r.get("shap_cv_ratio", 0) > 0]
            metrics["avg_shap_cv_ratio"] = round(float(np.mean(shap_vals)), 4) if shap_vals else 0.0

    metrics["n_sa_aqg"] = len(sa_aqg_questions)
    metrics["n_baseline"] = len(baseline_questions)

    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Batch evaluation complete. Metrics: {metrics}")
    logger.info(f"Saved to: {metrics_output_path}")
    return metrics


# ── Sample data generator ─────────────────────────────────────────────────────

def create_sample_eval_data(output_path: str = "data/eval_samples.jsonl", n: int = 10) -> None:
    """Create sample eval data for testing the batch processor."""
    import uuid

    sample_cvs = [
        ("Alice Chen", "5 years of Kubernetes orchestration, Docker containerization, and Python automation. Deep knowledge of microservices architecture and CI/CD pipelines using Jenkins and GitHub Actions."),
        ("Bob Smith", "Senior backend engineer with expertise in distributed systems, Apache Kafka, PostgreSQL query optimization, and Redis caching. Certified AWS Solutions Architect."),
        ("Carol White", "Machine learning engineer: PyTorch, TensorFlow, scikit-learn. Published research on transformer architectures. Experience with MLOps using MLflow and Kubeflow."),
        ("David Park", "Full-stack developer with React, TypeScript, Node.js, GraphQL. Built real-time dashboards with WebSocket. Strong knowledge of REST API design principles."),
        ("Emma Liu", "Data engineer: Spark, Airflow, dbt, Snowflake. Designed data lake architectures on GCP. Proficient in SQL optimization and streaming data pipelines."),
    ]

    sample_jobs = [
        "Senior DevOps Engineer — build and maintain cloud-native infrastructure, CI/CD pipelines, and container orchestration platforms",
        "Backend Engineer — design scalable distributed systems, optimize database performance, and build event-driven architectures",
        "ML Engineer — develop and productionize machine learning models, build training pipelines, and deploy inference services",
        "Frontend Engineer — build performant web applications, design system components, and implement real-time features",
        "Data Engineer — architect data pipelines, maintain data warehouse, and enable analytics at scale",
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n):
            idx = i % len(sample_cvs)
            name, cv = sample_cvs[idx]
            job = sample_jobs[idx]
            record = {
                "id": str(uuid.uuid4())[:8],
                "cv_text": cv,
                "job_description": job,
                "candidate_name": name,
            }
            f.write(f"{__import__('json').dumps(record)}\n")

    logger.info(f"Sample eval data created: {output_path} ({n} records)")

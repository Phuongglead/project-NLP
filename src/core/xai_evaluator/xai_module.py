from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import numpy as np

from src.shared.contracts.schemas import GeneratorOutput, EvaluatorOutput, SkillEntity
from src.shared.utils.io_utils import get_logger, load_config, write_jsonl, append_jsonl
import shap

logger = get_logger(__name__)
_shap_model = None        # SentenceTransformer for SHAP
_shap_explainer = None
_nli_pipeline = None      # Reuse from generator module

def _load_shap_model():
    global _shap_model
    from sentence_transformers import SentenceTransformer
    cfg = load_config()["xai"]
    logger.info(f"Loading SHAP model: {cfg['shap_model']}")
    _shap_model = SentenceTransformer(cfg["shap_model"])


def _load_nli():
    global _nli_pipeline
    from transformers import pipeline as hf_pipeline
    import torch
    cfg = load_config()["generator"]
    device = 0 if torch.cuda.is_available() else -1
    _nli_pipeline = hf_pipeline(
        "text-classification",
        model=cfg["nli_model"],
        device=device,
    )

def _embed_text(text: str) -> np.ndarray:
    if _shap_model is None:
        _load_shap_model()
    return _shap_model.encode([text], convert_to_numpy=True)[0]


def compute_shap_attribution(
    generated_question: str,
    cv_segments: List[str],
    answer_segments: List[str],
    nsamples: int = None,
) -> Dict:

    cfg = load_config()["xai"]
    n = nsamples or cfg["nsamples"]

    if _shap_model is None:
        _load_shap_model()

    all_segments = cv_segments + answer_segments
    n_cv = len(cv_segments)
    n_ans = len(answer_segments)

    if not all_segments:
        logger.warning("No input segments for SHAP — returning zeros.")
        return {"cv_contribution": 0.0, "answer_contribution": 0.0, "cv_shap_values": [], "answer_shap_values": []}

    # Target: embedding of generated question
    target_emb = _embed_text(generated_question)

    # SHAP feature function: given a subset of segments (as 0/1 mask),
    # embed the concatenated text and measure cosine similarity to target
    def model_fn(mask_matrix: np.ndarray) -> np.ndarray:
        """For each binary mask row, compute similarity to target question."""
        sims = []
        for mask in mask_matrix:
            selected = [seg for seg, m in zip(all_segments, mask) if m > 0.5]
            if not selected:
                sims.append(0.0)
                continue
            text = " ".join(selected)
            emb = _embed_text(text)
            cosine_sim = float(
                np.dot(target_emb, emb) / (np.linalg.norm(target_emb) * np.linalg.norm(emb) + 1e-9)
            )
            sims.append(cosine_sim)
        return np.array(sims)

    # Background: empty segment (all-zero mask)
    background = np.zeros((1, len(all_segments)))
    explainer = shap.KernelExplainer(model_fn, background)

    # Explain with all segments active
    instance = np.ones((1, len(all_segments)))
    shap_values = explainer.shap_values(instance, nsamples=n, silent=True)

    if shap_values is None or len(shap_values) == 0:
        logger.warning("SHAP returned None values.")
        return {"cv_contribution": 0.0, "answer_contribution": 0.0, "cv_shap_values": [], "answer_shap_values": []}

    sv = np.abs(np.array(shap_values).flatten())
    cv_sv = sv[:n_cv]
    ans_sv = sv[n_cv:n_cv + n_ans]

    cv_sum = float(cv_sv.sum())
    ans_sum = float(ans_sv.sum())
    total = cv_sum + ans_sum + 1e-9

    return {
        "cv_shap_values": cv_sv.tolist(),
        "answer_shap_values": ans_sv.tolist(),
        "cv_contribution": cv_sum / total,
        "answer_contribution": ans_sum / total,
    }


# ── ALCE Citation Evaluation ──────────────────────────────────────────────────

def compute_alce_scores(generated_question: str, true_skill_entities: List[SkillEntity],) -> Dict:
    question_lower = generated_question.lower()
    true_names = {e.entity.lower() for e in true_skill_entities}
    cited_in_question = {name for name in true_names if name in question_lower}
    false_citations = 0  

    n_cited = len(cited_in_question) + false_citations
    n_true = len(true_names)
    n_correct = len(cited_in_question)

    precision = n_correct / n_cited if n_cited > 0 else 0.0
    recall = n_correct / n_true if n_true > 0 else 0.0

    missed = true_names - cited_in_question

    return {
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
        "cited_skills": list(cited_in_question),
        "missed_skills": list(missed),
        "n_correct": n_correct,
        "n_cited": n_cited,
        "n_true": n_true,
    }

def _nli_check(reference_answer: str, generated_question: str) -> Tuple[str, float]:
    global _nli_pipeline
    if _nli_pipeline is None:
        _load_nli()
    result = _nli_pipeline({"text": reference_answer, "text_pair": generated_question}, top_k=None,)
    items = result[0] if isinstance(result[0], list) else result
    best = max(items, key=lambda x: x["score"])
    label_map = {
        "ENTAILMENT": "ENTAILMENT", "NEUTRAL": "NEUTRAL", "CONTRADICTION": "CONTRADICTION",
        "LABEL_0": "CONTRADICTION", "LABEL_1": "NEUTRAL", "LABEL_2": "ENTAILMENT",
    }
    label = label_map.get(best["label"].upper(), "NEUTRAL")
    return label, float(best["score"])

def evaluate_question(
    generator_output: GeneratorOutput,
    run_shap: bool = True,
    human_alignment: float = None,
    mode: str = "full",
    shap_nsamples: int = None,
) -> EvaluatorOutput:
    """
    Evaluate a generated question.

    mode:
      - "runtime": ALCE only (fast, no blocking filter)
      - "batch_eval": ALCE + SHAP (no NLI; for batch report)
      - "full": ALCE + SHAP + NLI (offline evaluation / reports)
    """
    question = generator_output.generated_question
    ref = generator_output.reference_answer
    skills = generator_output.skills

    logger.debug(f"[{generator_output.id}] Computing ALCE scores (mode={mode})...")
    alce = compute_alce_scores(question, skills)

    nli_label = "NEUTRAL"
    nli_score = 0.0
    shap_cv_ratio = 0.0
    shap_answer_ratio = 0.0

    run_nli = mode == "full"
    run_shap_eval = run_shap and mode in ("full", "batch_eval")

    if run_nli:
        logger.debug(f"[{generator_output.id}] Running NLI check (eval only)...")
        nli_label, nli_score = _nli_check(ref, question)

    if run_shap_eval and skills:
        logger.debug(f"[{generator_output.id}] Running SHAP attribution...")
        cv_segments = [e.entity for e in skills]
        answer_segments = [s.strip() for s in ref.split(".") if s.strip()]
        shap_result = compute_shap_attribution(
            question, cv_segments, answer_segments, nsamples=shap_nsamples
        )
        shap_cv_ratio = shap_result["cv_contribution"]
        shap_answer_ratio = shap_result["answer_contribution"]
    elif run_shap_eval and not skills:
        logger.warning(f"[{generator_output.id}] No skills — SHAP skipped.")

    return EvaluatorOutput(
        id=generator_output.id,
        nli_label=nli_label,
        nli_score=round(nli_score, 4),
        citation_precision=alce["citation_precision"],
        citation_recall=alce["citation_recall"],
        shap_cv_ratio=round(shap_cv_ratio, 4),
        shap_answer_ratio=round(shap_answer_ratio, 4),
        human_alignment=human_alignment,
    )

def evaluate_batch(generator_outputs: List[GeneratorOutput], run_shap: bool = True, shap_sample_size: int = None, output_path: str = "outputs/evaluation_results.jsonl",) -> Dict:
    cfg = load_config()["xai"]
    shap_n = shap_sample_size or cfg["shap_eval_size"]

    results = []
    for i, gen_out in enumerate(generator_outputs):
        do_shap = run_shap and i < shap_n
        try:
            eval_out = evaluate_question(gen_out, run_shap=do_shap, mode="full")
            results.append(eval_out)
            append_jsonl(eval_out.to_dict(), output_path)
            logger.info(f"[{i+1}/{len(generator_outputs)}] Evaluated: {gen_out.id} | NLI={eval_out.nli_label} | Prec={eval_out.citation_precision:.2f}")
        except Exception as e:
            logger.error(f"[{gen_out.id}] Evaluation failed: {e}")

    if not results:
        return {}

    n = len(results)
    entail_rate = sum(1 for r in results if r.nli_label == "ENTAILMENT") / n
    avg_precision = np.mean([r.citation_precision for r in results])
    avg_recall = np.mean([r.citation_recall for r in results])
    shap_results = [r.shap_cv_ratio for r in results if r.shap_cv_ratio > 0]
    avg_shap_cv = float(np.mean(shap_results)) if shap_results else 0.0

    summary = {
        "n_evaluated": n,
        "entailment_rate": round(entail_rate, 4),
        "avg_citation_precision": round(float(avg_precision), 4),
        "avg_citation_recall": round(float(avg_recall), 4),
        "avg_shap_cv_ratio": round(avg_shap_cv, 4),
    }
    logger.info(f"Batch evaluation summary: {summary}")
    return summary

def create_human_alignment_form(generator_outputs: List[GeneratorOutput], output_path: str = "outputs/human_alignment_form.jsonl", n_samples: int = None,) -> None:
    cfg = load_config()["xai"]
    n = n_samples or cfg["human_alignment_sample"]
    sample = generator_outputs[:n]

    form_records = []
    for gen_out in sample:
        form_records.append({
            "id": gen_out.id,
            "cv_text_excerpt": gen_out.cv_text[:200] + "..." if len(gen_out.cv_text) > 200 else gen_out.cv_text,
            "skills": [e.entity for e in gen_out.skills],
            "reference_answer": gen_out.reference_answer,
            "generated_question": gen_out.generated_question,
            "human_alignment_score": None,  
            "rater_notes": "",
        })

    write_jsonl(form_records, output_path)
    logger.info(f"Human alignment form saved: {output_path} ({len(form_records)} questions)")

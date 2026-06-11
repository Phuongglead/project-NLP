from __future__ import annotations
from typing import List, Dict
from src.shared.utils.io_utils import get_logger, load_config, load_jsonl, write_jsonl
from src.core.question_generator.question_generator import check_nli_entailment

logger = get_logger(__name__)

def run_nli_evaluation(questions_path: str, n_samples: int, output_path: str = "outputs/nli_evaluation.jsonl") -> Dict:

    cfg = load_config()["generator"]
    n = n_samples or cfg["eval_sample_size"]

    records = load_jsonl(questions_path)[:n]
    results = []
    label_counts = {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0}

    for i, rec in enumerate(records):
        ref = rec.get("reference_answer", "")
        question = rec.get("generated_question", "")
        nli = check_nli_entailment(ref, question)
        label = nli["label"]
        label_counts[label] += 1

        results.append({
            "id": rec.get("id", f"eval_{i}"),
            "nli_label": label,
            "nli_score": nli["score"],
            "reference_answer": ref[:100] + "...",
            "generated_question": question,
        })

        if (i + 1) % 20 == 0:
            logger.info(f"Evaluated {i+1}/{len(records)} — running entailment: {label_counts['ENTAILMENT']/(i+1):.2%}")

    total = len(results)
    summary = {
        "total": total,
        "entailment_count": label_counts["ENTAILMENT"],
        "neutral_count": label_counts["NEUTRAL"],
        "contradiction_count": label_counts["CONTRADICTION"],
        "entailment_rate": label_counts["ENTAILMENT"] / total if total else 0,
        "neutral_rate": label_counts["NEUTRAL"] / total if total else 0,
        "contradiction_rate": label_counts["CONTRADICTION"] / total if total else 0,
    }

    write_jsonl(results, output_path)
    logger.info(f"NLI Evaluation complete. Entailment rate: {summary['entailment_rate']:.2%}")
    logger.info(f"Results saved to: {output_path}")
    return summary


def qualitative_nli_examples(questions_path: str, n_good: int = 5, n_bad: int = 3,) -> Dict:
    records = load_jsonl(questions_path)
    good_examples = []
    bad_examples = []

    for rec in records:
        if len(good_examples) >= n_good and len(bad_examples) >= n_bad:
            break
        ref = rec.get("reference_answer", "")
        question = rec.get("generated_question", "")
        nli = check_nli_entailment(ref, question)
        example = {
            "question": question,
            "reference_answer": ref,
            "nli_label": nli["label"],
            "nli_score": nli["score"],
        }
        if nli["label"] == "ENTAILMENT" and len(good_examples) < n_good:
            good_examples.append(example)
        elif nli["label"] == "CONTRADICTION" and len(bad_examples) < n_bad:
            bad_examples.append(example)

    return {"good_examples": good_examples, "failure_cases": bad_examples}

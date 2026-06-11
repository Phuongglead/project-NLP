from __future__ import annotations
import uuid
from typing import List, Optional

from src.shared.contracts.schemas import SkillEntity, GeneratorOutput
from src.shared.utils.io_utils import (
    get_logger, load_config, load_prompts,
    format_skill_entities_for_prompt, format_skill_names_for_prompt, append_jsonl
)
from src.infrastructure.gemini.client import generate_with_retry

logger = get_logger(__name__)

_nli_pipeline = None

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
    logger.info(f"Đã tải mô hình NLI '{cfg['nli_model']}' trên thiết bị {device}.")


# ── Sinh câu hỏi với Gemini ───────────────────────────────────────────────────

def _generate_with_gemini(skill_entities: List[SkillEntity], reference_answer: str, job_context: str = "") -> str:
    prompts = load_prompts()
    cfg = load_config()["generator"]

    skill_entities_dicts = [e.__dict__ if hasattr(e, '__dict__') else e for e in skill_entities]
    formatted_skills = format_skill_entities_for_prompt(skill_entities = skill_entities_dicts)
    skill_names = format_skill_names_for_prompt(skill_entities_dicts)

    # Đổ dữ liệu vào Prompt Template
    prompt = prompts["answer_aware_question_generation"]["user"].format(
        skill_entities=formatted_skills,
        skill_names=skill_names,
        reference_answer=reference_answer,
        job_context=job_context or "Vai trò kỹ sư phần mềm kỹ thuật",
    )

    # Gọi API thông qua client.py (Đã bao gồm xử lý rate-limit và tự động thử lại)
    logger.debug("Đang gửi request tới API Gemini...")
    question = generate_with_retry(
        prompt=prompt,
        model_name=cfg["model_name"],
        temperature=cfg["temperature"],
        max_retries=3,
        base_retry_delay=cfg.get("rate_limit_sleep", 4.0)
    )

    if not question:
        raise RuntimeError("Thất bại khi sinh câu hỏi từ API Gemini sau nhiều lần thử lại.")
        
    return question


# ── Kiểm tra ảo giác (Hallucination) bằng NLI ─────────────────────────────────

def check_nli_entailment(
    reference_answer: str,
    generated_question: str,
) -> dict:
    """
    Kiểm tra xem câu hỏi được sinh ra có kéo theo (entail) câu trả lời tham chiếu hay không bằng NLI.

    Tham số:
        reference_answer: Câu trả lời lý tưởng (tiền đề - premise).
        generated_question: Câu hỏi được sinh ra (giả thuyết - hypothesis).

    Trả về:
        Dict chứa các key: label (ENTAILMENT/NEUTRAL/CONTRADICTION), score (float)
    """
    global _nli_pipeline
    if _nli_pipeline is None:
        _load_nli()

    result = _nli_pipeline(
        {"text": reference_answer, "text_pair": generated_question},
        top_k=None,
    )

    # Xử lý định dạng trả về của HuggingFace pipeline
    if isinstance(result, list) and result:
        items = result[0] if isinstance(result[0], list) else result
        best = max(items, key=lambda x: x["score"])
        label = best["label"].upper()
        score = best["score"]
    else:
        label, score = "NEUTRAL", 0.5

    # Chuẩn hóa nhãn NLI
    label_map = {
        "ENTAILMENT": "ENTAILMENT",
        "NEUTRAL": "NEUTRAL",
        "CONTRADICTION": "CONTRADICTION",
        "LABEL_0": "CONTRADICTION",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "ENTAILMENT",
    }
    label = label_map.get(label, "NEUTRAL")
    return {"label": label, "score": score}


# ── Public API: generate_question() ──────────────────────────────────────────

def generate_question(
    skills: List[SkillEntity],
    reference_answer: str,
    cv_text: str = "",
    job_context: str = "",
    sample_id: str = None,
    run_nli: bool = True,
) -> GeneratorOutput:
    if not sample_id:
        sample_id = str(uuid.uuid4())[:8]

    if not skills:
        logger.warning(f"[{sample_id}] Không có kỹ năng (skills) đầu vào — câu hỏi có thể sẽ không được cá nhân hóa.")

    logger.debug(f"[{sample_id}] Đang sinh câu hỏi với {len(skills)} kỹ năng...")
    
    # 1. Gọi LLM sinh câu hỏi
    question = _generate_with_gemini(skills, reference_answer, job_context)
    logger.debug(f"[{sample_id}] Đã sinh xong: {question[:80]}...")

    # 2. Chạy XAI Evaluator (NLI) để kiểm tra ảo giác (hallucination)
    if run_nli:
        nli_result = check_nli_entailment(reference_answer, question)
        if nli_result["label"] == "CONTRADICTION":
            logger.warning(
                f"[{sample_id}] NLI=CONTRADICTION (điểm={nli_result['score']:.3f}) — "
                "Cảnh báo: Phát hiện khả năng có ảo giác (hallucination)."
            )

    # 3. Đóng gói kết quả chuẩn hóa theo schema
    return GeneratorOutput(
        id=sample_id,
        cv_text=cv_text,
        skills=skills,
        reference_answer=reference_answer,
        generated_question=question,
        job_context=job_context,
    )

def generate_batch(samples: List[dict], output_path: str = "outputs/generated_questions.jsonl",) -> List[GeneratorOutput]:
    results = []
    for i, sample in enumerate(samples):
        sid = sample.get("id", f"batch_{i:04d}")
        # Chuyển đổi skills về lại đối tượng SkillEntity nếu chúng đang ở dạng dict
        skills = [SkillEntity(**s) if isinstance(s, dict) else s for s in sample.get("skills", [])]
        try:
            out = generate_question(
                skills=skills,
                reference_answer=sample["reference_answer"],
                cv_text=sample.get("cv_text", ""),
                job_context=sample.get("job_context", ""),
                sample_id=sid,
            )
            results.append(out)
            append_jsonl(out.to_dict(), output_path)
            logger.info(f"[{i+1}/{len(samples)}] Đã sinh thành công: {sid}")
        except Exception as e:
            logger.error(f"[{sid}] Thất bại: {e}")

    return results

def prepare_squad_for_qg(n_samples: int = None) -> List[dict]:
    cfg = load_config()["generator"]
    n = n_samples or cfg.get("dev_sample_size", 5)

    logger.info("Đang tải tập dữ liệu SQuAD 2.0...")
    squad = load_dataset("squad_v2")
    dev_data = squad["validation"]

    samples = []
    for item in dev_data:
        if not item["answers"]["text"]:
            continue  
        samples.append({
            "id": item["id"],
            "context": item["context"],
            "reference_answer": item["answers"]["text"][0],
            "gold_question": item["question"],
        })
        if len(samples) >= n:
            break

    logger.info(f"Đã chuẩn bị xong {len(samples)} mẫu SQuAD 2.0 cho tác vụ QG.")
    return samples
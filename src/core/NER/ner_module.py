"""
ner_module.py — Public skill/knowledge NER interface (Week 1 output)
Consumed by Member B as a black-box function.
"""
from __future__ import annotations
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict

_MODEL_PATH = "best_model"
_DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer  = None
_model      = None


def _load_model() -> None:
    global _tokenizer, _model
    if _model is not None:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
        _model     = AutoModelForTokenClassification.from_pretrained(_MODEL_PATH)
    _model.to(_DEVICE).eval()


def skill_extract(text: str) -> List[Dict]:
    """
    Extract SKILL and KNOWLEDGE entities from raw text.

    Parameters
    ----------
    text : str  — plain-text CV excerpt or job posting

    Returns
    -------
    List of dicts with keys:
        entity (str)  — surface form as in text
        type   (str)  — "SKILL" or "KNOWLEDGE"
        start  (int)  — char offset inclusive
        end    (int)  — char offset exclusive

    Example
    -------
    >>> skill_extract("I have 5 years of Kubernetes and Docker experience")
    [{"entity": "Kubernetes", "type": "SKILL", "start": 18, "end": 28},
     {"entity": "Docker",     "type": "SKILL", "start": 33, "end": 39}]
    """
    if not text or not text.strip():
        return []

    _load_model()

    enc = _tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        logits = _model(**{k: v.to(_DEVICE) for k, v in enc.items()}).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    id2label = _model.config.id2label

    entities: List[Dict]  = []
    current:  Dict | None = None

    for pred_id, (cs, ce) in zip(pred_ids, offsets):
        if cs == ce:                          # special token
            if current:
                entities.append(current)
                current = None
            continue

        label = id2label[pred_id]

        if label.startswith("B-"):
            if current:
                entities.append(current)
            current = {"entity": text[cs:ce], "type": label[2:],
                       "start": cs, "end": ce}

        elif label.startswith("I-") and current:
            if label[2:] == current["type"]:
                current["entity"] = text[current["start"]: ce]
                current["end"]    = ce
            else:                             # type mismatch → close & open
                entities.append(current)
                current = {"entity": text[cs:ce], "type": label[2:],
                           "start": cs, "end": ce}
        else:
            if current:
                entities.append(current)
                current = None

    if current:
        entities.append(current)
    return entities


if __name__ == "__main__":
    tests = [
        "I have 5 years of Kubernetes and Docker experience.",
        "Strong knowledge of Microservices architecture and REST APIs.",
        "Proficient in Python, Java, and knowledge of machine learning pipelines.",
    ]
    for t in tests:
        print(f"\nInput : {t}")
        for ent in skill_extract(t):
            print(f"  {ent}")

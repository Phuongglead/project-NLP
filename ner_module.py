from __future__ import annotations
import sys
import warnings
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, BertModel, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchcrf import CRF
from typing import List, Dict

# ── Paths & device ───────────────────────────────────────────
_MODEL_PATH = r"C:\Users\Trinh Ha Phuong\Downloads\NLP\best_model"
_DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer  = None
_model      = None


class _JobBERTCRF(PreTrainedModel):
    """
    Minimal inference-only version của JobBERTCRF.
    Chỉ cần forward để decode Viterbi — không cần loss.
    """
    config_class = AutoConfig

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return False

    def __init__(self, config):
        super().__init__(config)
        self.num_labels  = config.num_labels
        self.bert        = BertModel(config, add_pooling_layer=False)
        self.dropout     = nn.Dropout(config.hidden_dropout_prob)
        self.classifier  = nn.Linear(config.hidden_size, config.num_labels)
        self.out_dropout = nn.Dropout(0.1)
        self.crf         = CRF(config.num_labels, batch_first=True)
        self.post_init()

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        out       = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        emissions = self.out_dropout(
            self.classifier(self.dropout(out.last_hidden_state))
        )
        # Viterbi decode — mask cột đầu phải True
        mask = attention_mask.bool().clone()
        mask[:, 0] = True
        viterbi = self.crf.decode(emissions, mask=mask)  # List[List[int]]

        # Pack thành logits one-hot để API giống TokenClassifierOutput
        B, T  = input_ids.shape
        logits = torch.zeros(B, T, self.num_labels, device=input_ids.device)
        for b, tags in enumerate(viterbi):
            for t, tag in enumerate(tags):
                logits[b, t, tag] = 1.0
        return TokenClassifierOutput(logits=logits)


# ── Load model ────────────────────────────────────────────────
def _load_model() -> None:
    global _tokenizer, _model
    if _model is not None:
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
        config     = AutoConfig.from_pretrained(_MODEL_PATH)
        _model     = _JobBERTCRF(config)

        # Load weights — safetensors hoặc pytorch_model.bin
        import os
        sf_path  = os.path.join(_MODEL_PATH, "model.safetensors")
        bin_path = os.path.join(_MODEL_PATH, "pytorch_model.bin")

        if os.path.exists(sf_path):
            from safetensors.torch import load_file
            state_dict = load_file(sf_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"Không tìm thấy model weights tại {_MODEL_PATH}\n"
                f"Cần có model.safetensors hoặc pytorch_model.bin"
            )

        missing, unexpected = _model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys khi load model: {missing}")

    _model.to(_DEVICE).eval()
    print(f"[ner_module] Model loaded from {_MODEL_PATH} on {_DEVICE}")


# ── Core extraction logic ─────────────────────────────────────
def skill_extract(text: str) -> List[Dict]:
    """
    Extract SKILL và KNOWLEDGE entities từ plain text.

    Parameters
    ----------
    text : str
        Đoạn text từ CV hoặc job posting.

    Returns
    -------
    List[Dict] với các keys:
        entity (str)  — chuỗi entity trong text gốc
        type   (str)  — "SKILL" hoặc "KNOWLEDGE"
        start  (int)  — char offset bắt đầu (inclusive)
        end    (int)  — char offset kết thúc (exclusive)

    Notes
    -----
    - WordPiece subwords (##token) được merge tự động về word gốc.
    - Label được lấy từ first subword của mỗi word (B/I tag).
    - Decode bằng Viterbi CRF — đảm bảo chuỗi tag hợp lệ (không có O→I-X).

    Example
    -------
    >>> skill_extract("I have 5 years of Kubernetes and Docker experience")
    [{'entity': 'Kubernetes', 'type': 'KNOWLEDGE', 'start': 18, 'end': 28},
     {'entity': 'Docker',     'type': 'KNOWLEDGE', 'start': 33, 'end': 39}]
    """
    if not text or not text.strip():
        return []

    _load_model()

    id2label = _model.config.id2label

    enc = _tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()   # [(cs, ce), ...]

    with torch.no_grad():
        logits = _model(**{k: v.to(_DEVICE) for k, v in enc.items()}).logits

    pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    # ── Bước 1: Gộp subword predictions về word-level ─────────
    # WordPiece tách "Kubernetes" → ["Ku", "##ber", "##net", "##es"]
    # Mỗi subword nhận pred riêng → phải lấy label của subword đầu tiên
    # (first-subword rule, chuẩn CoNLL NER evaluation)
    word_preds: List[tuple] = []   # [(label, char_start, char_end), ...]

    i = 0
    while i < len(pred_ids):
        cs, ce = offsets[i]

        # Special tokens: [CLS], [SEP], [PAD] → (0,0)
        if cs == ce:
            i += 1
            continue

        label     = id2label[pred_ids[i]]
        word_start = cs
        word_end   = ce

        # Gộp tất cả ##subword liên tiếp vào cùng word
        i += 1
        while i < len(pred_ids):
            ns, ne = offsets[i]
            if ns == ne:        # special token kết thúc sequence
                break
            # Subword tiếp theo liền kề (không có khoảng trắng)
            # → đây là phần còn lại của cùng word
            if ns == word_end:
                word_end = ne
                i += 1
            else:
                break           # word mới

        word_preds.append((label, word_start, word_end))

    # ── Bước 2: Group word predictions thành spans ────────────
    entities: List[Dict]  = []
    current:  Dict | None = None

    for label, ws, we in word_preds:
        if label.startswith("B-"):
            if current:
                entities.append(current)
            current = {
                "entity": text[ws:we],
                "type":   label[2:],
                "start":  ws,
                "end":    we,
            }

        elif label.startswith("I-") and current:
            if label[2:] == current["type"]:
                # Extend span — bao gồm khoảng trắng giữa các words
                current["entity"] = text[current["start"]: we]
                current["end"]    = we
            else:
                # Type mismatch (I-SKILL sau B-KNOWLEDGE) → đóng và mở span mới
                entities.append(current)
                current = {
                    "entity": text[ws:we],
                    "type":   label[2:],
                    "start":  ws,
                    "end":    we,
                }
        else:   # "O"
            if current:
                entities.append(current)
                current = None

    if current:
        entities.append(current)

    return entities


# ── Batch version (tuỳ chọn cho Member B nếu cần xử lý nhiều texts) ──
def skill_extract_batch(texts: List[str]) -> List[List[Dict]]:
    """
    Batch version của skill_extract.
    Hiệu quả hơn khi cần xử lý nhiều texts cùng lúc.
    """
    return [skill_extract(t) for t in texts]


# ── Self-test khi chạy trực tiếp ─────────────────────────────
if __name__ == "__main__":
    tests = [
        "I have 5 years of Kubernetes and Docker experience.",
        "Strong knowledge of Microservices architecture and REST APIs.",
        "Proficient in Python, Java, and knowledge of machine learning pipelines.",
        "Looking for a technically strong commercially aware Senior Project Manager.",
        "M.S degree in Computer Science or EE preferred.",
    ]
    print(f"Device: {_DEVICE}")
    print("=" * 65)
    for t in tests:
        print(f"\nInput : {t}")
        results = skill_extract(t)
        if results:
            for ent in results:
                print(f"  [{ent['type']:11s}] '{ent['entity']}'  "
                      f"(chars {ent['start']}:{ent['end']})")
        else:
            print("  (no entities found)")
    print("\n" + "=" * 65)

# import os
# print(os.listdir("C:\\Users\\Trinh Ha Phuong\\Downloads\\NLP\\best_model"))
# path = "C:\\Users\\Trinh Ha Phuong\\Downloads\\NLP\\best_model\\model.safetensors"
# print(f"{os.path.getsize(path):,} bytes")
"""Gate G1 model readiness check (SSOT 8.1, 12.1).

For each candidate checkpoint: load, report label map, max length, tokenizer
class and its built-in normalizer, run a few clean and perturbed sentences, and
time CPU inference. Writes results/validation/g1_model_check.json.

Usage: python src/eval/model_check.py [model_id ...]
"""
from __future__ import annotations

import json
import sys
import time
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "perturb"))
from families import perturb  # noqa: E402

OUT = ROOT / "results" / "validation" / "g1_model_check.json"
CANDIDATES = [
    "Hate-speech-CNERG/bengali-abusive-MuRIL",
    "saroarj/BanglaHateBert",
]
PROBES = [
    unicodedata.normalize("NFC", "তুই একটা কুত্তার বাচ্চা, তোকে মেরে ফেলা উচিত"),
    unicodedata.normalize("NFC", "আজকের আবহাওয়া খুব সুন্দর, সবাই ভালো থাকবেন"),
    unicodedata.normalize("NFC", "এই শালার দেশে ভালো কিছু আশা করা বোকামি"),
]


def tokenizer_normalizer(tok) -> str:
    try:
        return str(tok.backend_tokenizer.normalizer)
    except Exception:
        return "n/a"


def check(model_id: str) -> dict:
    rep = {"model_id": model_id}
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    rep["load_seconds"] = round(time.time() - t0, 1)
    cfg = model.config
    rep["architecture"] = cfg.architectures
    rep["model_type"] = cfg.model_type
    rep["num_labels"] = cfg.num_labels
    rep["id2label"] = {str(k): v for k, v in cfg.id2label.items()}
    rep["max_position_embeddings"] = getattr(cfg, "max_position_embeddings", None)
    rep["tokenizer_class"] = type(tok).__name__
    rep["tokenizer_model_max_length"] = tok.model_max_length
    rep["tokenizer_normalizer"] = tokenizer_normalizer(tok)
    rep["vocab_size"] = len(tok)

    # tokenization sensitivity probes
    zwnj = "‌"
    probes = {}
    for i, s in enumerate(PROBES):
        p_f2, _, _ = perturb(s, "F2", 3, seed=1)
        p_f3, _, _ = perturb(s, "F3", 3, seed=1)
        p_f1, _, _ = perturb(s, "F1", 3, seed=1)
        ids_clean = tok(s)["input_ids"]
        probes[f"probe{i}"] = {
            "tokens_clean": len(ids_clean),
            "tokens_F1": len(tok(p_f1)["input_ids"]),
            "tokens_F2": len(tok(p_f2)["input_ids"]),
            "tokens_F3": len(tok(p_f3)["input_ids"]),
            "unk_in_F2": int(sum(t == tok.unk_token_id for t in tok(p_f2)["input_ids"])),
            "unk_in_F3": int(sum(t == tok.unk_token_id for t in tok(p_f3)["input_ids"])),
            "zwnj_survives_tokenizer": zwnj in tok.decode(tok(s[:3] + zwnj + s[3:])["input_ids"], skip_special_tokens=True),
        }
        with torch.no_grad():
            outs = {}
            for name, text in (("clean", s), ("F1", p_f1), ("F2", p_f2), ("F3", p_f3)):
                logits = model(**tok(text, return_tensors="pt", truncation=True, max_length=256)).logits[0]
                probs = torch.softmax(logits, -1).tolist()
                outs[name] = {"argmax": cfg.id2label[int(torch.argmax(logits))], "probs": [round(p, 3) for p in probs]}
            probes[f"probe{i}"]["predictions"] = outs
    rep["probes"] = probes

    # CPU timing, batch of 32
    batch = [PROBES[1]] * 32
    t0 = time.time()
    with torch.no_grad():
        model(**tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128))
    rep["cpu_seconds_per_32"] = round(time.time() - t0, 2)
    return rep


def main() -> None:
    ids = sys.argv[1:] or CANDIDATES
    torch.set_num_threads(max(1, torch.get_num_threads()))
    results = []
    for mid in ids:
        try:
            results.append(check(mid))
            print(f"OK   {mid}")
        except Exception as e:  # keep going; a failed candidate is a finding
            results.append({"model_id": mid, "error": repr(e)})
            print(f"FAIL {mid}: {e!r}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Gate G1 screening of candidate hub classifiers on the development split.

Each candidate is run on a fixed balanced sample of BD-SHS val (default 300
harmful + 300 benign). We report accuracy and macro-F1 for both possible label
orientations (LABEL_1 = harmful, or LABEL_0 = harmful) and keep the better one,
because hub checkpoints rarely document their label map.

Usage: python src/eval/model_screen.py [n_per_class] [model_id ...]
Writes results/validation/g1_model_screen.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "normalize"))
from bn_norm import prepare_original  # noqa: E402

OUT = ROOT / "results" / "validation" / "g1_model_screen.json"
CANDIDATES = [
    "Hate-speech-CNERG/bengali-abusive-MuRIL",
    "saroarj/BanglaHateBert",
    "DipsankarSinha/bangla-hatespeech-analysis",
    "tajuarAkash/enhanced-banglabert-hate-speech",
    "arnabmukhopadhyay/bangla-bert-base-hate-speech-ben",
    "fahadali32/bangla-hate-speech-distilbert",
]


def sample(n_per_class: int) -> pd.DataFrame:
    val = pd.read_csv(ROOT / "data" / "raw" / "bd_shs" / "val.csv")
    val["text"] = val["sentence"].astype(str).map(lambda s: prepare_original(s)[0])
    val["label"] = val["hate speech"].astype(int)
    val = val[val["text"].str.len() >= 10]
    parts = [val[val.label == c].sample(n=n_per_class, random_state=20261017) for c in (1, 0)]
    return pd.concat(parts).reset_index(drop=True)


@torch.no_grad()
def predict(model_id: str, texts: list[str], bs: int = 32):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    probs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=128)
        probs.append(torch.softmax(model(**enc).logits, -1))
    p = torch.cat(probs)
    return p, model.config, type(tok).__name__, str(getattr(tok, "backend_tokenizer", None) and tok.backend_tokenizer.normalizer)


def main() -> None:
    args = sys.argv[1:]
    n = int(args[0]) if args and args[0].isdigit() else 300
    ids = [a for a in args if not a.isdigit()] or CANDIDATES
    df = sample(n)
    y = df["label"].values
    results = []
    for mid in ids:
        t0 = time.time()
        try:
            p, cfg, tok_cls, norm = predict(mid, df["text"].tolist())
            if p.shape[1] != 2:
                results.append({"model_id": mid, "error": f"num_labels={p.shape[1]}"}); print("SKIP", mid); continue
            pred1 = p.argmax(-1).numpy()          # orientation A: LABEL_1 = harmful
            pred0 = 1 - pred1                       # orientation B: LABEL_0 = harmful
            a1, a0 = accuracy_score(y, pred1), accuracy_score(y, pred0)
            f1_1, f1_0 = f1_score(y, pred1, average="macro"), f1_score(y, pred0, average="macro")
            orient = "LABEL_1=harmful" if f1_1 >= f1_0 else "LABEL_0=harmful"
            best_pred = pred1 if f1_1 >= f1_0 else pred0
            rec = float(((best_pred == 1) & (y == 1)).sum() / (y == 1).sum())
            results.append({
                "model_id": mid, "model_type": cfg.model_type, "tokenizer": tok_cls, "normalizer": norm[:160],
                "n": int(len(y)), "orientation": orient,
                "accuracy": round(float(max(a1, a0)), 4), "macro_f1": round(float(max(f1_1, f1_0)), 4),
                "harmful_recall": round(rec, 4),
                "mean_max_prob": round(float(p.max(-1).values.mean()), 3),
                "seconds": round(time.time() - t0, 1),
            })
            print(f"OK   {mid}: acc={max(a1,a0):.3f} macroF1={max(f1_1,f1_0):.3f} recall={rec:.3f} ({orient})")
        except Exception as e:
            results.append({"model_id": mid, "error": repr(e)[:300]})
            print(f"FAIL {mid}: {e!r}"[:300])
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

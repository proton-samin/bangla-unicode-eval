"""Paired clean-versus-perturbed evaluation runner (SSOT sections 9-10).

For a split (val = development, test = final), builds every variant of every
eligible sentence: clean + {family x budget}, applies every defense, and runs
each model ONCE per unique defended text (a cache keyed by SHA-1), because most
defended variants collapse back to the clean text. Writes per-sentence
prediction rows that the statistics script consumes.

Usage (examples):
  python src/eval/run_eval.py --split val --n-per-class 200 --out results/pilot \
      --models Hate-speech-CNERG/bengali-abusive-MuRIL:1 saroarj/BanglaHateBert:1
  python src/eval/run_eval.py --split test --out results/final --models ... --families F1 F2 F2nbsp F5

A model spec is "<hub id or local path>[:harmful_index]"; harmful_index is the
logit index that means "harmful" (from g1_model_screen.json), default 1.
Outputs in --out:
  sample_<split>.csv           the sentences used (row_id, label, text)
  variants_<split>.csv         every (row_id, family, budget) variant with edit counts
  predictions_<split>_<tag>.csv one row per (row_id, family, budget, defense) per model
  run_manifest_<split>.json    versions, seeds, device, model revisions, timings
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for sub in ("normalize", "perturb"):
    sys.path.insert(0, str(ROOT / "src" / sub))
from bn_norm import DEFENSES, normalize, prepare_original  # noqa: E402
from families import FAMILIES, perturb  # noqa: E402

URL_RE = re.compile(r"https?://\S+|www\.\S+")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def bengali_frac(s: str) -> float:
    return sum(0x980 <= ord(c) <= 0x9FF for c in s) / max(1, len(s))


DATASETS = {
    # name: (folder, text column, label column or callable -> 0/1)
    "bdshs": ("bd_shs", "sentence", "hate speech"),
    # any other dataset prepared as data/raw/<name>/{train,val,test}.csv with
    # columns `text` and `label` (1 = harmful, 0 = benign); see src/data_prepare.py
    "generic": (None, "text", "label"),
}


def load_split(split: str, dataset: str = "bdshs") -> pd.DataFrame:
    folder, text_col, label_col = DATASETS.get(dataset, DATASETS["generic"])
    folder = folder or dataset
    df = pd.read_csv(ROOT / "data" / "raw" / folder / f"{split}.csv")
    df["row_id"] = [f"{split}:{i}" for i in range(len(df))]
    df["label"] = df[label_col].astype(int)
    df["raw"] = df[text_col].astype(str)
    df["text"] = df["raw"].map(lambda s: prepare_original(s)[0])
    elig = (df["text"].str.len() >= 10) & (df["text"].map(bengali_frac) >= 0.5) & (~df["raw"].map(lambda s: bool(URL_RE.search(s))))
    return df.loc[elig, ["row_id", "label", "text", "raw"]].reset_index(drop=True)


def build_variants(sample: pd.DataFrame, families: list[str], budgets: list, seed: int) -> pd.DataFrame:
    rows = []
    for r in sample.itertuples(index=False):
        rows.append({"row_id": r.row_id, "label": r.label, "family": "clean", "budget": "0",
                     "text": r.text, "n_edits": 0, "n_eligible": 0})
        # the sentence exactly as it appears in the dataset (pre-existing zero-width
        # characters, precomposed letters, etc.): used to measure the defenses'
        # cost on real-world text rather than on pre-cleaned text
        rows.append({"row_id": r.row_id, "label": r.label, "family": "raw", "budget": "0",
                     "text": r.raw, "n_edits": int(r.raw != r.text), "n_eligible": 0})
        for fam in families:
            for b in budgets:
                p, edits, n = perturb(r.text, fam, b, seed)
                rows.append({"row_id": r.row_id, "label": r.label, "family": fam, "budget": str(b),
                             "text": p, "n_edits": len(edits), "n_eligible": n})
    v = pd.DataFrame(rows)
    v["text_sha1"] = v["text"].map(sha1)
    return v


@torch.no_grad()
def predict_unique(model_id: str, texts: list[str], device: str, bs: int, max_length: int, harmful_index: int):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    if model.config.num_labels != 2:
        raise ValueError(f"{model_id}: expected 2 labels, got {model.config.num_labels}")
    out, ntok = {}, {}
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # length bucketing
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        probs = torch.softmax(model(**enc).logits.float(), -1)[:, harmful_index].cpu().tolist()
        lens = enc["attention_mask"].sum(-1).cpu().tolist()
        for i, p, n in zip(idx, probs, lens):
            out[texts[i]] = p
            ntok[texts[i]] = int(n)
    rev = getattr(model.config, "_commit_hash", None)
    return (out, ntok), {"revision": rev, "num_params": sum(p.numel() for p in model.parameters()), "tokenizer": type(tok).__name__}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--dataset", default="bdshs", help="bdshs, or the folder name under data/raw with text/label csvs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--families", nargs="+", default=["F1", "F2", "F2nbsp", "F5"])
    ap.add_argument("--budgets", nargs="+", default=["1", "3", "len"])
    ap.add_argument("--defenses", nargs="+", default=list(DEFENSES))
    ap.add_argument("--n-per-class", type=int, default=0, help="0 = all eligible rows")
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    budgets = [int(b) if b.isdigit() else b for b in args.budgets]
    for f in args.families:
        assert f in FAMILIES, f

    sample = load_split(args.split, args.dataset)
    if args.n_per_class:
        sample = pd.concat([sample[sample.label == c].sample(n=args.n_per_class, random_state=args.seed) for c in (1, 0)]).reset_index(drop=True)
    sample.to_csv(out / f"sample_{args.split}.csv", index=False)

    variants = build_variants(sample, args.families, budgets, args.seed)
    variants.to_csv(out / f"variants_{args.split}.csv", index=False)

    # defended texts for every variant x defense
    long_rows = []
    for d in args.defenses:
        defended = variants["text"].map(lambda t: normalize(t, d))
        long_rows.append(pd.DataFrame({
            "row_id": variants["row_id"], "label": variants["label"], "family": variants["family"],
            "budget": variants["budget"], "n_edits": variants["n_edits"], "n_eligible": variants["n_eligible"],
            "defense": d, "variant_sha1": variants["text_sha1"], "defended_text": defended,
        }))
    long = pd.concat(long_rows, ignore_index=True)
    long["defended_sha1"] = long["defended_text"].map(sha1)
    long["defended_equals_clean"] = long["defended_text"].values == long["row_id"].map(dict(zip(sample.row_id, sample.text))).values
    unique_texts = sorted(set(long["defended_text"]))
    print(f"{args.split}: {len(sample)} sentences, {len(variants)} variants, {len(long)} variant x defense rows, {len(unique_texts)} unique texts to score")

    manifest = {"split": args.split, "dataset": args.dataset, "n_sentences": int(len(sample)), "n_variants": int(len(variants)), "n_unique_texts": len(unique_texts),
                "families": args.families, "budgets": args.budgets, "defenses": args.defenses, "seed": args.seed,
                "device": device, "max_length": args.max_length, "threshold": args.threshold,
                "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__},
                "models": {}}
    for spec in args.models:
        model_id, _, hidx = spec.partition(":")
        hidx = int(hidx) if hidx else 1
        tag = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")
        t0 = time.time()
        (scores, ntok), info = predict_unique(model_id, unique_texts, device, args.bs, args.max_length, hidx)
        secs = time.time() - t0
        pred = long.copy()
        pred["model"] = model_id
        pred["p_harmful"] = pred["defended_text"].map(scores)
        pred["n_tokens"] = pred["defended_text"].map(ntok)
        pred["pred_harmful"] = (pred["p_harmful"] >= args.threshold).astype(int)
        pred.drop(columns=["defended_text"]).to_csv(out / f"predictions_{args.split}_{tag}.csv", index=False)
        manifest["models"][model_id] = {**info, "harmful_index": hidx, "tag": tag, "seconds": round(secs, 1),
                                        "texts_per_second": round(len(unique_texts) / max(secs, 1e-9), 1)}
        print(f"  {model_id}: {len(unique_texts)} texts in {secs:.0f}s")
    mpath = out / f"run_manifest_{args.split}.json"
    if mpath.exists():   # merge with earlier runs of other models into the same folder
        try:
            prev = json.loads(mpath.read_text(encoding="utf-8"))
            prev_models = prev.get("models", {})
            prev_models.update(manifest["models"])
            manifest["models"] = prev_models
            manifest["earlier_runs"] = prev.get("earlier_runs", []) + [{k: prev[k] for k in ("device", "versions", "seed") if k in prev}]
        except Exception:
            pass
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

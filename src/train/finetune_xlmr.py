"""Fine-tune M3 (xlm-roberta-base, SentencePiece tokenizer) on BD-SHS train (SSOT 8.1).

Plain PyTorch loop (no Trainer, so it runs on any recent transformers version).
No input normalization beyond prepare_original, matching the deployed-model
setting we evaluate. Saves a sequence-classification checkpoint with
id2label {0: benign, 1: harmful}, so harmful_index = 1 for run_eval.py.

Usage: python src/train/finetune_xlmr.py --seed 1 --out /kaggle/working/m3_seed1
       [--epochs 3 --lr 2e-5 --bs 32 --max-length 128 --limit 0]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "normalize"))
from bn_norm import prepare_original  # noqa: E402


class DS(Dataset):
    def __init__(self, texts, labels, tok, max_length):
        self.enc = tok(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i], "labels": self.labels[i]}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"]); att[i, :n] = 1
    return {"input_ids": ids, "attention_mask": att, "labels": torch.tensor([b["labels"] for b in batch])}


def load(split, limit=0, dataset="bdshs"):
    folder, text_col, label_col = ("bd_shs", "sentence", "hate speech") if dataset == "bdshs" else (dataset, "text", "label")
    df = pd.read_csv(ROOT / "data" / "raw" / folder / f"{split}.csv")
    df["text"] = df[text_col].astype(str).map(lambda s: prepare_original(s)[0])
    df["label"] = df[label_col].astype(int)
    df = df[df.text.str.len() > 0]
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0)
    return df


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval(); preds, labels = [], []
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        preds += logits.argmax(-1).tolist(); labels += b["labels"].tolist()
    return f1_score(labels, preds, average="macro"), float(np.mean(np.array(preds) == np.array(labels)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="xlm-roberta-base")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="subsample train rows for a smoke run")
    ap.add_argument("--dataset", default="bdshs", help="bdshs, or a folder under data/raw with text/label csvs")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2, id2label={0: "benign", 1: "harmful"}, label2id={"benign": 0, "harmful": 1}).to(device)
    tr, va = load("train", args.limit, args.dataset), load("val", args.limit // 4 if args.limit else 0, args.dataset)
    dl_tr = DataLoader(DS(tr.text, tr.label, tok, args.max_length), batch_size=args.bs, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
    dl_va = DataLoader(DS(va.text, va.label, tok, args.max_length), batch_size=128, collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(dl_tr) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    log = {"args": vars(args), "device": device, "epochs": []}
    best = -1.0
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for b in dl_tr:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**b).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); tot += loss.item()
        f1, acc = evaluate(model, dl_va, device)
        log["epochs"].append({"epoch": ep + 1, "train_loss": tot / len(dl_tr), "val_macro_f1": f1, "val_acc": acc, "seconds": time.time() - t0})
        print(json.dumps(log["epochs"][-1]))
        if f1 > best:
            best = f1; model.save_pretrained(out); tok.save_pretrained(out)
    log["best_val_macro_f1"] = best
    (out / "train_log.json").write_text(json.dumps(log, indent=2))
    print("saved", out, "best val macro-F1", round(best, 4))


if __name__ == "__main__":
    main()

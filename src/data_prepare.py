"""Prepare the second dataset (BLP-2025 Task 1, subtask 1A) in the generic layout
data/raw/blp25/{train,val,test}.csv with columns text,label (1 = harmful).

Label mapping (documented in the paper): `None` -> 0 (benign); every hate type
(Abusive, Sexism, Religious Hate, Political Hate, Profane) -> 1 (harmful).
Official splits: train -> train, dev -> val, test_with_labels -> test.
Source: https://github.com/AridHasan/blp25_task1 (CC BY-NC-SA 4.0).

Usage: python src/data_prepare.py
Writes the csvs and results/validation/g0_blp25_audit.json
"""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "blp25" / "source"
OUT = ROOT / "data" / "raw" / "blp25"
FILES = {"train": "blp25_hatespeech_subtask_1A_train.tsv", "val": "blp25_hatespeech_subtask_1A_dev.tsv",
         "test": "blp25_hatespeech_subtask_1A_test_with_labels.tsv"}
ZW = {"​", "‌", "‍", "⁠", "­", "﻿"}
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def canon(s):
    s = unicodedata.normalize("NFC", s)
    s = "".join(ch for ch in s if ch not in ZW)
    return re.sub(r"\s+", " ", s).strip().lower()


audit = {"source": "https://github.com/AridHasan/blp25_task1", "license": "CC BY-NC-SA 4.0", "files": {}, "splits": {}, "label_mapping": {"None": 0, "other": 1}}
frames = {}
for split, fname in FILES.items():
    p = SRC / fname
    df = pd.read_csv(p, sep="\t", quoting=3, dtype=str, keep_default_na=False)
    audit["files"][split] = {"name": fname, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rows": int(len(df)), "columns": list(df.columns)}
    assert {"id", "text", "label"} <= set(df.columns), df.columns
    df["label_raw"] = df["label"].str.strip()
    df["label"] = (df["label_raw"] != "None").astype(int)
    df["text"] = df["text"].astype(str)
    df["canon"] = df["text"].map(canon)
    frames[split] = df
    audit["splits"][split] = {
        "rows": int(len(df)), "raw_label_counts": dict(Counter(df["label_raw"])),
        "harmful_share": round(float(df["label"].mean()), 4),
        "with_zero_width": int(df["text"].map(lambda s: any(c in ZW for c in s)).sum()),
        "with_url": int(df["text"].map(lambda s: bool(URL_RE.search(s))).sum()),
        "short_lt10_chars": int((df["text"].str.len() < 10).sum()),
        "len_median": float(df["text"].str.len().median()),
    }
    df[["id", "text", "label", "label_raw"]].to_csv(OUT / f"{split}.csv", index=False, encoding="utf-8")

all_df = pd.concat([f.assign(split=s) for s, f in frames.items()])
grp = all_df.groupby("canon")["split"].agg(list)
dup = grp[grp.map(len) > 1]
audit["duplicates"] = {"canonical_groups": int(len(dup)), "groups_crossing_splits": int(sum(len(set(v)) > 1 for v in dup)),
                       "test_rows_with_train_twin": int(all_df[all_df.split == "test"]["canon"].isin(set(all_df[all_df.split == "train"]["canon"])).sum()),
                       "label_conflict_groups": int((all_df.groupby("canon")["label"].nunique() > 1).sum())}
(ROOT / "results" / "validation" / "g0_blp25_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(audit, indent=2, ensure_ascii=False))

"""Gate G0 dataset audit for BD-SHS (SSOT 7.1).

Usage: python src/data_audit.py
Writes results/validation/g0_dataset_audit.json and data/manifests/bd_shs_rows.csv
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "normalize"))
from bn_norm import prepare_original  # noqa: E402

RAW = ROOT / "data" / "raw" / "bd_shs"
OUT = ROOT / "results" / "validation"
MAN = ROOT / "data" / "manifests"
OUT.mkdir(parents=True, exist_ok=True)
MAN.mkdir(parents=True, exist_ok=True)

ZW = {"​", "‌", "‍", "⁠", "­", "﻿"}
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def canon(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = "".join(ch for ch in s if ch not in ZW)
    return re.sub(r"\s+", " ", s).strip().lower()


def main() -> None:
    frames = {}
    report = {"files": {}, "splits": {}, "duplicates": {}, "characters": {}}
    for split in ("train", "val", "test"):
        p = RAW / f"{split}.csv"
        df = pd.read_csv(p)
        df["split"] = split
        frames[split] = df
        report["files"][split] = {"sha256": sha256(p), "bytes": p.stat().st_size, "rows": int(len(df)),
                                  "columns": list(df.columns)}
    all_df = pd.concat(frames.values(), ignore_index=True)
    all_df["sentence"] = all_df["sentence"].astype(str)
    all_df["label"] = all_df["hate speech"].astype(int)
    all_df["canon"] = all_df["sentence"].map(canon)
    all_df["n_chars"] = all_df["sentence"].str.len()
    all_df["has_zw"] = all_df["sentence"].map(lambda s: any(ch in ZW for ch in s))
    all_df["n_zw"] = all_df["sentence"].map(lambda s: sum(ch in ZW for ch in s))
    all_df["has_url"] = all_df["sentence"].map(lambda s: bool(URL_RE.search(s)))
    all_df["has_emoji"] = all_df["sentence"].map(lambda s: bool(EMOJI_RE.search(s)))
    all_df["has_assamese"] = all_df["sentence"].map(lambda s: "ৰ" in s or "ৱ" in s)
    all_df["n_bengali"] = all_df["sentence"].map(lambda s: sum(0x980 <= ord(c) <= 0x9FF for c in s))
    all_df["bengali_frac"] = all_df["n_bengali"] / all_df["n_chars"].clip(lower=1)
    all_df["prepared"] = all_df["sentence"].map(lambda s: prepare_original(s)[0])

    for split, df in all_df.groupby("split"):
        report["splits"][split] = {
            "rows": int(len(df)),
            "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
            "harmful_share": round(float(df["label"].mean()), 4),
            "len_chars_quantiles": {q: float(df["n_chars"].quantile(float(q))) for q in ("0.05", "0.25", "0.5", "0.75", "0.95")},
            "short_lt10_chars": int((df["n_chars"] < 10).sum()),
            "with_zero_width": int(df["has_zw"].sum()),
            "zero_width_chars_total": int(df["n_zw"].sum()),
            "with_url": int(df["has_url"].sum()),
            "with_emoji": int(df["has_emoji"].sum()),
            "with_assamese_letters": int(df["has_assamese"].sum()),
            "bengali_frac_lt_0_5": int((df["bengali_frac"] < 0.5).sum()),
            "changed_by_prepare_original": int((df["prepared"] != df["sentence"]).sum()),
        }

    # duplicates: exact and canonical, within and across splits
    for key, name in (("sentence", "exact"), ("canon", "canonical")):
        grp = all_df.groupby(key)["split"].agg(list)
        dup = grp[grp.map(len) > 1]
        within = Counter()
        cross = 0
        conflict = 0
        for text, splits in dup.items():
            if len(set(splits)) > 1:
                cross += 1
            else:
                within[splits[0]] += 1
        lab = all_df.groupby(key)["label"].nunique()
        conflict = int((lab > 1).sum())
        test_rows_with_train_twin = int(all_df[(all_df.split == "test")][key].isin(set(all_df[all_df.split == "train"][key])).sum())
        val_rows_with_train_twin = int(all_df[(all_df.split == "val")][key].isin(set(all_df[all_df.split == "train"][key])).sum())
        report["duplicates"][name] = {
            "duplicate_groups": int(len(dup)),
            "groups_within_split": dict(within),
            "groups_crossing_splits": cross,
            "label_conflict_groups": conflict,
            "test_rows_with_a_train_twin": test_rows_with_train_twin,
            "val_rows_with_a_train_twin": val_rows_with_train_twin,
        }

    # character inventory of interest
    zw_counter = Counter(ch for s in all_df["sentence"] for ch in s if ch in ZW)
    report["characters"]["zero_width_by_codepoint"] = {f"U+{ord(k):04X}": v for k, v in zw_counter.items()}
    punct = Counter(ch for s in all_df["sentence"] for ch in s if ch in "।|!.,:-?")
    report["characters"]["punctuation_counts"] = dict(punct)
    f1_sites = Counter()
    for s in all_df["prepared"]:
        for pat in ("য়", "ড়", "ঢ়", "ো", "ৌ"):
            f1_sites[pat.encode("unicode_escape").decode()] += s.count(pat)
    report["characters"]["f1_canonical_sites_total"] = dict(f1_sites)
    report["characters"]["sentences_with_ra_or_ba"] = int(all_df["prepared"].map(lambda s: ("র" in s) or ("ব" in s)).sum())

    # eligibility for the evaluation sample (test, >= 10 chars, majority Bengali)
    test = all_df[all_df.split == "test"]
    elig = test[(test.n_chars >= 10) & (test.bengali_frac >= 0.5) & (~test.has_url)]
    report["evaluation_pool"] = {
        "test_rows": int(len(test)),
        "eligible_rows": int(len(elig)),
        "eligible_label_counts": {str(k): int(v) for k, v in elig["label"].value_counts().sort_index().items()},
        "rule": "test split, >=10 chars, >=50% Bengali code points, no URL",
    }

    # row manifest with stable ids
    all_df["row_id"] = [f"{s}:{i}" for s, i in zip(all_df["split"], all_df.groupby("split").cumcount())]
    all_df["sha1_sentence"] = all_df["sentence"].map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:12])
    all_df[["row_id", "split", "label", "n_chars", "has_zw", "has_url", "has_emoji", "has_assamese", "bengali_frac", "sha1_sentence"]].to_csv(MAN / "bd_shs_rows.csv", index=False)

    (OUT / "g0_dataset_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

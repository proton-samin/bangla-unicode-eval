"""Semantic review scoring (SSOT section 6): agreement, Cohen's kappa, per-family
preservation rates, and the 95% admission rule.

Usage: python src/stats/review_agreement.py [review_A.csv review_B.csv]
Defaults to results/validation/review/review_REVIEWER1.csv and review_REVIEWER2.csv.
Writes results/validation/review/review_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "results" / "validation" / "review"
a_path = Path(sys.argv[1]) if len(sys.argv) > 2 else REV / "review_REVIEWER1.csv"
b_path = Path(sys.argv[2]) if len(sys.argv) > 2 else REV / "review_REVIEWER2.csv"

a = pd.read_csv(a_path, encoding="utf-8-sig").set_index("item_id")
b = pd.read_csv(b_path, encoding="utf-8-sig").set_index("item_id")
key = pd.read_csv(REV / "review_key.csv", encoding="utf-8-sig").set_index("item_id")
cols = ["same_meaning_YN", "same_label_YN", "readable_YN"]
for df, name in ((a, a_path.name), (b, b_path.name)):
    missing = df[cols].isna().any(axis=1) | (df[cols] == "").any(axis=1)
    if missing.any():
        raise SystemExit(f"{name}: {int(missing.sum())} rows not filled: {list(df.index[missing][:10])}")
    for c in cols:
        bad = ~df[c].str.upper().isin(["Y", "N"])
        if bad.any():
            raise SystemExit(f"{name}: column {c} has values other than Y/N in rows {list(df.index[bad][:10])}")

summary = {"n_items": int(len(key)), "columns": {}, "families": {}}
for c in cols:
    ya, yb = a[c].str.upper() == "Y", b[c].str.upper() == "Y"
    summary["columns"][c] = {
        "agreement": float((ya == yb).mean()),
        "kappa": float(cohen_kappa_score(ya, yb)) if ya.nunique() > 1 or yb.nunique() > 1 else 1.0,
        "both_yes": float((ya & yb).mean()),
        "disagreements": [i for i in a.index if ya[i] != yb[i]],
    }
for fam, items in key.groupby("family").groups.items():
    ya = (a.loc[items, "same_meaning_YN"].str.upper() == "Y") & (a.loc[items, "same_label_YN"].str.upper() == "Y")
    yb = (b.loc[items, "same_meaning_YN"].str.upper() == "Y") & (b.loc[items, "same_label_YN"].str.upper() == "Y")
    both = ya & yb
    summary["families"][fam] = {
        "n": int(len(items)),
        "preserved_both_reviewers": float(both.mean()),
        "needs_adjudication": [i for i in items if ya[i] != yb[i]],
        "admitted_95pct_rule_before_adjudication": bool(both.mean() >= 0.95),
    }
(REV / "review_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))

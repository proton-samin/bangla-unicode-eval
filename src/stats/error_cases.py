"""Structured error-analysis sample (SSOT 11.5): per model, 20 flipped and 20
non-flipped F1 cases at budget 3, plus every benign comment whose prediction
changed under BN-norm applied to the raw text. Texts are joined back from the
sample/variants files so the authors can read them.

Usage: python src/stats/error_cases.py --dir results/final --split test
Writes --dir/analysis/error_cases_<tag>.csv
"""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--n", type=int, default=20)
args = ap.parse_args()
d = Path(args.dir)
sample = pd.read_csv(d / f"sample_{args.split}.csv").set_index("row_id")
variants = pd.read_csv(d / f"variants_{args.split}.csv")
for f in sorted(d.glob(f"predictions_{args.split}_*.csv")):
    tag = f.stem.replace(f"predictions_{args.split}_", "")
    p = pd.read_csv(f)
    clean = p[(p.family == "clean") & (p.defense == "none")].set_index("row_id")
    tp = clean.index[(clean.label == 1) & (clean.pred_harmful == 1)]
    f1 = p[(p.family == "F1") & (p.budget.astype(str) == "3") & (p.defense == "none") & (p.n_edits > 0)].set_index("row_id")
    f1 = f1.loc[f1.index.intersection(tp)]
    flipped = f1[f1.pred_harmful == 0].sample(n=min(args.n, (f1.pred_harmful == 0).sum()), random_state=1)
    kept = f1[f1.pred_harmful == 1].sample(n=min(args.n, (f1.pred_harmful == 1).sum()), random_state=1)
    vtext = variants[(variants.family == "F1") & (variants.budget.astype(str) == "3")].set_index("row_id")["text"]
    rows = []
    for name, g in (("F1_flipped", flipped), ("F1_kept", kept)):
        for rid, r in g.iterrows():
            rows.append({"case": name, "row_id": rid, "label": r.label, "p_clean": clean.loc[rid, "p_harmful"], "p_perturbed": r.p_harmful,
                         "tokens_clean": clean.loc[rid, "n_tokens"], "tokens_perturbed": r.n_tokens,
                         "clean_text": sample.loc[rid, "text"], "perturbed_text": vtext.loc[rid]})
    raw0 = p[(p.family == "raw") & (p.defense == "none")].set_index("row_id")
    rawbn = p[(p.family == "raw") & (p.defense == "bn")].set_index("row_id")
    ben = raw0.index[(raw0.label == 0) & (raw0.pred_harmful != rawbn.loc[raw0.index, "pred_harmful"])]
    for rid in ben:
        rows.append({"case": "benign_changed_by_BNnorm_on_raw", "row_id": rid, "label": 0, "p_clean": raw0.loc[rid, "p_harmful"],
                     "p_perturbed": rawbn.loc[rid, "p_harmful"], "tokens_clean": raw0.loc[rid, "n_tokens"], "tokens_perturbed": rawbn.loc[rid, "n_tokens"],
                     "clean_text": sample.loc[rid, "raw"], "perturbed_text": sample.loc[rid, "text"]})
    out = d / "analysis" / f"error_cases_{tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(tag, pd.DataFrame(rows)["case"].value_counts().to_dict())

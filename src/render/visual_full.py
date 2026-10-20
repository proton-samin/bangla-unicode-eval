"""Exhaustive visual check of every perturbed variant actually evaluated
(SSOT section 5, extended): for each variant in results/<dir>/variants_<split>.csv
with at least one edit, render original and perturbed in each font and record
the verdict. Reports the share of variants that are pixel-identical per family,
budget and font.

Usage: python src/render/visual_full.py --dir results/final --split test
Writes --dir/analysis/visual_full.csv and visual_full_summary.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "render"))
from render import available_fonts, visual_check  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--budgets", nargs="+", default=["1", "3", "len"])
args = ap.parse_args()
d = Path(args.dir)
sample = pd.read_csv(d / f"sample_{args.split}.csv").set_index("row_id")
v = pd.read_csv(d / f"variants_{args.split}.csv")
v = v[(~v.family.isin(["clean", "raw"])) & (v.n_edits > 0) & (v.budget.astype(str).isin(args.budgets))]
fonts = available_fonts()
rows = []
cache = {}
for i, r in enumerate(v.itertuples(index=False)):
    orig = sample.loc[r.row_id, "text"]
    for f in fonts:
        k = (r.text, f)
        if k not in cache:
            try:
                res = visual_check(orig, r.text, f)
                cache[k] = (res.verdict, res.diff_pixels, res.diff_fraction)
            except Exception as e:  # keep going; record the failure
                cache[k] = ("error", -1, -1.0)
        verdict, dpx, dfrac = cache[k]
        rows.append({"row_id": r.row_id, "family": r.family, "budget": str(r.budget), "font": f,
                     "verdict": verdict, "diff_pixels": dpx, "diff_fraction": dfrac})
    if i % 2000 == 0:
        print(f"{i}/{len(v)} variants", flush=True)
df = pd.DataFrame(rows)
an = d / "analysis"; an.mkdir(exist_ok=True)
df.to_csv(an / "visual_full.csv", index=False)
summary = {}
for (fam, bud, f), g in df.groupby(["family", "budget", "font"]):
    summary.setdefault(fam, {}).setdefault(bud, {})[f] = {
        "n": int(len(g)), "identical": round(float((g.verdict == "identical").mean()), 4),
        "identical_or_near": round(float(g.verdict.isin(["identical", "near"]).mean()), 4),
        "perceptible": int((g.verdict == "perceptible").sum()), "errors": int((g.verdict == "error").sum())}
(an / "visual_full_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
for fam, buds in summary.items():
    for bud, fs in buds.items():
        print(fam, bud, "  ".join(f"{f}: ident {x['identical']:.4f} ({x['perceptible']} perceptible of {x['n']})" for f, x in fs.items()))

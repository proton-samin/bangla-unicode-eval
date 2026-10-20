"""Aggregate M3 seeds (and any other models) from an analysis folder into the
numbers the paper tables need: per-model primary evasion at budget 3 under each
defense, clean accuracy, and for seed groups the mean and standard deviation
across seeds plus the pooled k/n.

Usage: python src/stats/aggregate_seeds.py --dir results/kaggle/results/final [--pattern m3_seed]
Prints a compact table and writes --dir/analysis/aggregate.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--pattern", default="m3_seed")
args = ap.parse_args()
an = Path(args.dir) / "analysis"
summary = json.load(open(an / "summary.json", encoding="utf-8"))
prim = pd.read_csv(an / "primary.csv")
cc = pd.read_csv(an / "clean_cost.csv")

rows = []
for r in summary:
    m = r["model"]
    p = prim[(prim.model == m) & (prim.budget.astype(str) == "3")]
    for fam in ("F1", "F2", "F2nbsp", "F5"):
        for d in ("none", "nfc", "bn"):
            q = p[(p.family == fam) & (p.defense == d)]
            if len(q):
                q = q.iloc[0]
                rows.append({"model": m, "seed_group": args.pattern in m, "family": fam, "defense": d,
                             "k": int(q.k_evaded), "n": int(q.n_eligible), "evasion": float(q.evasion),
                             "boot_lo": float(q.boot_lo), "boot_hi": float(q.boot_hi)})
    c = cc[cc.model == m].set_index("defense")
    rows.append({"model": m, "seed_group": args.pattern in m, "family": "clean", "defense": "none",
                 "k": 0, "n": int(r["n"]["sentences"]), "evasion": float(r["n"]["clean_accuracy_none"]),
                 "boot_lo": float("nan"), "boot_hi": float("nan")})
    for d in ("nfc", "bn"):
        if d in c.index:
            rows.append({"model": m, "seed_group": args.pattern in m, "family": "raw_fpr_increase_pp", "defense": d,
                         "k": 0, "n": int(c.loc[d, "n"]), "evasion": float(c.loc[d, "fpr_increase_pp"]),
                         "boot_lo": float("nan"), "boot_hi": float("nan")})
df = pd.DataFrame(rows)
df.to_csv(an / "aggregate.csv", index=False)

seeds = df[df.seed_group]
if len(seeds):
    g = seeds.groupby(["family", "defense"]).agg(mean=("evasion", "mean"), sd=("evasion", "std"), k=("k", "sum"), n=("n", "sum"), seeds=("model", "nunique"))
    g["pooled"] = g.k / g.n.where(g.n > 0)
    print(f"== seed group '{args.pattern}' ({int(g.seeds.max())} seeds): mean, sd across seeds; pooled k/n ==")
    print(g.round(4).to_string())
others = df[~df.seed_group]
if len(others):
    print("\n== other models ==")
    print(others.pivot_table(index=["model", "family"], columns="defense", values="evasion").round(4).to_string())

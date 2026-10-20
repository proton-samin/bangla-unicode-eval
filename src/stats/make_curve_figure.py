"""Figure 2: evasion rate versus edit budget per backbone group, both datasets.

Usage: python src/stats/make_curve_figure.py --root results/kaggle_round2/results
Writes figures/budget_curve.pdf and .png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--root", default="results/kaggle_round2/results")
args = ap.parse_args()
C = pd.read_csv(ROOT / args.root / "round2_curve.csv")
C = C[C.budget != "len"]
C["budget"] = C.budget.astype(int)
ORDER = ["own XLM-R", "own MuRIL", "own BanglaBERT", "public MuRIL (M1)", "public BanglaBERT (M2)"]
STYLE = {"own XLM-R": ("C0", "o"), "own MuRIL": ("C1", "s"), "own BanglaBERT": ("C2", "^"), "public MuRIL (M1)": ("C1", "s"), "public BanglaBERT (M2)": ("C2", "^")}
fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.4), sharex=True)
for i, ds in enumerate(("bdshs", "blp25")):
    for j, fam in enumerate(("F1", "F2")):
        ax = axes[j, i]
        for g in ORDER:
            q = C[(C.dataset == ds) & (C.family == fam) & (C.group == g)].sort_values("budget")
            if q.empty:
                continue
            col, mk = STYLE[g]
            ls = "-" if g.startswith("own") else "--"
            ax.errorbar(q.budget, 100 * q.evasion_mean, yerr=100 * q.evasion_sd, color=col, marker=mk, linestyle=ls, linewidth=1.0, markersize=3, capsize=1.5, label=g)
        ax.set_xticks([1, 3, 5, 10])
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_title({"bdshs": "BD-SHS", "blp25": "BLP-2025"}[ds], fontsize=8)
        if i == 0:
            ax.set_ylabel({"F1": "F1 canonical\nevasion (%)", "F2": "F2 invisible\nevasion (%)"}[fam], fontsize=7.5)
        if j == 1:
            ax.set_xlabel("edits per comment", fontsize=7.5)
        ax.tick_params(labelsize=7)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=(0, 0.09, 1, 1), pad=0.4)
out = ROOT / "figures" / "budget_curve"
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=160)
print("wrote", out.with_suffix(".pdf"))

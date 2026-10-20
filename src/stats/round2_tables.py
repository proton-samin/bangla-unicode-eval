"""Round 2 tables: per backbone group (three seeds) and per public checkpoint,
on each dataset. Reads the analysis/ folders written by analyze.py.

Usage: python src/stats/round2_tables.py --root results/kaggle_round2/results
Writes <root>/round2_summary.csv, round2_perchar.csv, round2_curve.csv and prints
the numbers the paper needs (mean ± sd over seeds, pooled k/n, seed ranges).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="results/kaggle_round2/results")
args = ap.parse_args()
root = Path(args.root)

GROUPS = {"own XLM-R": "xlmr_seed", "own MuRIL": "muril_seed", "own BanglaBERT": "banglabert_seed",
          "public MuRIL (M1)": "bengali-abusive-MuRIL", "public BanglaBERT (M2)": "enhanced-banglabert"}
FAMS = ["F1", "F2", "F2nbsp", "F5"]
SUBS = ["F2_zwnj", "F2_zwj", "F2_zwsp", "F2_wj", "F2_shy"]


def group_of(model: str):
    for g, pat in GROUPS.items():
        if pat in model:
            return g
    return None


rows, perchar, curve = [], [], []
for ds in ("bdshs", "blp25"):
    an = root / f"{ds}_final" / "analysis"
    prim = pd.read_csv(an / "primary.csv"); prim["group"] = prim.model.map(group_of)
    cc = pd.read_csv(an / "clean_cost.csv"); cc["group"] = cc.model.map(group_of)
    summ = {r["model"]: r for r in json.load(open(an / "summary.json", encoding="utf-8"))}
    rec = pd.read_csv(an / "recall.csv"); rec["group"] = rec.model.map(group_of)
    for g in GROUPS:
        pg = prim[prim.group == g]
        if pg.empty:
            continue
        models = sorted(pg.model.unique())
        acc = [summ[m]["n"]["clean_accuracy_none"] for m in models]
        tps = [summ[m]["n"]["clean_true_positives_none"] for m in models]
        for fam in FAMS:
            for d in ("none", "nfc", "bn"):
                q = pg[(pg.family == fam) & (pg.budget.astype(str) == "3") & (pg.defense == d)]
                if q.empty:
                    continue
                rows.append({"dataset": ds, "group": g, "n_models": len(models), "family": fam, "defense": d,
                             "evasion_mean": q.evasion.mean(), "evasion_sd": q.evasion.std(ddof=0) if len(q) > 1 else 0.0,
                             "evasion_min": q.evasion.min(), "evasion_max": q.evasion.max(),
                             "k_pooled": int(q.k_evaded.sum()), "n_pooled": int(q.n_eligible.sum()),
                             "ci_lo_min": q.boot_lo.min(), "ci_hi_max": q.boot_hi.max(),
                             "clean_acc_mean": float(np.mean(acc)), "clean_acc_sd": float(np.std(acc)),
                             "tp_mean": float(np.mean(tps))})
        # recall loss on all harmful (F1, F2 at budget 3, D0)
        for fam in ("F1", "F2"):
            q = rec[(rec.group == g) & (rec.family == fam) & (rec.budget.astype(str) == "3") & (rec.defense == "none")]
            if len(q):
                rows.append({"dataset": ds, "group": g, "n_models": len(models), "family": fam, "defense": "recall_loss_pp",
                             "evasion_mean": q.recall_loss_pp.mean(), "evasion_sd": q.recall_loss_pp.std(ddof=0) if len(q) > 1 else 0.0,
                             "evasion_min": q.boot_lo_pp.min(), "evasion_max": q.boot_hi_pp.max(), "k_pooled": 0, "n_pooled": int(q.n_harmful.sum()),
                             "ci_lo_min": np.nan, "ci_hi_max": np.nan, "clean_acc_mean": float(np.mean(acc)), "clean_acc_sd": float(np.std(acc)), "tp_mean": float(np.mean(tps))})
        # clean cost on raw text, D3 and NFC
        for d in ("nfc", "bn"):
            q = cc[(cc.group == g) & (cc.defense == d)]
            if len(q):
                rows.append({"dataset": ds, "group": g, "n_models": len(models), "family": "raw_text", "defense": d,
                             "evasion_mean": q.fpr_increase_pp.mean(), "evasion_sd": q.fpr_increase_pp.std(ddof=0) if len(q) > 1 else 0.0,
                             "evasion_min": q.accuracy_drop_pp.mean(), "evasion_max": float(q.non_inferior.all()), "k_pooled": int(q.n_predictions_changed.sum()), "n_pooled": int(q.n.sum()),
                             "ci_lo_min": np.nan, "ci_hi_max": np.nan, "clean_acc_mean": float(np.mean(acc)), "clean_acc_sd": float(np.std(acc)), "tp_mean": float(np.mean(tps))})
        # per-character sub-families at budget 3, D0
        for sub in SUBS:
            q = pg[(pg.family == sub) & (pg.budget.astype(str) == "3") & (pg.defense == "none")]
            if len(q):
                perchar.append({"dataset": ds, "group": g, "subfamily": sub, "evasion_mean": q.evasion.mean(), "evasion_sd": q.evasion.std(ddof=0) if len(q) > 1 else 0.0,
                                "k_pooled": int(q.k_evaded.sum()), "n_pooled": int(q.n_eligible.sum())})
        # budget curve, D0
        for fam in ("F1", "F2"):
            for b in ("1", "3", "5", "10", "len"):
                q = pg[(pg.family == fam) & (pg.budget.astype(str) == b) & (pg.defense == "none")]
                if len(q):
                    curve.append({"dataset": ds, "group": g, "family": fam, "budget": b, "evasion_mean": q.evasion.mean(),
                                  "evasion_sd": q.evasion.std(ddof=0) if len(q) > 1 else 0.0, "k_pooled": int(q.k_evaded.sum()), "n_pooled": int(q.n_eligible.sum())})

S = pd.DataFrame(rows); P = pd.DataFrame(perchar); C = pd.DataFrame(curve)
S.to_csv(root / "round2_summary.csv", index=False); P.to_csv(root / "round2_perchar.csv", index=False); C.to_csv(root / "round2_curve.csv", index=False)
pd.set_option("display.width", 220)
for ds in ("bdshs", "blp25"):
    print(f"\n===== {ds}: evasion % at budget 3 (mean over seeds [min, max]); clean acc =====")
    s = S[(S.dataset == ds) & (S.family.isin(FAMS)) & (S.defense == "none")].copy()
    s["cell"] = s.apply(lambda r: f"{100*r.evasion_mean:.1f} [{100*r.evasion_min:.1f},{100*r.evasion_max:.1f}] k/n={r.k_pooled}/{r.n_pooled}", axis=1)
    print(s.pivot(index="group", columns="family", values="cell").to_string())
    print(s.drop_duplicates("group")[["group", "n_models", "clean_acc_mean", "clean_acc_sd", "tp_mean"]].round(4).to_string(index=False))
    print(f"--- {ds}: under defenses (F1 and F2 evasion %, budget 3) ---")
    d = S[(S.dataset == ds) & (S.family.isin(["F1", "F2"])) & (S.defense.isin(["none", "nfc", "bn"]))]
    print(d.pivot_table(index=["group", "family"], columns="defense", values="evasion_mean").mul(100).round(2).to_string())
    print(f"--- {ds}: recall loss pp (F1, F2) and raw-text FPR increase pp (nfc, bn), non_inferior flag in evasion_max ---")
    print(S[(S.dataset == ds) & (S.defense.isin(["recall_loss_pp"]))][["group", "family", "evasion_mean", "evasion_min", "evasion_max"]].round(2).to_string(index=False))
    print(S[(S.dataset == ds) & (S.family == "raw_text")][["group", "defense", "evasion_mean", "evasion_min", "evasion_max", "k_pooled"]].round(3).to_string(index=False))
    print(f"--- {ds}: per-character F2 at budget 3 ---")
    print(P[P.dataset == ds].pivot(index="group", columns="subfamily", values="evasion_mean").mul(100).round(2).to_string())
    print(f"--- {ds}: budget curve (evasion %) ---")
    print(C[C.dataset == ds].pivot_table(index=["group", "family"], columns="budget", values="evasion_mean").mul(100).round(2)[["1", "3", "5", "10", "len"]].to_string())

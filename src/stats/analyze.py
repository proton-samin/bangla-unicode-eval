"""Statistical analysis of paired predictions (SSOT section 11).

Reads every predictions_<split>_<tag>.csv in --dir and writes --dir/analysis/:
  summary.json         all estimates with intervals and n
  primary.csv          evasion rate among clean true positives per model x family x budget x defense
  recall.csv           paired recall loss on all harmful rows
  benign.csv           false-positive rate on benign rows (perturbed, and clean under each defense)
  clean_cost.csv       clean-text accuracy / FPR change per defense vs none, with CI and margin test
  tokens.csv           token-count deltas per family (tokenization sensitivity)
  confirmatory.csv     H1-H4 with Holm-adjusted p-values
  tab_primary.tex, tab_defense.tex   LaTeX bodies for the paper tables

Estimates: point estimate, 95% Clopper-Pearson interval for single proportions,
95% percentile bootstrap (B resamples, resampling SENTENCES so repeated
transformations of one sentence stay together) for every rate and difference.
Confirmatory tests: exact binomial one-sided test of evasion > NULL_RATE for
H1/H2, exact McNemar for H3 (defense lowers evasion), and a non-inferiority
check for H4 (clean accuracy and FPR change <= MARGIN pp). Holm-Bonferroni over
the confirmatory family. Everything else is labelled exploratory.

Usage: python src/stats/analyze.py --dir results/pilot --split val [--B 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

NULL_RATE = 0.01      # "negligible evasion" null for H1/H2 (pre-specified, SSOT 11.4)
MARGIN_PP = 0.5       # non-inferiority margin for clean-text cost (SSOT 11.2 H4)
PRIMARY_BUDGET = "3"
ATTACK_FAMILIES = ("F2", "F5")      # confirmatory attack families after the visual gate
CONTROL_FAMILIES = ("F1", "F2nbsp")


def cp_interval(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def boot_mean(x: np.ndarray, B: int, rng: np.random.Generator):
    if len(x) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def boot_diff(a: np.ndarray, b: np.ndarray, B: int, rng: np.random.Generator):
    """Paired difference mean(a) - mean(b), resampling the same sentence indices."""
    if len(a) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(a), size=(B, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * stats.binom.cdf(k, n, 0.5)))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj.tolist()


def analyse_model(pred: pd.DataFrame, B: int, rng) -> dict:
    model = pred["model"].iloc[0]
    out = {"model": model, "primary": [], "recall": [], "benign": [], "clean_cost": [], "tokens": [], "confirmatory": []}
    clean = pred[pred.family == "clean"].pivot(index="row_id", columns="defense", values="pred_harmful")
    labels = pred[pred.family == "clean"].drop_duplicates("row_id").set_index("row_id")["label"]
    tp_ids = clean.index[(labels.loc[clean.index] == 1) & (clean["none"] == 1)]
    harm_ids = labels.index[labels == 1]
    ben_ids = labels.index[labels == 0]
    out["n"] = {"sentences": int(len(labels)), "harmful": int(len(harm_ids)), "benign": int(len(ben_ids)),
                "clean_true_positives_none": int(len(tp_ids)),
                "clean_accuracy_none": float((clean["none"] == labels.loc[clean.index]).mean())}

    pert = pred[(~pred.family.isin(["clean", "raw"])) & (pred.n_edits > 0)]
    for (fam, bud, d), g in pert.groupby(["family", "budget", "defense"]):
        g = g.set_index("row_id")
        # primary: evasion among clean true positives (deployment view: the
        # classifier caught the clean comment without any defense; does the
        # perturbed comment, passed through defense d, still get caught?)
        gt = g.loc[g.index.intersection(tp_ids)]
        k, n = int((gt.pred_harmful == 0).sum()), int(len(gt))
        rate = k / n if n else float("nan")
        lo_cp, hi_cp = cp_interval(k, n)
        lo_b, hi_b = boot_mean((gt.pred_harmful.values == 0).astype(float), B, rng)
        # attack-only view: eligible = clean true positives UNDER THE SAME defense,
        # so the defense's own cost on clean text is removed. For the control
        # families this must be zero under NFC/NFKC.
        tp_d = clean.index[(labels.loc[clean.index] == 1) & (clean[d] == 1)]
        gd = g.loc[g.index.intersection(tp_d)]
        k_d, n_d = int((gd.pred_harmful == 0).sum()), int(len(gd))
        out["primary"].append({"family": fam, "budget": str(bud), "defense": d, "k_evaded": k, "n_eligible": n,
                               "evasion": rate, "cp_lo": lo_cp, "cp_hi": hi_cp, "boot_lo": lo_b, "boot_hi": hi_b,
                               "k_same_defense": k_d, "n_same_defense": n_d,
                               "evasion_same_defense": k_d / n_d if n_d else float("nan")})
        # recall loss on all harmful with this variant
        gh = g.loc[g.index.intersection(harm_ids)]
        if len(gh):
            rc = clean.loc[gh.index, d].values.astype(float)
            rp = gh.pred_harmful.values.astype(float)
            lo, hi = boot_diff(rc, rp, B, rng)
            out["recall"].append({"family": fam, "budget": str(bud), "defense": d, "n_harmful": int(len(gh)),
                                  "recall_clean": float(rc.mean()), "recall_perturbed": float(rp.mean()),
                                  "recall_loss_pp": float((rc.mean() - rp.mean()) * 100), "boot_lo_pp": lo * 100, "boot_hi_pp": hi * 100})
        # benign: false positives under perturbation
        gb = g.loc[g.index.intersection(ben_ids)]
        if len(gb):
            fc = clean.loc[gb.index, d].values.astype(float)
            fp = gb.pred_harmful.values.astype(float)
            lo, hi = boot_diff(fp, fc, B, rng)
            out["benign"].append({"family": fam, "budget": str(bud), "defense": d, "n_benign": int(len(gb)),
                                  "fpr_clean": float(fc.mean()), "fpr_perturbed": float(fp.mean()),
                                  "fpr_change_pp": float((fp.mean() - fc.mean()) * 100), "boot_lo_pp": lo * 100, "boot_hi_pp": hi * 100})

    # clean-text cost of each defense (accuracy and FPR vs none), non-inferiority at MARGIN_PP.
    # Measured on the RAW dataset text when available (real-world input with its
    # pre-existing zero-width characters), otherwise on the prepared clean text.
    if (pred.family == "raw").any():
        base = pred[pred.family == "raw"].pivot(index="row_id", columns="defense", values="pred_harmful")
        out["clean_cost_basis"] = "raw dataset text"
    else:
        base = clean
        out["clean_cost_basis"] = "prepared clean text"
    y = labels.loc[base.index].values
    acc_none = (base["none"].values == y).astype(float)
    fpr_none = (base["none"].values[y == 0] == 1).astype(float)
    for d in base.columns:
        acc_d = (base[d].values == y).astype(float)
        fpr_d = (base[d].values[y == 0] == 1).astype(float)
        lo_a, hi_a = boot_diff(acc_none, acc_d, B, rng)       # accuracy drop
        lo_f, hi_f = boot_diff(fpr_d, fpr_none, B, rng)       # FPR increase
        out["clean_cost"].append({"defense": d, "n": int(len(y)), "accuracy": float(acc_d.mean()), "fpr": float(fpr_d.mean()),
                                  "accuracy_drop_pp": float((acc_none.mean() - acc_d.mean()) * 100), "acc_drop_ci_pp": [lo_a * 100, hi_a * 100],
                                  "fpr_increase_pp": float((fpr_d.mean() - fpr_none.mean()) * 100), "fpr_inc_ci_pp": [lo_f * 100, hi_f * 100],
                                  "n_predictions_changed": int((base[d].values != base["none"].values).sum()),
                                  "non_inferior": bool(hi_a * 100 <= MARGIN_PP and hi_f * 100 <= MARGIN_PP)})

    # tokenization sensitivity (defense none)
    if "n_tokens" in pred.columns:
        base = pred[(pred.family == "clean") & (pred.defense == "none")].set_index("row_id")["n_tokens"]
        for (fam, bud), g in pert[pert.defense == "none"].groupby(["family", "budget"]):
            g = g.set_index("row_id")
            delta = g["n_tokens"] - base.loc[g.index]
            flips = g.loc[g.index.intersection(tp_ids), "pred_harmful"] == 0
            out["tokens"].append({"family": fam, "budget": str(bud), "n": int(len(g)), "mean_token_delta": float(delta.mean()),
                                  "share_token_count_changed": float((delta != 0).mean()),
                                  "mean_token_delta_flipped": float(delta.loc[flips.index[flips]].mean()) if flips.any() else float("nan"),
                                  "mean_token_delta_not_flipped": float(delta.loc[flips.index[~flips]].mean()) if (~flips).any() else float("nan")})

    # confirmatory hypotheses
    tests = []
    prim = pd.DataFrame(out["primary"])
    for fam in ATTACK_FAMILIES:
        row = prim[(prim.family == fam) & (prim.budget == PRIMARY_BUDGET) & (prim.defense == "none")]
        if len(row):
            r = row.iloc[0]
            p = float(stats.binom.sf(r.k_evaded - 1, r.n_eligible, NULL_RATE)) if r.n_eligible else float("nan")
            tests.append({"hypothesis": f"H1/H2 {fam}: evasion > {NULL_RATE:.0%} at budget {PRIMARY_BUDGET}, D0",
                          "k": int(r.k_evaded), "n": int(r.n_eligible), "estimate": float(r.evasion), "p_raw": p})
        # H3: BN-norm lowers evasion vs none (McNemar on the same sentences)
        g0 = pert[(pert.family == fam) & (pert.budget == PRIMARY_BUDGET) & (pert.defense == "none")].set_index("row_id")
        g3 = pert[(pert.family == fam) & (pert.budget == PRIMARY_BUDGET) & (pert.defense == "bn")].set_index("row_id")
        ids = g0.index.intersection(g3.index).intersection(tp_ids)
        if len(ids):
            e0 = (g0.loc[ids, "pred_harmful"] == 0).values
            e3 = (g3.loc[ids, "pred_harmful"] == 0).values
            b, c = int((e0 & ~e3).sum()), int((~e0 & e3).sum())
            tests.append({"hypothesis": f"H3 {fam}: BN-norm evasion < D0 evasion (McNemar)", "k": b, "n": b + c,
                          "estimate": float(e0.mean() - e3.mean()), "p_raw": mcnemar_exact(b, c) if b > c else 1.0})
    if tests:
        adj = holm([t["p_raw"] for t in tests])
        for t, a in zip(tests, adj):
            t["p_holm"] = a
            t["reject_at_0.05"] = bool(a < 0.05)
    cc = pd.DataFrame(out["clean_cost"]).set_index("defense")
    if "bn" in cc.index:
        tests.append({"hypothesis": f"H4: BN-norm clean accuracy drop and FPR increase <= {MARGIN_PP} pp (CI upper bounds)",
                      "estimate": float(cc.loc["bn", "accuracy_drop_pp"]), "p_raw": float("nan"), "p_holm": float("nan"),
                      "reject_at_0.05": bool(cc.loc["bn", "non_inferior"])})
    out["confirmatory"] = tests
    return out


def latex_primary(primary: pd.DataFrame, models: list[str]) -> str:
    fams = [f for f in ("F1", "F2", "F2nbsp", "F5") if f in set(primary.family)]
    lines = []
    for fam in fams:
        cells = []
        for m in models:
            r = primary[(primary.model == m) & (primary.family == fam) & (primary.budget == PRIMARY_BUDGET) & (primary.defense == "none")]
            cells.append(f"{100*r.evasion.iloc[0]:.1f} [{100*r.boot_lo.iloc[0]:.1f}, {100*r.boot_hi.iloc[0]:.1f}]" if len(r) and r.n_eligible.iloc[0] else "--")
        label = {"F1": "F1 canonical (control)", "F2": "F2 invisible", "F2nbsp": "F2 NBSP (control)", "F5": "F5 khanda ta"}[fam]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    ns = []
    for m in models:
        r = primary[(primary.model == m) & (primary.budget == PRIMARY_BUDGET) & (primary.defense == "none")]
        ns.append(str(int(r.n_eligible.max())) if len(r) else "--")
    lines.append(r"\midrule")
    lines.append("$n$ eligible (max over families) & " + " & ".join(ns) + r" \\")
    return "\n".join(lines)


def latex_defense(primary: pd.DataFrame, clean_cost: pd.DataFrame, models: list[str]) -> str:
    defs = ["none", "nfc", "nfkc", "bn"]
    lines = []
    for fam in [f for f in ("F1", "F2", "F5") if f in set(primary.family)]:
        cells = []
        for d in defs:
            r = primary[(primary.family == fam) & (primary.budget == PRIMARY_BUDGET) & (primary.defense == d)]
            # pool across models by summing k and n (exploratory presentation)
            k, n = int(r.k_evaded.sum()), int(r.n_eligible.sum())
            cells.append(f"{100*k/n:.1f}" if n else "--")
        lines.append(f"{fam} evasion (\\%) & " + " & ".join(cells) + r" \\")
    for metric, name in (("accuracy_drop_pp", "Clean acc. drop (pp)"), ("fpr_increase_pp", "Clean FPR increase (pp)")):
        cells = []
        for d in defs:
            r = clean_cost[clean_cost.defense == d]
            cells.append("0" if d == "none" else f"{r[metric].mean():.2f}")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261017)
    args = ap.parse_args()
    d = Path(args.dir)
    files = sorted(d.glob(f"predictions_{args.split}_*.csv"))
    if not files:
        raise SystemExit(f"no predictions_{args.split}_*.csv in {d}")
    rng = np.random.default_rng(args.seed)
    results = [analyse_model(pd.read_csv(f), args.B, rng) for f in files]
    an = d / "analysis"; an.mkdir(exist_ok=True)
    (an / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    tables = {}
    for key in ("primary", "recall", "benign", "clean_cost", "tokens", "confirmatory"):
        frames = [pd.DataFrame(r[key]).assign(model=r["model"]) for r in results if r[key]]
        if frames:
            tables[key] = pd.concat(frames, ignore_index=True)
            tables[key].to_csv(an / f"{key}.csv", index=False)
    models = [r["model"] for r in results]
    if "primary" in tables:
        (an / "tab_primary.tex").write_text(latex_primary(tables["primary"], models), encoding="utf-8")
        (an / "tab_defense.tex").write_text(latex_defense(tables["primary"], tables["clean_cost"], models), encoding="utf-8")
    # console summary
    for r in results:
        print(f"\n== {r['model']}  n={r['n']}")
        p = pd.DataFrame(r["primary"])
        p = p[(p.budget == PRIMARY_BUDGET)]
        print(p.pivot(index="family", columns="defense", values="evasion").round(3).to_string())
        print(pd.DataFrame(r["clean_cost"]).set_index("defense")[["accuracy", "fpr", "accuracy_drop_pp", "fpr_increase_pp", "non_inferior"]].round(4).to_string())
        for t in r["confirmatory"]:
            print("  ", t["hypothesis"], "| est", round(t["estimate"], 4), "| p_holm", None if t.get("p_holm") != t.get("p_holm") else round(t["p_holm"], 4), "| reject", t["reject_at_0.05"])


if __name__ == "__main__":
    main()

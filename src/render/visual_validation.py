"""Per-transformation visual validation on the development partition (SSOT 5).

For every sub-transformation of every family, sample up to N development
sentences that contain an eligible site, apply exactly one edit of that kind,
render original and perturbed in each font, and record the verdict. A family
(or sub-transformation) is admitted to the confirmatory analysis only if >= 95%
of instances are identical or near-identical in at least two of three fonts.

Usage: python src/render/visual_validation.py [N]
Writes results/validation/visual_by_transformation.json and .csv
"""
from __future__ import annotations

import json
import random
import sys
import unicodedata
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for sub in ("render", "perturb", "normalize"):
    sys.path.insert(0, str(ROOT / "src" / sub))
from render import available_fonts, visual_check, save_pair_png  # noqa: E402
from families import FAMILIES, F2_CHARS, Edit, apply  # noqa: E402
from bn_norm import prepare_original  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SEED = 20261017
OUT = ROOT / "results" / "validation"
FIG = OUT / "visual_examples"
FIG.mkdir(parents=True, exist_ok=True)

val = pd.read_csv(ROOT / "data" / "raw" / "bd_shs" / "val.csv")
val["prepared"] = val["sentence"].astype(str).map(lambda s: prepare_original(s)[0])
sentences = [s for s in val["prepared"] if len(s) >= 10]
rng = random.Random(SEED)


def single_edit_variants(text: str):
    """Yield (code, perturbed) for one edit of each sub-transformation present."""
    for fam, elig in FAMILIES.items():
        sites = elig(text)
        if not sites:
            continue
        if fam == "F2":
            site = rng.choice(sites)
            for code, ch in F2_CHARS.items():
                yield f"F2:{code}", text[: site.pos] + ch + text[site.pos:]
        else:
            by_kind = defaultdict(list)
            for e in sites:
                by_kind[e.kind].append(e)
            for kind, es in by_kind.items():
                e = rng.choice(es)
                yield kind, apply(text, [e], fam, rng)


fonts = available_fonts()
rows = []
counts = defaultdict(int)
order = list(range(len(sentences)))
rng.shuffle(order)
for idx in order:
    s = sentences[idx]
    for code, p in single_edit_variants(s):
        if counts[code] >= N or p == s:
            continue
        counts[code] += 1
        verdicts = {}
        for f in fonts:
            r = visual_check(s, p, f)
            verdicts[f] = r
            rows.append({"code": code, "sentence_idx": idx, **asdict(r)})
        if counts[code] <= 2:  # keep a couple of example images per transformation
            safe = code.split(":")[0] + "_" + "".join(c if c.isalnum() else f"u{ord(c):04x}" for c in code.split(":", 1)[-1])
            for f in fonts:
                save_pair_png(s, p, f, FIG / f"{safe}_{f}_{counts[code]}.png")
    if all(c >= N for c in counts.values()) and len(counts) >= 12:
        pass

df = pd.DataFrame(rows)
df.to_csv(OUT / "visual_by_transformation.csv", index=False)

summary = {}
for code, g in df.groupby("code"):
    per_font = {}
    for f, gf in g.groupby("font"):
        per_font[f] = {
            "n": int(len(gf)),
            "identical": round(float((gf.verdict == "identical").mean()), 3),
            "identical_or_near": round(float(gf.verdict.isin(["identical", "near"]).mean()), 3),
            "median_diff_fraction": round(float(gf.diff_fraction.median()), 4),
        }
    # pass if >= 95% identical-or-near in at least two fonts
    n_pass_fonts = sum(v["identical_or_near"] >= 0.95 for v in per_font.values())
    summary[code] = {"per_font": per_font, "fonts_passing": n_pass_fonts, "admitted": n_pass_fonts >= 2}

(OUT / "visual_by_transformation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
for code, v in sorted(summary.items()):
    flag = "ADMIT " if v["admitted"] else "reject"
    detail = "  ".join(f"{f}:{pf['identical_or_near']:.2f}" for f, pf in v["per_font"].items())
    print(f"{flag} {code:28s} {detail}")

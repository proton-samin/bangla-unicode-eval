"""Semantic and label preservation review sheets (SSOT section 6).

For each admitted family, sample 40 development sentences with at least one
eligible site, apply the budget-3 perturbation, and render original and
perturbed text (Noto Sans Bengali, plus the other fonts stacked) to PNG. Two
reviewer CSVs are produced with empty columns to fill: same_meaning,
same_label, readable (each Y/N) and a free-text note.

Usage: python src/render/review_sheets.py [families...]
Writes results/validation/review/{images, review_REVIEWER1.csv, review_REVIEWER2.csv, review_key.csv}
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
for sub in ("render", "perturb", "normalize"):
    sys.path.insert(0, str(ROOT / "src" / sub))
from render import FONTS, available_fonts, render, DIFF_THRESHOLD  # noqa: E402
from families import perturb  # noqa: E402
from bn_norm import prepare_original  # noqa: E402

FAMS = sys.argv[1:] or ["F1", "F2", "F5"]
N_PER_FAMILY = 40
SEED = 20261017
OUT = ROOT / "results" / "validation" / "review"
IMG = OUT / "images"
IMG.mkdir(parents=True, exist_ok=True)

val = pd.read_csv(ROOT / "data" / "raw" / "bd_shs" / "val.csv")
val["text"] = val["sentence"].astype(str).map(lambda s: prepare_original(s)[0])
val["label"] = val["hate speech"].astype(int)
val = val[val["text"].str.len() >= 10].reset_index(drop=True)
rng = random.Random(SEED)


def sheet_image(original: str, perturbed: str, path: Path) -> None:
    """Original (top row) and perturbed (bottom row) in each available font, labelled."""
    blocks = []
    for f in available_fonts():
        a = render(original, str(FONTS[f]))
        b = render(perturbed, str(FONTS[f]))
        w = max(a.shape[1], b.shape[1])
        pa = np.zeros((a.shape[0], w), np.uint8); pa[:, : a.shape[1]] = a
        pb = np.zeros((b.shape[0], w), np.uint8); pb[:, : b.shape[1]] = b
        blocks.append((f, 255 - pa, 255 - pb))
    width = max(max(x[1].shape[1], x[2].shape[1]) for x in blocks) + 120
    height = sum(x[1].shape[0] + x[2].shape[0] + 24 for x in blocks)
    canvas = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(canvas)
    y = 0
    for f, a, b in blocks:
        draw.text((4, y + 4), f"{f}  A", fill=0)
        canvas.paste(Image.fromarray(a), (110, y)); y += a.shape[0]
        draw.text((4, y + 4), f"{f}  B", fill=0)
        canvas.paste(Image.fromarray(b), (110, y)); y += b.shape[0] + 24
    canvas.save(path)


rows = []
for fam in FAMS:
    cands = [i for i, t in enumerate(val["text"]) if perturb(t, fam, 3, SEED)[2] > 0]
    rng.shuffle(cands)
    for k, i in enumerate(cands[:N_PER_FAMILY]):
        t = val.at[i, "text"]
        p, edits, n = perturb(t, fam, 3, SEED)
        item = f"{fam}_{k:02d}"
        # Randomise which of A/B is the original so reviewers cannot tell by position.
        swap = rng.random() < 0.5
        a, b = (p, t) if swap else (t, p)
        sheet_image(a, b, IMG / f"{item}.png")
        rows.append({"item_id": item, "family": fam, "val_index": int(i), "label": int(val.at[i, "label"]),
                     "n_edits": len(edits), "original_is": "B" if swap else "A",
                     "text_A": a, "text_B": b, "image": f"images/{item}.png"})

key = pd.DataFrame(rows)
key.to_csv(OUT / "review_key.csv", index=False, encoding="utf-8-sig")
sheet = key[["item_id", "family", "image"]].copy()
for col in ("same_meaning_YN", "same_label_YN", "readable_YN", "note"):
    sheet[col] = ""
sheet.to_csv(OUT / "review_REVIEWER1.csv", index=False, encoding="utf-8-sig")
sheet.to_csv(OUT / "review_REVIEWER2.csv", index=False, encoding="utf-8-sig")
print(f"{len(key)} items written to {OUT}")
print(key.groupby("family").size().to_dict())

"""Figure 1 for the paper: original vs perturbed renderings per family and font,
with code-point annotations and the pixel-difference map.

Usage: python src/render/make_figure.py  -> figures/render_examples.pdf (+ .png)
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3 (IEEE PDF eXpress requirement)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("render", "perturb", "normalize"):
    sys.path.insert(0, str(ROOT / "src" / sub))
from render import FONTS, available_fonts, render, DIFF_THRESHOLD, visual_check  # noqa: E402
from families import perturb  # noqa: E402

EXAMPLES = [  # family, short neutral Bangla phrase containing an eligible site
    ("F1", unicodedata.normalize("NFC", "সবাই ভালো থাকবেন")),           # ো -> ে + া
    ("F2", unicodedata.normalize("NFC", "আজকের আবহাওয়া সুন্দর")),      # zero-width insertion
    ("F5", unicodedata.normalize("NFC", "উৎসব জমে উঠেছে")),              # khanda ta -> legacy sequence
    ("F3", unicodedata.normalize("NFC", "বাংলার রূপ")),                  # Assamese ra/va (rejected family)
]
FONT_LABEL = {"noto": "Noto Sans Bengali", "solaiman": "SolaimanLipi", "kalpurush": "Kalpurush"}


def codepoints(s: str) -> str:
    return " ".join(f"{ord(c):04X}" for c in s)


def main() -> None:
    fonts = available_fonts()
    fig, axes = plt.subplots(len(EXAMPLES), len(fonts), figsize=(3.5 * len(fonts), 1.55 * len(EXAMPLES)), squeeze=False)
    for r, (fam, text) in enumerate(EXAMPLES):
        pert, edits, n = perturb(text, fam, 1, seed=3)
        for c, f in enumerate(fonts):
            a = render(text, str(FONTS[f])); b = render(pert, str(FONTS[f]))
            h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
            pa = np.zeros((h, w), np.uint8); pa[: a.shape[0], : a.shape[1]] = a
            pb = np.zeros((h, w), np.uint8); pb[: b.shape[0], : b.shape[1]] = b
            diff = np.abs(pa.astype(int) - pb.astype(int)) > DIFF_THRESHOLD
            rgb = np.stack([255 - pa] * 3, -1).astype(float)
            rgb[diff] = [220, 30, 30]
            canvas = np.vstack([np.stack([255 - pa] * 3, -1), np.stack([255 - pb] * 3, -1), rgb.astype(np.uint8)])
            ax = axes[r, c]
            ax.imshow(canvas, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            vc = visual_check(text, pert, f)
            if r == 0:
                ax.set_title(FONT_LABEL[f], fontsize=9)
            if c == 0:
                e = edits[0]
                ax.set_ylabel(f"{fam}\n{codepoints(e.old) or 'insert'} → {codepoints(e.new) or 'ZW'}", fontsize=7)
            ax.text(0.99, 0.02, f"{vc.verdict}, Δpx={vc.diff_pixels}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.5))
    fig.text(0.005, 0.5, "rows per cell: original · perturbed · difference (red)", rotation=90, va="center", fontsize=7)
    plt.tight_layout(rect=(0.02, 0, 1, 1))
    out = ROOT / "figures" / "render_examples"
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

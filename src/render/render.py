"""Shaping-aware rendering and pixel comparison (SSOT section 5).

Text is shaped with HarfBuzz (uharfbuzz) and rasterised glyph by glyph with
FreeType (freetype-py), so Bengali conjuncts, vowel signs and default-ignorable
characters (ZWJ, ZWNJ, ZWSP, WJ, SHY) are handled the way a browser or OS text
stack handles them. Rendering conditions are fixed: 32 px, grayscale
anti-aliasing, left-aligned on a common baseline, no hinting.

visual_check(original, perturbed, font_path) returns the differing-pixel
fraction over the union of inked pixels, the largest connected differing region
in pixels, and a verdict: "identical" | "near" | "perceptible".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import freetype
import numpy as np
import uharfbuzz as hb
from scipy import ndimage

PX = 32
PAD = 8
INK_THRESHOLD = 16          # 0-255 grayscale value counted as ink
DIFF_THRESHOLD = 16         # |a-b| above this counts as a differing pixel
NEAR_FRACTION = 0.005       # SSOT: < 0.5 % of glyph-area pixels
NEAR_REGION_PX = 9          # SSOT: no connected region above 9 px^2

ROOT = Path(__file__).resolve().parents[2]
FONTS = {
    "noto": ROOT / "fonts" / "NotoSansBengali-Regular.ttf",
    "solaiman": ROOT / "fonts" / "SolaimanLipi.ttf",
    "kalpurush": ROOT / "fonts" / "Kalpurush.ttf",
}


@lru_cache(maxsize=8)
def _load(font_path: str):
    blob = hb.Blob.from_file_path(font_path)
    face = hb.Face(blob)
    font = hb.Font(face)
    font.scale = (PX * 64, PX * 64)          # 26.6 fixed point, matches FreeType
    ft = freetype.Face(font_path)
    ft.set_char_size(PX * 64)
    return font, ft


def shape(text: str, font_path: str):
    font, _ = _load(font_path)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True, "liga": True})
    return list(zip(buf.glyph_infos, buf.glyph_positions))


def render(text: str, font_path: str) -> np.ndarray:
    """Grayscale image (0 = paper, 255 = full ink), fixed height, width to fit."""
    _, ft = _load(font_path)
    glyphs = shape(text, font_path)
    asc = ft.size.ascender >> 6
    desc = -(ft.size.descender >> 6)
    height = asc + desc + 2 * PAD
    pen_x = 0.0
    draws = []
    for info, pos in glyphs:
        ft.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING)
        g = ft.glyph
        bmp = g.bitmap
        if bmp.width and bmp.rows:
            arr = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.pitch)[:, : bmp.width]
            x = int(round(pen_x + pos.x_offset / 64.0)) + g.bitmap_left
            y = PAD + asc - g.bitmap_top - int(round(pos.y_offset / 64.0))
            draws.append((x, y, arr))
        pen_x += pos.x_advance / 64.0
    width = int(np.ceil(pen_x)) + 2 * PAD
    for x, y, arr in draws:
        width = max(width, PAD + x + arr.shape[1] + PAD)
    img = np.zeros((height, max(width, 1)), dtype=np.uint8)
    for x, y, arr in draws:
        x0, y0 = PAD + x, y
        h, w = arr.shape
        ys, xs = slice(max(y0, 0), min(y0 + h, height)), slice(max(x0, 0), min(x0 + w, img.shape[1]))
        sub = arr[max(0, -y0): max(0, -y0) + (ys.stop - ys.start), max(0, -x0): max(0, -x0) + (xs.stop - xs.start)]
        if sub.size:
            img[ys, xs] = np.maximum(img[ys, xs], sub)
    return img


@dataclass
class VisualResult:
    font: str
    width_a: int
    width_b: int
    ink_union: int
    diff_pixels: int
    diff_fraction: float
    max_region_px: int
    verdict: str


def _pad_to(a: np.ndarray, b: np.ndarray):
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    pa = np.zeros((h, w), np.uint8); pa[: a.shape[0], : a.shape[1]] = a
    pb = np.zeros((h, w), np.uint8); pb[: b.shape[0], : b.shape[1]] = b
    return pa, pb


def visual_check(original: str, perturbed: str, font_key: str) -> VisualResult:
    font_path = str(FONTS[font_key])
    a, b = _pad_to(render(original, font_path), render(perturbed, font_path))
    ink = ((a > INK_THRESHOLD) | (b > INK_THRESHOLD)).sum()
    diff = np.abs(a.astype(int) - b.astype(int)) > DIFF_THRESHOLD
    n_diff = int(diff.sum())
    if n_diff:
        labels, n = ndimage.label(diff)
        sizes = ndimage.sum(diff, labels, index=range(1, n + 1))
        max_region = int(max(sizes))
    else:
        max_region = 0
    frac = n_diff / ink if ink else 0.0
    if n_diff == 0:
        verdict = "identical"
    elif frac < NEAR_FRACTION and max_region <= NEAR_REGION_PX:
        verdict = "near"
    else:
        verdict = "perceptible"
    return VisualResult(font_key, a.shape[1], b.shape[1], int(ink), n_diff, round(frac, 5), max_region, verdict)


def available_fonts() -> list[str]:
    return [k for k, p in FONTS.items() if p.exists()]


def save_pair_png(original: str, perturbed: str, font_key: str, out_path: Path) -> None:
    """Side-by-side PNG (original above perturbed, diff below) for review sheets and figures."""
    from PIL import Image
    font_path = str(FONTS[font_key])
    a, b = _pad_to(render(original, font_path), render(perturbed, font_path))
    diff = (np.abs(a.astype(int) - b.astype(int)) > DIFF_THRESHOLD).astype(np.uint8) * 255
    canvas = np.vstack([255 - a, 255 - b, 255 - diff])
    Image.fromarray(canvas).save(out_path)


if __name__ == "__main__":
    import json, sys
    s = sys.argv[1] if len(sys.argv) > 1 else "তোমায় ভালোবাসি"
    for k in available_fonts():
        r = visual_check(s, s + "‌", k)
        print(json.dumps(asdict(r), ensure_ascii=False))

"""Perturbation families F1-F4 (SSOT section 4). Frozen at Gate G2.

Each family exposes:
    eligible(text) -> list of edit sites
    apply(text, sites) -> perturbed text
and `perturb(text, family, budget, seed)` picks `budget` sites at random with a
fixed seed. Budgets: an int, or "len" = ceil(len(eligible)/10), min 1.

Inputs are assumed to be *prepared originals* (NFC + zero-width characters
stripped, see bn_norm.prepare_original), so F1 edits always go from the NFC form
to its canonical equivalent.
"""
from __future__ import annotations

import math
import random
import unicodedata
from dataclasses import dataclass

VIRAMA = "্"
ZWNJ, ZWJ, ZWSP, WJ, SHY, NBSP = "‌", "‍", "​", "⁠", "­", " "
NUKTA = "়"
DANDA = "।"


def is_bengali(ch: str) -> bool:
    return 0x0980 <= ord(ch) <= 0x09FF


def is_bengali_letter(ch: str) -> bool:
    return is_bengali(ch) and unicodedata.category(ch) == "Lo"


def is_mark(ch: str) -> bool:
    return unicodedata.category(ch) in ("Mn", "Mc")


@dataclass(frozen=True)
class Edit:
    pos: int          # index in the original string
    kind: str         # sub-transformation code
    old: str
    new: str


# ---------------------------------------------------------------- F1 -------
# Bangla-canonical (precomposed) form -> canonically equivalent alternative.
# For the nukta letters the alternative IS the NFC form, so NFC "neutralises"
# F1 only in the sense that it maps both spellings to the decomposed one.
F1_MAP = {
    "য়": "য়",   # য় -> য + ়
    "ড়": "ড়",   # ড় -> ড + ়
    "ঢ়": "ঢ়",   # ঢ় -> ঢ + ়
    "ো": "ো",   # ো -> ে + া
    "ৌ": "ৌ",   # ৌ -> ে + ৗ
}


def f1_eligible(text: str) -> list[Edit]:
    return [Edit(i, "F1:" + ch.encode("unicode_escape").decode(), ch, F1_MAP[ch])
            for i, ch in enumerate(text) if ch in F1_MAP]


# ---------------------------------------------------------------- F2 -------
F2_CHARS = {"zwnj": ZWNJ, "zwj": ZWJ, "zwsp": ZWSP, "wj": WJ, "shy": SHY}


def f2_eligible(text: str) -> list[Edit]:
    """Insertion points after a complete Bengali cluster: previous char is a
    Bengali letter or mark, not virama; next char is not a mark and not virama
    (so the inserted joiner never changes conjunct or vowel-sign shaping).

    Known limitation found by the exhaustive post-hoc check (19 Oct 2026, kept
    as frozen for the reported results): Noto Sans Bengali shapes a format
    character inserted directly after chandrabindu U+0981 visibly (23 of
    14,691 evaluated variants). A future revision should also exclude
    positions where the previous character is U+0981."""
    sites = []
    for i in range(1, len(text)):
        prev, nxt = text[i - 1], text[i]
        if not is_bengali(prev) or prev == VIRAMA or prev == NUKTA:
            continue
        if is_mark(nxt) or nxt == VIRAMA:
            continue
        if not (is_bengali_letter(prev) or is_mark(prev)):
            continue
        sites.append(Edit(i, "F2:insert", "", ""))  # char chosen at apply time
    return sites


def f2_nbsp_eligible(text: str) -> list[Edit]:
    """Control variant: ordinary space -> NBSP between Bengali words (NFKC undoes it)."""
    return [Edit(i, "F2nbsp", " ", NBSP) for i, ch in enumerate(text)
            if ch == " " and 0 < i < len(text) - 1 and is_bengali(text[i - 1]) and is_bengali(text[i + 1])]


# ---------------------------------------------------------------- F3 -------
F3_MAP = {"র": "ৰ", "ব": "ৱ"}   # র -> ৰ, ব -> ৱ


def f3_eligible(text: str) -> list[Edit]:
    return [Edit(i, "F3:" + ch, ch, F3_MAP[ch]) for i, ch in enumerate(text) if ch in F3_MAP]


# ---------------------------------------------------------------- F4 -------
F4_MAP = {
    DANDA: "৷",   # danda -> currency numerator four (visually a danda)
    "|": "৷",
    "!": "ǃ",     # latin retroflex click
    ".": "․",     # one dot leader
    ",": "‚",     # single low-9 quotation mark
    ":": "꞉",     # modifier letter colon
    "-": "‐",     # hyphen
    "?": "？",     # fullwidth question mark (may fail the visual check)
}


def f4_eligible(text: str) -> list[Edit]:
    return [Edit(i, "F4:" + ch, ch, F4_MAP[ch]) for i, ch in enumerate(text) if ch in F4_MAP]


# ---------------------------------------------------------------- F5 -------
# Legacy (pre-Unicode-4.1) spelling of khanda ta: TA + VIRAMA + ZWJ renders as
# the same glyph as U+09CE in shaping-aware fonts, is not canonically
# equivalent, and survives NFC/NFKC. BN-norm rule 7 maps it back.
KHANDA_TA = "ৎ"
KHANDA_TA_LEGACY = "ত" + VIRAMA + ZWJ


def f5_eligible(text: str) -> list[Edit]:
    return [Edit(i, "F5:khanda_ta", KHANDA_TA, KHANDA_TA_LEGACY) for i, ch in enumerate(text) if ch == KHANDA_TA]


# Per-character sub-families of F2 (added 20 Oct 2026 as an EXTENSION for the
# second experiment round; F2 itself is unchanged): same eligibility as F2, but
# every edit inserts one fixed character, so the effect of each invisible
# character can be reported separately.
F2_SUBFAMILIES = {f"F2_{code}": ch for code, ch in F2_CHARS.items()}

FAMILIES = {
    "F1": f1_eligible,
    "F2": f2_eligible,
    "F2nbsp": f2_nbsp_eligible,
    "F3": f3_eligible,
    "F4": f4_eligible,
    "F5": f5_eligible,
    **{name: f2_eligible for name in F2_SUBFAMILIES},
}


# ---------------------------------------------------------------- apply ----
def resolve_budget(budget, n_eligible: int) -> int:
    if budget == "len":
        return max(1, math.ceil(n_eligible / 10))
    return int(budget)


def apply(text: str, edits: list[Edit], family: str, rng: random.Random) -> str:
    """Apply edits right-to-left so earlier indices stay valid."""
    out = text
    for e in sorted(edits, key=lambda e: e.pos, reverse=True):
        if family == "F2":
            ch = F2_CHARS[rng.choice(list(F2_CHARS))]
            out = out[:e.pos] + ch + out[e.pos:]
        elif family in F2_SUBFAMILIES:
            out = out[:e.pos] + F2_SUBFAMILIES[family] + out[e.pos:]
        elif e.old == "":
            raise ValueError("insertion edit outside F2")
        else:
            out = out[:e.pos] + e.new + out[e.pos + len(e.old):]
    return out


def perturb(text: str, family: str, budget, seed: int) -> tuple[str, list[Edit], int]:
    """Return (perturbed_text, edits_applied, n_eligible). If fewer sites than the
    budget exist, all sites are used; the caller records the shortfall."""
    sites = FAMILIES[family](text)
    n = len(sites)
    if n == 0:
        return text, [], 0
    k = min(resolve_budget(budget, n), n)
    rng = random.Random(f"{seed}|{family}|{budget}|{text}")
    chosen = rng.sample(sites, k)
    return apply(text, chosen, family, rng), chosen, n

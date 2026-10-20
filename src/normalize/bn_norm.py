"""Unicode normalization defenses D0-D3 (SSOT section 10).

D0  none    : text as received
D1  nfc     : unicodedata NFC
D2  nfkc    : unicodedata NFKC (the csebuetnlp normalizer default)
D3  bn      : NFC followed by the six conservative BN-norm rules below

Every rule is documented with its risk. Rule 6 is a prohibition: nothing beyond
rules 1-5 is ever changed (no letters, vowel signs, nukta, anusvara, visarga,
chandrabindu or digits).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

# --- Bengali block helpers ---------------------------------------------------
BENGALI_RANGE = (0x0980, 0x09FF)
VIRAMA = "্"
ZWNJ, ZWJ = "‌", "‍"
ZWSP, WJ, SHY, BOM = "​", "⁠", "­", "﻿"
NBSP = " "
DANDA = "।"          # Bengali uses the Devanagari danda code point
DOUBLE_DANDA = "॥"
ASSAMESE_RA, ASSAMESE_VA = "ৰ", "ৱ"
BENGALI_RA, BENGALI_BA = "র", "ব"

# Rule 1: format characters that never carry meaning in Bangla prose.
ALWAYS_REMOVE = {ZWSP, WJ, SHY, BOM}

# Rule 5: punctuation look-alikes mapped back to the ordinary form.
DANDA_LOOKALIKES = {"|": DANDA, "｜": DANDA, "∣": DANDA, "৷": DANDA, "ǀ": DANDA}
ASCII_LOOKALIKES = {
    "ǃ": "!",   # latin letter retroflex click
    "․": ".",   # one dot leader
    "‚": ",",   # single low-9 quotation mark
    "꞉": ":",   # modifier letter colon
    "ː": ":",   # modifier letter triangular colon
    "‐": "-",   # hyphen
    "‑": "-",   # non-breaking hyphen
    "？": "?", "！": "!", "，": ",", "．": ".", "：": ":", "；": ";",
}


def is_bengali(ch: str) -> bool:
    return bool(ch) and BENGALI_RANGE[0] <= ord(ch) <= BENGALI_RANGE[1]


@dataclass
class NormReport:
    """What a normalization call changed; used for clean-text cost analysis."""
    removed_format_chars: int = 0
    removed_joiners: int = 0
    kept_joiners_after_virama: int = 0
    spaces_mapped: int = 0
    assamese_mapped: int = 0
    assamese_guard_blocked: bool = False
    punctuation_mapped: int = 0
    changed: bool = False
    notes: list[str] = field(default_factory=list)


def _rule_1_3(text: str, rep: NormReport) -> str:
    out = []
    prev = ""
    for ch in text:
        if ch in ALWAYS_REMOVE:                      # rule 1
            rep.removed_format_chars += 1
            continue
        if ch in (ZWNJ, ZWJ):                        # rule 2
            if prev == VIRAMA:
                rep.kept_joiners_after_virama += 1
                out.append(ch)
                prev = ch
            else:
                rep.removed_joiners += 1
            continue
        if ch != " " and unicodedata.category(ch) == "Zs":   # rule 3
            rep.spaces_mapped += 1
            out.append(" ")
            prev = " "
            continue
        out.append(ch)
        prev = ch
    return "".join(out)


def assamese_guard_allows(text: str) -> bool:
    """Rule 4 guard. Map Assamese ra/va to Bengali ra/ba only when the Bengali
    forms still dominate, i.e. the text is Bangla with a few substituted letters,
    not genuine Assamese (where ra is written exclusively as U+09F0)."""
    n_assamese = text.count(ASSAMESE_RA) + text.count(ASSAMESE_VA)
    if n_assamese == 0:
        return False
    # Genuine Assamese writes ra exclusively as U+09F0 and never uses Bengali
    # র U+09B0 (Assamese does use ব, so ba carries no signal). Block only when
    # no Bengali র is present AND at least two Assamese forms occur. Known
    # limit (reported in the paper): a sentence whose every র is substituted
    # and that has 2+ Assamese forms is not repaired by rule 4.
    return BENGALI_RA in text or n_assamese < 2


def _rule_4(text: str, rep: NormReport) -> str:
    if ASSAMESE_RA not in text and ASSAMESE_VA not in text:
        return text
    if not assamese_guard_allows(text):
        rep.assamese_guard_blocked = True
        rep.notes.append("assamese letters dominate; rule 4 not applied")
        return text
    n = text.count(ASSAMESE_RA) + text.count(ASSAMESE_VA)
    rep.assamese_mapped += n
    return text.replace(ASSAMESE_RA, BENGALI_RA).replace(ASSAMESE_VA, BENGALI_BA)


def _rule_5(text: str, rep: NormReport) -> str:
    out = []
    for i, ch in enumerate(text):
        if ch in DANDA_LOOKALIKES:
            # Only when it sits next to Bengali text; a pipe in a URL or code stays.
            left = text[i - 1] if i > 0 else ""
            right = text[i + 1] if i + 1 < len(text) else ""
            if is_bengali(left) or is_bengali(right) or (left == " " and is_bengali(text[i - 2] if i > 1 else "")):
                out.append(DANDA_LOOKALIKES[ch])
                rep.punctuation_mapped += 1
                continue
        if ch in ASCII_LOOKALIKES:
            out.append(ASCII_LOOKALIKES[ch])
            rep.punctuation_mapped += 1
            continue
        out.append(ch)
    return "".join(out)


KHANDA_TA = "ৎ"
KHANDA_TA_LEGACY = "ত" + VIRAMA + ZWJ

# Rule 0: Bangla canonical composition. NFC *decomposes* the three
# composition-excluded nukta letters (U+09DC, U+09DD, U+09DF), but Bangla text in
# the wild, and the corpora Bangla models are trained on, overwhelmingly use the
# precomposed letters (BD-SHS: 81% precomposed). We therefore canonicalise to
# NFC and then recompose these three letters, so both spellings of a word map to
# the form the models were trained on. Two-part vowel signs are composed by NFC
# already (99% precomposed in BD-SHS).
RECOMPOSE = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}


def bangla_canonical(text: str) -> str:
    out = unicodedata.normalize("NFC", text)
    for dec, pre in RECOMPOSE.items():
        out = out.replace(dec, pre)
    return out


def _rule_7(text: str, rep: NormReport) -> str:
    """Rule 7: legacy khanda-ta sequence TA+VIRAMA+ZWJ -> U+09CE, as the Unicode
    Standard recommends since 4.1. Risk: none; the sequence has no other reading."""
    n = text.count(KHANDA_TA_LEGACY)
    if n:
        rep.punctuation_mapped += 0  # keep counters simple; tracked in notes
        rep.notes.append(f"khanda_ta_legacy_mapped={n}")
        text = text.replace(KHANDA_TA_LEGACY, KHANDA_TA)
    return text


def normalize(text: str, mode: str = "bn", report: NormReport | None = None) -> str:
    """Apply defense `mode` in {"none", "nfc", "nfkc", "bn"}."""
    rep = report if report is not None else NormReport()
    if mode == "none":
        return text
    if mode == "nfc":
        out = unicodedata.normalize("NFC", text)
    elif mode == "nfkc":
        out = unicodedata.normalize("NFKC", text)
    elif mode == "bn":
        out = bangla_canonical(text)                 # rule 0
        out = _rule_1_3(out, rep)
        out = _rule_7(out, rep)
        out = _rule_4(out, rep)
        out = _rule_5(out, rep)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    rep.changed = out != text
    return out


def prepare_original(text: str) -> tuple[str, NormReport]:
    """Cleaning applied to ORIGINAL sentences before any perturbation (SSOT 7.1):
    Bangla canonical composition (rule 0) plus rules 1-3 only. Pre-existing
    confusables and punctuation variants are counted by the audit but left as
    they are."""
    rep = NormReport()
    out = bangla_canonical(text)
    out = _rule_1_3(out, rep)
    rep.changed = out != text
    return out, rep


DEFENSES = ("none", "nfc", "nfkc", "bn")

"""Unit tests for perturbation families. Run: python -m pytest src/perturb -q"""
import sys, pathlib, unicodedata
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "normalize"))

from families import (F2_CHARS, FAMILIES, VIRAMA, f2_eligible, perturb, resolve_budget)
from bn_norm import normalize

from bn_norm import bangla_canonical

S = bangla_canonical("তোমায় ভালোবাসি, বাংলা। কেন রাগ করো?")


def test_f1_is_canonically_equivalent_and_bn_norm_restores():
    p, edits, n = perturb(S, "F1", 3, seed=1)
    assert n >= 2 and p != S
    assert any(e.old == "য়" for e in edits) or any(e.old == "ো" for e in edits)
    # canonically equivalent: NFC of both sides agrees
    assert unicodedata.normalize("NFC", p) == unicodedata.normalize("NFC", S)
    # BN-norm maps the attack back to the precomposed canonical form
    assert normalize(p, "bn") == S
    # NFC maps the clean text AWAY from the precomposed form (the cost we measure)
    assert normalize(S, "nfc") != S


def test_f2_never_inserts_next_to_virama_or_before_marks():
    sites = f2_eligible("ক" + VIRAMA + "ত" + "া")
    for e in sites:
        assert e.pos not in (1, 2)   # not after ক before virama, not after virama
    p, edits, n = perturb(S, "F2", 3, seed=1)
    assert len(edits) == 3
    stripped = "".join(ch for ch in p if ch not in F2_CHARS.values())
    assert stripped == S
    assert normalize(p, "bn") == S           # BN-norm undoes F2
    assert normalize(p, "nfc") != S          # NFC does not


def test_f2nbsp_is_undone_by_nfkc():
    p, edits, n = perturb(S, "F2nbsp", 1, seed=1)
    assert " " in p and normalize(p, "nfkc") == normalize(S, "nfkc")
    assert normalize(p, "bn") == S


def test_f3_substitutes_only_ra_ba_and_bn_norm_restores():
    # budget 1 leaves a Bengali ra in the text, so rule 4's guard allows repair
    p, edits, n = perturb(S, "F3", 1, seed=1)
    assert all(e.old in ("র", "ব") for e in edits)
    assert normalize(p, "bn") == S
    assert normalize(p, "nfkc") != normalize(S, "nfkc")


def test_f4_touches_only_punctuation():
    p, edits, n = perturb(S, "F4", 3, seed=1)
    assert all(not ("ঀ" <= e.old <= "৿") or e.old == "।" for e in edits)
    assert normalize(p, "bn") == S


def test_f5_khanda_ta_round_trip():
    s = "উৎসব আর জগৎ"
    p, edits, n = perturb(s, "F5", 3, seed=1)
    assert n == 2 and "ৎ" not in p and p != s
    assert normalize(p, "bn") == s and normalize(p, "nfkc") == p


def test_f2_subfamilies_insert_one_fixed_character():
    from families import F2_SUBFAMILIES
    for name, ch in F2_SUBFAMILIES.items():
        p, edits, n = perturb(S, name, 3, seed=1)
        assert len(edits) == 3 and p.count(ch) == 3
        others = [c for c in F2_CHARS.values() if c != ch]
        assert not any(o in p for o in others)
        assert normalize(p, "bn") == S


def test_budget_rules_and_determinism():
    assert resolve_budget("len", 25) == 3 and resolve_budget("len", 1) == 1
    a = perturb(S, "F3", 1, seed=7)[0]
    b = perturb(S, "F3", 1, seed=7)[0]
    assert a == b
    p, edits, n = perturb("abc", "F3", 3, seed=1)
    assert p == "abc" and edits == [] and n == 0

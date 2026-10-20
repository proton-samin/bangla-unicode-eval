"""Unit tests for the BN-norm defense. Run: python -m pytest src/normalize -q"""
import unicodedata

from bn_norm import (ASSAMESE_RA, DANDA, NBSP, SHY, VIRAMA, WJ, ZWJ, ZWNJ, ZWSP,
                     NormReport, assamese_guard_allows, normalize, prepare_original)

# য় precomposed (U+09DF) and decomposed (য U+09AF + ় U+09BC)
YYA_PRE = "য়"
YYA_DEC = "য়"
# ো precomposed (U+09CB) and its two parts (ে U+09C7 + া U+09BE)
O_PRE = "ো"
O_DEC = "ো"


def test_nfc_neutralises_family_f1():
    assert normalize("ক" + YYA_PRE, "nfc") == normalize("ক" + YYA_DEC, "nfc")
    assert normalize("ক" + O_PRE, "nfc") == normalize("ক" + O_DEC, "nfc")
    # composition exclusion: NFC yields the DEcomposed nukta form for both inputs
    assert unicodedata.normalize("NFC", YYA_PRE) == YYA_DEC


def test_rule0_bangla_canonical_recomposes_nukta_letters():
    from bn_norm import bangla_canonical
    assert bangla_canonical("ক" + YYA_DEC) == "ক" + YYA_PRE
    assert bangla_canonical("ক" + YYA_PRE) == "ক" + YYA_PRE
    assert bangla_canonical("ক" + O_DEC) == "ক" + O_PRE
    assert normalize("ক" + YYA_DEC, "bn") == "ক" + YYA_PRE


def test_nfc_nfkc_do_not_remove_joiners():
    s = "বা" + ZWNJ + "ংলা"
    assert ZWNJ in normalize(s, "nfc")
    assert ZWNJ in normalize(s, "nfkc")


def test_nfkc_maps_nbsp_but_nfc_does_not():
    s = "ভাল" + NBSP + "না"
    assert NBSP in normalize(s, "nfc")
    assert NBSP not in normalize(s, "nfkc")


def test_rule1_removes_format_chars():
    rep = NormReport()
    out = normalize("বা" + ZWSP + "ং" + WJ + "লা" + SHY, "bn", rep)
    assert out == "বাংলা"
    assert rep.removed_format_chars == 3


def test_rule2_removes_joiners_except_after_virama():
    rep = NormReport()
    # র‍্য : RA + ZWJ + VIRAMA + YA is a legitimate ya-phala form; here the ZWJ is
    # before the virama so it is removed; the one after virama below is kept.
    s = "ক" + ZWNJ + "ক" + VIRAMA + ZWJ + "য"      # (TA+VIRAMA+ZWJ is rule 7's khanda ta)
    out = normalize(s, "bn", rep)
    assert out == "কক" + VIRAMA + ZWJ + "য"
    assert rep.removed_joiners == 1 and rep.kept_joiners_after_virama == 1


def test_rule3_maps_unicode_spaces():
    assert normalize("ভাল" + NBSP + "না", "bn") == "ভাল না"
    assert normalize("ভাল না", "bn") == "ভাল না"   # thin space


def test_rule4_assamese_guard():
    bangla_with_one_sub = "আমার সোনার বাংলা আমি তোমায় ভালোবাসি".replace("র", ASSAMESE_RA, 1)
    assert assamese_guard_allows(bangla_with_one_sub)
    assert ASSAMESE_RA not in normalize(bangla_with_one_sub, "bn")
    # a single substituted letter with no Bengali ra/ba left is still repaired
    assert normalize("কৰ", "bn") == "কর"
    # all ra/ba substituted and at least two of them: guard blocks (known limit)
    assert not assamese_guard_allows("কৰ কৰ")
    # genuine Assamese: ra always U+09F0, Bengali ব may legitimately appear
    genuine_assamese = "মই অসমীয়া ভাষা ভাল পাওঁ আৰু ৰাতি কিতাপ পঢ়োঁ"
    assert genuine_assamese.count(ASSAMESE_RA) >= 2 and "র" not in genuine_assamese
    assert not assamese_guard_allows(genuine_assamese)
    rep = NormReport()
    from bn_norm import bangla_canonical
    assert normalize(genuine_assamese, "bn", rep) == bangla_canonical(genuine_assamese)
    assert rep.assamese_guard_blocked


def test_rule5_punctuation_next_to_bengali_only():
    assert normalize("ভাল না|", "bn") == "ভাল না" + DANDA
    assert normalize("ভাল না ৷", "bn") == "ভাল না " + DANDA
    assert normalize("a|b", "bn") == "a|b"            # pipe in Latin context untouched
    assert normalize("কিǃ", "bn") == "কি!"


def test_rule6_letters_never_change():
    from bn_norm import bangla_canonical
    s = "ড়় ঃ ং ঁ ১২৩ য়"   # letters, nukta, visarga, anusvara, chandrabindu, digits
    assert normalize(s, "bn") == bangla_canonical(s)
    assert "ঃ ং ঁ ১২৩" in normalize(s, "bn")


def test_rule7_legacy_khanda_ta():
    legacy = "উত" + VIRAMA + ZWJ + "সব"      # উৎসব written the pre-4.1 way
    assert normalize(legacy, "bn") == "উৎসব"
    assert normalize(legacy, "nfc") == legacy       # NFC/NFKC leave it alone
    assert normalize(legacy, "nfkc") == legacy


def test_prepare_original_strips_zw_but_keeps_punctuation_variants():
    out, rep = prepare_original("বাং" + ZWNJ + "লা|")
    assert out == "বাংলা|"
    assert rep.removed_joiners == 1

from __future__ import annotations

import numpy as np
import pytest
from transformers import PreTrainedTokenizerBase

from cilab.stats import bootstrap_ci, paired_pct_change, paired_share, resample_indices, window_totals
from cilab.text import decode, encode
from cilab.tokclass import (
    FREQUENT,
    KNOWLEDGE,
    MIDDLE,
    NO_WORD,
    RARE,
    REPLACEMENT,
    STYLE,
    WS_TAG,
    Word,
    check_offsets,
    class_sums,
    frequency_classes,
    offset_mismatches,
    owned_spans,
    pos_classes,
    token_pos_tags,
    token_word_index,
    window_class_sums,
    word_class,
)

MULTIBYTE = "Naïve café — 東京 🙂 مرحبا 𝔘nicode.\n\n  Tabs\tand  spaces, 1,234.56 (approx). " * 6

SENTENCE = "The 2026 laureate, Dr. Smith, explained it.\n\nWow!"
# hand-tagged words (Universal POS), as a tagger would return them
WORDS = (Word(0, 3, "DET", "the"), Word(4, 8, "NUM", "2026"), Word(9, 17, "NOUN", "laureate"),
         Word(17, 18, "PUNCT", ","), Word(19, 22, "PROPN", "dr."), Word(23, 28, "PROPN", "smith"),
         Word(28, 29, "PUNCT", ","), Word(30, 39, "VERB", "explained"), Word(40, 42, "PRON", "it"),
         Word(42, 43, "PUNCT", "."), Word(43, 45, "SPACE", "\n\n"), Word(45, 48, "INTJ", "wow"),
         Word(48, 49, "PUNCT", "!"))
# GPT-2-style token spans of SENTENCE, leading spaces included, written out by hand
TOKENS = ((0, 3), (3, 6), (6, 8), (8, 17), (17, 18), (18, 21), (21, 22), (22, 28), (28, 29), (29, 39), (39, 42),
          (42, 43), (43, 44), (44, 45), (45, 48), (48, 49))
COMMON = frozenset({"the", "it", "explained"})


def test_gpt2_offsets_round_trip(gpt2_tok: PreTrainedTokenizerBase) -> None:
    enc = gpt2_tok(MULTIBYTE, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    assert ids == encode(gpt2_tok, MULTIBYTE)
    check_offsets(offsets, len(MULTIBYTE))
    assert "".join(MULTIBYTE[s:e] for s, e in owned_spans(offsets)) == MULTIBYTE
    pieces = [decode(gpt2_tok, [i]) for i in ids]
    assert offset_mismatches(MULTIBYTE, offsets, pieces) == []
    assert sum(REPLACEMENT in p for p in pieces) > 0  # split multi-byte characters are exercised
    whole = [p for p in pieces if REPLACEMENT not in p]
    assert len(whole) > len(pieces) // 2


def test_check_offsets_rejects_gaps_reversals_and_short_cover() -> None:
    check_offsets([(0, 2), (1, 2), (2, 5)], 5)  # overlap from a split character is fine
    for bad in ([(0, 2), (3, 5)], [(0, 3), (2, 4), (1, 5)], [(0, 2), (2, 4)], [(1, 5)], [(0, 0), (0, 5)]):
        with pytest.raises(ValueError):
            check_offsets(bad, 5)


def test_class_assignment_on_a_hand_made_sentence() -> None:
    check_offsets(TOKENS, len(SENTENCE))
    widx = token_word_index(SENTENCE, TOKENS, WORDS)
    assert widx.tolist() == [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, NO_WORD, NO_WORD, 11, 12]
    k, m, s = KNOWLEDGE, MIDDLE, STYLE
    # The | ␣20 26 | ␣laureate | , | ␣Dr . | ␣Smith | , | ␣explained | ␣it | . | \n \n | Wow | !
    assert pos_classes(widx, WORDS, COMMON).tolist() == [s, k, k, k, s, k, k, k, s, m, s, s, s, s, s, s]
    tags = token_pos_tags(widx, WORDS).tolist()
    assert tags[1] == "NUM" and tags[6] == "PROPN" and tags[12] == WS_TAG and tags[13] == WS_TAG
    assert word_class("VERB", "explained", frozenset()) == KNOWLEDGE  # rare open-class word
    assert word_class("ADV", "quickly", frozenset()) == MIDDLE
    assert word_class("X", "etc", COMMON) == MIDDLE


def test_token_spanning_a_boundary_takes_its_first_non_space_character() -> None:
    text = "end,\n\nthe"
    words = (Word(0, 3, "NOUN", "end"), Word(3, 4, "PUNCT", ","), Word(4, 6, "SPACE", "\n\n"), Word(6, 9, "DET", "the"))
    widx = token_word_index(text, [(0, 3), (3, 6), (6, 9)], words)
    assert widx.tolist() == [0, 1, 3]
    with pytest.raises(ValueError):  # a non-space character outside every word
        token_word_index("ab", [(0, 2)], (Word(1, 2, "NOUN", "b"),))


def test_frequency_classes_split_at_the_occurrence_median() -> None:
    # counts per occurrence: 3 3 3 2 2 1, median 2.5
    assert frequency_classes([1, 1, 1, 2, 2, 3]).tolist() == [FREQUENT] * 3 + [RARE] * 3
    assert frequency_classes([5, 5]).tolist() == [FREQUENT, FREQUENT]


def test_per_class_sums_equal_the_total() -> None:
    rng = np.random.default_rng(0)
    bits = rng.gamma(2.0, 2.0, size=500)
    codes = rng.integers(0, 3, size=500).astype(np.int8)
    lengths = [128, 128, 128, 116]
    cs = class_sums(bits, codes, 3)
    assert np.isclose(cs.sum(), bits.sum())
    wcs = window_class_sums(bits, codes, lengths, 3)
    assert wcs.shape == (4, 3)
    assert np.allclose(wcs.sum(axis=0), cs)
    assert np.allclose(wcs.sum(axis=1), window_totals(bits, lengths))
    assert np.isclose(class_sums(bits, np.zeros(500, dtype=np.int8), 3)[0], bits.sum())
    with pytest.raises(ValueError):
        window_class_sums(bits, codes, [100], 3)


def test_class_shares_sum_to_100_and_drift_matches_the_total() -> None:
    rng = np.random.default_rng(1)
    base = rng.gamma(5.0, 50.0, size=(20, 3))
    post = base * rng.uniform(1.0, 1.2, size=(20, 3))
    idx = resample_indices(20, 500, seed=0)
    shares = np.array([paired_share(base[:, c], post[:, c], base.sum(1), post.sum(1))(idx) for c in range(3)])
    assert np.allclose(shares.sum(axis=0), 100.0)
    point, lo, hi = bootstrap_ci(paired_pct_change(base.sum(1), post.sum(1)), idx)
    assert lo < point < hi
    assert np.isclose(point, 100 * (post.sum() - base.sum()) / base.sum())

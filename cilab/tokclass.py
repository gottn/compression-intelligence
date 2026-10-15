"""Token classes for splitting per-token bits by word type (exp05). Pure: offsets, tagged words and word lists are
parameters; the tokenizer and POS-tagger calls live in the experiment script.

Alignment rule: a token takes the word that covers its first non-whitespace character, using the character offsets
the fast tokenizer reports (`return_offsets_mapping=True`) on exactly the text whose bits were scored. Byte-level BPE
offsets include the token's leading space, and every piece of a split multi-byte character reports that character's
span, so consecutive spans may overlap but never leave a gap.

Primary classes (exp05 RESULTS.md, pre-registered):
  knowledge  PROPN, NUM, and NOUN/ADJ/VERB whose lowercased word is not among the common words
  middle     everything else (common NOUN/ADJ/VERB, ADV, X)
  style      DET, ADP, PRON, CCONJ, SCONJ, AUX, PART, INTJ, PUNCT, SPACE, SYM, and whitespace-only tokens
Frequency variant: rare / frequent by the token id's count in the corpus against the median over occurrences.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

POS_CLASSES: tuple[str, ...] = ("knowledge", "middle", "style")
FREQ_CLASSES: tuple[str, ...] = ("rare", "frequent")
KNOWLEDGE, MIDDLE, STYLE = 0, 1, 2
RARE, FREQUENT = 0, 1
KNOWLEDGE_POS = frozenset({"PROPN", "NUM"})
OPEN_POS = frozenset({"NOUN", "ADJ", "VERB"})  # knowledge when the word is not common, middle otherwise
STYLE_POS = frozenset({"DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PART", "INTJ", "PUNCT", "SPACE", "SYM"})
NO_WORD = -1  # word index of a token made only of whitespace
WS_TAG = "WS"  # POS label reported for such a token
REPLACEMENT = "\N{REPLACEMENT CHARACTER}"

Span = tuple[int, int]  # [start, end) in characters


@dataclass(frozen=True)
class Word:
    """A POS-tagged word: characters [start, end) of the text, Universal POS tag, lowercased surface form."""

    start: int
    end: int
    pos: str
    lower: str


def word_class(pos: str, lower: str, common: frozenset[str]) -> int:
    """KNOWLEDGE, MIDDLE or STYLE code of one tagged word."""
    if pos in KNOWLEDGE_POS or (pos in OPEN_POS and lower not in common):
        return KNOWLEDGE
    return STYLE if pos in STYLE_POS else MIDDLE


def check_offsets(offsets: Sequence[Span], n_chars: int) -> None:
    """Raise unless the token spans tile [0, n_chars): first start 0, every span non-empty, starts non-decreasing,
    no gap between a token and the next, last end n_chars."""
    if not offsets:
        if n_chars:
            raise ValueError("no tokens for a non-empty text")
        return
    if offsets[0][0] != 0:
        raise ValueError(f"first token starts at {offsets[0][0]}, not 0")
    reach, prev = 0, 0
    for t, (s, e) in enumerate(offsets):
        if e <= s:
            raise ValueError(f"token {t} has an empty span ({s}, {e})")
        if s < prev:
            raise ValueError(f"token {t} starts at {s}, before token {t - 1} ({prev})")
        if s > reach:
            raise ValueError(f"gap before token {t}: characters {reach}..{s} belong to no token")
        reach, prev = max(reach, e), s
    if reach != n_chars or offsets[-1][1] != n_chars:
        raise ValueError(f"tokens end at {offsets[-1][1]} (reach {reach}), text has {n_chars} chars")


def owned_spans(offsets: Sequence[Span]) -> list[Span]:
    """Disjoint spans that tile the text: token i owns [s_i, s_{i+1}), the last token [s_n, e_n). A leading piece of
    a split character owns nothing; its last piece owns the character."""
    starts = [s for s, _ in offsets]
    return [*pairwise(starts), *([tuple(offsets[-1])] if offsets else [])]


def offset_mismatches(text: str, offsets: Sequence[Span], pieces: Sequence[str]) -> list[int]:
    """Indices of tokens whose decoded piece is a whole string (no U+FFFD) but differs from the text under its
    offsets. `pieces[i]` is the decode of token i alone."""
    return [t for t, ((s, e), p) in enumerate(zip(offsets, pieces, strict=True))
            if REPLACEMENT not in p and text[s:e] != p]


def first_nonspace(text: str, start: int, end: int) -> int:
    """Index of the first non-whitespace character in text[start:end], or -1 if there is none."""
    for i in range(start, end):
        if not text[i].isspace():
            return i
    return -1


def char_word_index(words: Sequence[Word], n_chars: int) -> np.ndarray:
    """Index into `words` of the word covering each character; NO_WORD where none does. Raises if words overlap or
    leave the text."""
    idx = np.full(n_chars, NO_WORD, dtype=np.int64)
    for k, w in enumerate(words):
        if not 0 <= w.start < w.end <= n_chars:
            raise ValueError(f"word {k} span ({w.start}, {w.end}) outside the text")
        if (idx[w.start:w.end] != NO_WORD).any():
            raise ValueError(f"word {k} overlaps an earlier word")
        idx[w.start:w.end] = k
    return idx


def token_word_index(text: str, offsets: Sequence[Span], words: Sequence[Word]) -> np.ndarray:
    """Per token, the index of the word covering its first non-whitespace character; NO_WORD for a token made only
    of whitespace. Raises if that character lies in no word."""
    cw = char_word_index(words, len(text))
    out = np.empty(len(offsets), dtype=np.int64)
    for t, (s, e) in enumerate(offsets):
        i = first_nonspace(text, s, e)
        if i >= 0 and cw[i] == NO_WORD:
            raise ValueError(f"token {t}: character {i} ({text[i]!r}) is in no word")
        out[t] = NO_WORD if i < 0 else cw[i]
    return out


def pos_classes(word_idx: np.ndarray, words: Sequence[Word], common: frozenset[str]) -> np.ndarray:
    """KNOWLEDGE / MIDDLE / STYLE code (int8) per token from its word index; whitespace-only tokens are STYLE."""
    per_word = np.array([word_class(w.pos, w.lower, common) for w in words] + [STYLE], dtype=np.int8)
    return per_word[np.where(word_idx == NO_WORD, len(words), word_idx)]


def token_pos_tags(word_idx: np.ndarray, words: Sequence[Word]) -> np.ndarray:
    """POS tag per token (WS_TAG for whitespace-only tokens), as a string array."""
    tags = np.array([w.pos for w in words] + [WS_TAG], dtype=object)
    return tags[np.where(word_idx == NO_WORD, len(words), word_idx)]


def frequency_classes(ids: Sequence[int]) -> np.ndarray:
    """RARE (int8) where the token id's count in the sequence is below the median count over all occurrences,
    else FREQUENT."""
    a = np.asarray(ids)
    if a.size == 0:
        return np.zeros(0, dtype=np.int8)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    f = counts[inv.ravel()]
    return np.where(f < np.median(f), RARE, FREQUENT).astype(np.int8)


def class_sums(bits: np.ndarray, codes: np.ndarray, n_classes: int) -> np.ndarray:
    """Total bits per class code 0..n_classes−1 (float64); the entries sum to bits.sum()."""
    if bits.shape != codes.shape:
        raise ValueError(f"bits {bits.shape} vs codes {codes.shape}")
    return np.bincount(codes.astype(np.intp), weights=bits.astype(np.float64), minlength=n_classes)


def window_class_sums(bits: np.ndarray, codes: np.ndarray, lengths: Sequence[int], n_classes: int) -> np.ndarray:
    """(n_windows, n_classes) bit totals; window w is the next lengths[w] consecutive tokens."""
    if bits.shape != codes.shape or sum(lengths) != bits.size:
        raise ValueError(f"bits {bits.shape}, codes {codes.shape}, windows cover {sum(lengths)} tokens")
    win = np.repeat(np.arange(len(lengths)), lengths)
    flat = win * n_classes + codes.astype(np.intp)
    sums = np.bincount(flat, weights=bits.astype(np.float64), minlength=len(lengths) * n_classes)
    return sums.reshape(len(lengths), n_classes)

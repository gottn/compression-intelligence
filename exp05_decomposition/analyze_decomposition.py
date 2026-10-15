"""exp05: split post-training drift into token classes (knowledge / middle / style) and test which class predicts
forgetting and which predicts the instruction-following gain.

    .venv/bin/python exp05_decomposition/analyze_decomposition.py align     classes per tokenizer group and corpus,
                                                                           alignment checks (no bits summed by class)

Reads   ../exp01_posttraining_bpc/data/<corpus>.txt                       (the scored text, whole files)
        ../exp03_staged_drift/results/{tokens/*.npy, windows.csv, bpb.csv} (14 models, AAU T4)
        ../exp01_posttraining_bpc/results/{tokens/*__raw__*.npy, bpb.csv}  (SmolLM2 and Qwen3 pairs, Mac MPS)
        HF-cache tokenizers of all 18 models; spaCy en_core_web_sm; wordfreq's English list
Writes  results/classes/<tokenizer group>__<corpus>.npz, results/alignment.json, results/class_counts.csv,
        results/class_examples.md
Pure functions above the effects marker; tokenizers, spaCy and file I/O only below it.

Definitions are the pre-registered ones in RESULTS.md. Alignment: ids and character offsets from the fast tokenizer
on exactly the scored text; a token takes the POS and lowercased form of the spaCy word covering its first
non-whitespace character (cilab/tokclass.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cilab`

import numpy as np
import pandas as pd

from cilab.io import load_corpora, load_token_bits
from cilab.pairs import CORPORA
from cilab.stages import EXP03_STAGES, STAGES, tokenizer_groups
from cilab.text import Window, decode, encode, make_windows
from cilab.tokclass import (
    FREQ_CLASSES,
    POS_CLASSES,
    REPLACEMENT,
    Word,
    check_offsets,
    frequency_classes,
    offset_mismatches,
    pos_classes,
    token_pos_tags,
    token_word_index,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP01, EXP03 = ROOT / "exp01_posttraining_bpc", ROOT / "exp03_staged_drift"
WINDOW = 1024
TOP_COMMON = 2000
SPACY_MODEL, SPACY_EXCLUDE = "en_core_web_sm", ("parser", "ner", "lemmatizer")
EXP03_MODELS = frozenset(s.model for s in EXP03_STAGES)
UPOS: tuple[str, ...] = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
                         "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE", "WS")  # spaCy's UPOS + SPACE; WS = no word
EXAMPLE_GROUP, EXAMPLE_SLICE, EXAMPLE_TOP = "qwen2.5", (2000, 2040), 12


# ----------------------------------------------------------------------------- pure
def ids_digest(ids: Sequence[int]) -> str:
    """exp03's `ids_sha1`: first 16 hex digits of the SHA-1 of the ids as int64."""
    return hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:16]


def window_lengths(wins: Sequence[Window]) -> tuple[int, ...]:
    return tuple(e - b for b, e, _ in wins)


def classify(text: str, offsets: Sequence[tuple[int, int]], pieces: Sequence[str], ids: Sequence[int],
             words: Sequence[Word], common: frozenset[str]) -> dict[str, object]:
    """Checked alignment of one tokenization to one tagged text: POS class, frequency class and POS-tag code per
    token, plus the alignment counts. Raises if the offsets do not tile the text or a whole piece disagrees with it."""
    check_offsets(offsets, len(text))
    bad = offset_mismatches(text, offsets, pieces)
    if bad:
        t = bad[0]
        raise ValueError(f"{len(bad)} tokens decode to other text than their offsets, first {t}: "
                         f"{pieces[t]!r} vs {text[slice(*offsets[t])]!r}")
    widx = token_word_index(text, offsets, words)
    tags = token_pos_tags(widx, words)
    unknown = sorted(set(tags.tolist()) - set(UPOS))
    if unknown:
        raise ValueError(f"POS tags outside UPOS: {unknown}")
    tag_code = {t: i for i, t in enumerate(UPOS)}
    return {"pos": pos_classes(widx, words, common), "freq": frequency_classes(ids),
            "tag": np.array([tag_code[t] for t in tags.tolist()], dtype=np.int8),
            "n_tokens": len(ids), "n_split_char_pieces": sum(REPLACEMENT in p for p in pieces),
            "n_whole_pieces_checked": sum(REPLACEMENT not in p for p in pieces)}


def verify_model(model: str, group: str, corpus: str, ids: Sequence[int], ref_ids: Sequence[int], n_bits: int,
                 lengths: tuple[int, ...], exp03_lengths: tuple[int, ...] | None, exp03_sha1: str | None,
                 exp01_tokens: int | None) -> dict[str, object]:
    """One (model, corpus) alignment record; raises on any mismatch with the saved bits or the scorer's records."""
    rec: dict[str, object] = {"model": model, "tokenizer": group, "corpus": corpus, "n_ids": len(ids),
                              "n_bits": n_bits, "n_windows": len(lengths), "ids_sha1": ids_digest(ids)}
    checks = {"ids_identical_to_group": list(ids) == list(ref_ids), "n_ids_equals_n_bits": len(ids) == n_bits,
              "windows_cover_bits": sum(lengths) == n_bits}
    if exp03_lengths is not None:
        checks["windows_equal_exp03"] = lengths == exp03_lengths
    if exp03_sha1 is not None:
        checks["sha1_equals_exp03"] = rec["ids_sha1"] == exp03_sha1
    if exp01_tokens is not None:
        checks["tokens_equal_exp01"] = len(ids) == exp01_tokens
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{model} on {corpus}: alignment check failed: {failed} ({rec})")
    return {**rec, **checks}


def class_count_rows(group: str, corpus: str, cls: Mapping[str, np.ndarray]) -> list[dict[str, object]]:
    """Token counts per class (pos and freq schemes) and per (pos class, POS tag)."""
    n = len(cls["pos"])
    rows = []
    for scheme, names in (("pos", POS_CLASSES), ("freq", FREQ_CLASSES)):
        counts = np.bincount(cls[scheme].astype(np.intp), minlength=len(names))
        rows += [{"tokenizer": group, "corpus": corpus, "scheme": scheme, "class": c, "pos": "*",
                  "n_tokens": int(k), "frac_tokens": k / n} for c, k in zip(names, counts, strict=True)]
    for ci, c in enumerate(POS_CLASSES):
        tags = np.bincount(cls["tag"][cls["pos"] == ci].astype(np.intp), minlength=len(UPOS))
        rows += [{"tokenizer": group, "corpus": corpus, "scheme": "pos", "class": c, "pos": UPOS[t],
                  "n_tokens": int(k), "frac_tokens": k / n} for t, k in enumerate(tags) if k]
    return rows


def examples_markdown(examples: Mapping[str, Mapping[str, object]]) -> str:
    """Per corpus (one tokenizer group): a fixed slice of tokens with tag and class, and the most frequent pieces of
    each class, so the classification can be read by eye."""
    lo, hi = EXAMPLE_SLICE
    out = ["# exp05 class examples (auto-generated by analyze_decomposition.py align)", "",
           f"Tokenizer group `{EXAMPLE_GROUP}`. Per corpus: tokens {lo}..{hi - 1} with the POS of the word under "
           f"their first non-space character and their class, then the {EXAMPLE_TOP} most frequent pieces of each "
           "class. `WS` = a token made only of whitespace.", ""]
    for corpus, ex in examples.items():
        pieces, tags, pos, freq = ex["pieces"], ex["tags"], ex["pos"], ex["freq"]
        out += [f"## {corpus}", "", "| # | piece | POS | class | freq class |", "|---|---|---|---|---|"]
        out += [f"| {i} | `{pieces[i]!r}` | {UPOS[tags[i]]} | {POS_CLASSES[pos[i]]} | {FREQ_CLASSES[freq[i]]} |"
                for i in range(lo, min(hi, len(pieces)))]
        out += ["", "| class | most frequent pieces (count) |", "|---|---|"]
        for ci, c in enumerate(POS_CLASSES):
            sel = pd.Series([p for p, k in zip(pieces, pos, strict=True) if k == ci]).value_counts().head(EXAMPLE_TOP)
            out.append(f"| {c} | " + ", ".join(f"`{p!r}` ({n})" for p, n in sel.items()) + " |")
        out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------------- effects
def bits_dir(model: str) -> Path:
    """exp03's results for its 14 models, exp01's (RAW rows) for the SmolLM2 and Qwen3 pairs."""
    return EXP03 / "results" if model in EXP03_MODELS else EXP01 / "results"


def tag_corpus(nlp: object, text: str) -> tuple[Word, ...]:
    """spaCy words of the whole text: character span, Universal POS tag, lowercased form."""
    return tuple(Word(t.idx, t.idx + len(t.text), t.pos_, t.lower_) for t in nlp(text))


def load_tokenizer(model: str) -> object:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if not tok.is_fast:
        raise ValueError(f"{model}: offsets need a fast tokenizer")
    return tok


def cmd_align(out: Path) -> None:
    import spacy
    import wordfreq

    corpora = load_corpora(EXP01 / "data", CORPORA, None)
    nlp = spacy.load(SPACY_MODEL, exclude=list(SPACY_EXCLUDE))
    common = frozenset(wordfreq.top_n_list("en", TOP_COMMON))
    words = {c: tag_corpus(nlp, t) for c, t in corpora.items()}
    win = pd.read_csv(EXP03 / "results" / "windows.csv").drop_duplicates(["model", "corpus", "window"], keep="last")
    e3_len = {k: tuple(g.sort_values("window").tokens) for k, g in win.groupby(["model", "corpus"])}
    e3_bpb = pd.read_csv(EXP03 / "results" / "bpb.csv").drop_duplicates(["model", "condition", "corpus"], keep="last")
    e3_sha = {(r.model, r.corpus): r.ids_sha1 for r in e3_bpb[e3_bpb.condition == "raw"].itertuples()}
    e1_bpb = pd.read_csv(EXP01 / "results" / "bpb.csv").drop_duplicates(["model", "condition", "corpus"], keep="last")
    e1_tok = {(r.model, r.corpus): int(r.tokens) for r in e1_bpb[e1_bpb.condition == "raw"].itertuples()}
    (out / "classes").mkdir(parents=True, exist_ok=True)
    records, groups, counts, examples = [], {}, [], {}
    for group, models in tokenizer_groups(STAGES).items():
        toks = {m: load_tokenizer(m) for m in models}
        ref = toks[models[0]]
        groups[group] = {"classified_with": models[0], "models": models, "corpora": {}}
        for corpus, text in corpora.items():
            enc = ref(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
            if ids != encode(ref, text):
                raise ValueError(f"{group} on {corpus}: offsets call gives other ids than cilab.text.encode")
            pieces = [decode(ref, [i]) for i in ids]
            cls = classify(text, offsets, pieces, ids, words[corpus], common)
            lengths = window_lengths(make_windows(ref, ids, text, WINDOW))
            for m in models:
                m_ids = ids if m == models[0] else encode(toks[m], text)
                n_bits = load_token_bits(bits_dir(m), m, "raw", corpus).size
                in3 = m in EXP03_MODELS
                records.append(verify_model(m, group, corpus, m_ids, ids, n_bits, lengths,
                                            e3_len.get((m, corpus)) if in3 else None,
                                            e3_sha.get((m, corpus)) if in3 else None,
                                            None if in3 else e1_tok.get((m, corpus))))
            np.savez_compressed(out / "classes" / f"{group}__{corpus}.npz", pos=cls["pos"], freq=cls["freq"],
                                tag=cls["tag"], lengths=np.array(lengths, dtype=np.int64),
                                ids_sha1=np.array(ids_digest(ids)))
            groups[group]["corpora"][corpus] = {k: cls[k] for k in ("n_tokens", "n_split_char_pieces",
                                                                     "n_whole_pieces_checked")}
            counts += class_count_rows(group, corpus, cls)
            if group == EXAMPLE_GROUP:
                examples[corpus] = {"pieces": pieces, "tags": cls["tag"], "pos": cls["pos"], "freq": cls["freq"]}
            print(json.dumps({"group": group, "corpus": corpus, "tokens": len(ids), "windows": len(lengths)}), flush=True)
    report = {"n_pairs_verified": len(records), "all_checks_passed": True,
              "spacy": spacy.__version__, "spacy_model": f"{SPACY_MODEL} {nlp.meta['version']}",
              "spacy_pipes": nlp.pipe_names, "wordfreq": wordfreq.__version__ if hasattr(wordfreq, "__version__")
              else _dist_version("wordfreq"), "common_words": TOP_COMMON,
              "common_sha1": hashlib.sha1("\n".join(sorted(common)).encode()).hexdigest()[:16],
              "spacy_words": {c: len(w) for c, w in words.items()}, "groups": groups, "pairs": records}
    (out / "alignment.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    pd.DataFrame(counts).to_csv(out / "class_counts.csv", index=False, float_format="%.5f")
    (out / "class_examples.md").write_text(examples_markdown(examples))
    print(f"verified {len(records)} (model, corpus) pairs")


def _dist_version(name: str) -> str:
    from importlib.metadata import version
    return version(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["align"])
    ap.add_argument("--results", default=str(HERE / "results"))
    args = ap.parse_args()
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    {"align": cmd_align}[args.cmd](out)


if __name__ == "__main__":
    main()

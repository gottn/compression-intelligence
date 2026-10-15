"""exp05: split post-training drift into token classes (knowledge / middle / style) and test which class predicts
forgetting and which predicts the instruction-following gain.

    .venv/bin/python exp05_decomposition/analyze_decomposition.py align     classes per tokenizer group and corpus,
                                                                           alignment checks (no bits summed by class)
    .venv/bin/python exp05_decomposition/analyze_decomposition.py analyze   per-class drift and share with window-
                                                                           bootstrap CIs, correlations, H1-H4, figure

Reads   ../exp01_posttraining_bpc/data/<corpus>.txt                       (the scored text, whole files)
        ../exp03_staged_drift/results/{tokens/*.npy, windows.csv, bpb.csv} (14 models, AAU T4)
        ../exp01_posttraining_bpc/results/{tokens/*__raw__*.npy, bpb.csv}  (SmolLM2 and Qwen3 pairs, Mac MPS)
        HF-cache tokenizers of all 18 models; spaCy en_core_web_sm; wordfreq's English list (align only)
        ../exp04_benchmarks/results/scores.csv                             (forgetting, IFEval chat gain, MMLU)
Writes  align:   results/classes/<tokenizer group>__<corpus>.npz, results/alignment.json, results/class_counts.csv,
                 results/class_examples.md
        analyze: results/decomposition.csv, results/correlations.csv, results/summary.md, results/fig_decomposition.png
Pure functions above the effects marker; tokenizers, spaCy and file I/O only below it.

Definitions are the pre-registered ones in RESULTS.md. Alignment: ids and character offsets from the fast tokenizer
on exactly the scored text; a token takes the POS and lowercased form of the spaCy word covering its first
non-whitespace character (cilab/tokclass.py). Per post-trained checkpoint m, its family base b, corpus and class c:
drift_c = 100 (Σ_c bits_m − Σ_c bits_b) / Σ_c bits_b; share_c = 100 (Σ_c bits_m − Σ_c bits_b) / (Σ bits_m − Σ bits_b);
fresh = mean of the 3 fresh corpora. CIs: 5000 paired window resamples, seed 0 + corpus index (exp03); correlations:
10,000 checkpoint resamples, seed 0, leave-one-out (exp04).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cilab`

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cilab.io import load_corpora, load_token_bits
from cilab.pairs import CORPORA, FRESH_CORPORA
from cilab.stages import EXP03_STAGES, STAGES, Stage, family_base, tokenizer_groups
from cilab.stats import (
    bootstrap_ci,
    bootstrap_corr,
    leave_one_out,
    paired_pct_change,
    paired_share,
    pearson,
    resample_indices,
    spearman,
)
from cilab.text import Window, decode, encode, make_windows
from cilab.tokclass import (
    FREQ_CLASSES,
    POS_CLASSES,
    REPLACEMENT,
    Word,
    check_offsets,
    class_sums,
    frequency_classes,
    offset_mismatches,
    pos_classes,
    token_pos_tags,
    token_word_index,
    window_class_sums,
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
EXP04 = ROOT / "exp04_benchmarks"
N_BOOT_WIN, N_BOOT_CORR, SEED = 5000, 10_000, 0  # exp03's window bootstrap, exp04's checkpoint bootstrap
FRESH, ENWIK8, ALL = "fresh_mean", "enwik8_ctrl", "all"
REPRO_TOL_PP = 1e-3  # |drift over all tokens − exp04 Δbpb|, percentage points
SCHEMES: dict[str, tuple[str, ...]] = {"pos": POS_CLASSES, "freq": FREQ_CLASSES}
CHAT_STAGES: tuple[str, ...] = ("sft", "dpo", "rlvr", "instruct")
MATH_STAGES: tuple[str, ...] = ("math-cpt", "r1-distill")
ALL12, NO_MATH, CHAT8 = "all 12", "without Qwen2.5-Math and R1-Distill", "chat-tuned (sft/dpo/rlvr/instruct)"
SMOL_INSTRUCT, OLMO_DPO = "HuggingFaceTB/SmolLM2-1.7B-Instruct", "allenai/OLMo-2-0425-1B-DPO"
H1_R_NO_MATH, H2_STYLE_MIN, H2_KNOWLEDGE_MAX, ONE_DIM_R = 0.6, 0.8, 0.5, 0.9
E2_KNOWLEDGE_HEAVY = ("Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", SMOL_INSTRUCT)
PREDICTORS: tuple[str, ...] = ("dbpb_fresh", "drift_knowledge", "drift_middle", "drift_style", "share_knowledge",
                               "share_style", "contrast_ks", "drift_rare", "drift_frequent", "contrast_rf", "excess",
                               "dbpb_enwik8")
SUBSET_OUTCOMES: dict[str, tuple[str, ...]] = {ALL12: ("forgetting", "mmlu_loss", "mmlu_loss_pts"),
                                               NO_MATH: ("forgetting", "mmlu_loss"), CHAT8: ("gain_ifeval_chat",)}
CHANNEL_PAIRS: tuple[tuple[str, str], ...] = (("drift_knowledge", "drift_style"), ("drift_rare", "drift_frequent"))
LABEL = {"allenai/OLMo-2-0425-1B-SFT": "OLMo-2 SFT", "allenai/OLMo-2-0425-1B-DPO": "OLMo-2 DPO",
         "allenai/OLMo-2-0425-1B-Instruct": "OLMo-2 RLVR", "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B Inst",
         "hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo": "SimpleRL 0.5B", "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B Inst",
         "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo": "SimpleRL 1.5B", "Qwen/Qwen2.5-Math-1.5B": "Qwen2.5-Math",
         "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "R1-Distill", "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B Inst",
         SMOL_INSTRUCT: "SmolLM2 Inst", "Qwen/Qwen3-1.7B": "Qwen3-1.7B"}
# dataviz reference palette slots 1-2 (validated light-mode pair) for the two classes under test; MIDDLE is the
# remainder bucket and stays neutral. Text in ink tokens, never series colours.
CLASS_COLOR = {"knowledge": "#2a78d6", "style": "#eb6834", "middle": "#b8b7b1"}
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"

CLASS_TAGS = {"knowledge": ("PROPN", "NUM"), "style": ("DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PART", "INTJ",
                                                        "PUNCT", "SPACE", "SYM", "WS")}  # for labelling only
Key = tuple[str, str, str]  # (model, corpus, scheme)
# figure label placement: model -> (dx, dy points, ha), None = covered by a group label
Place = Mapping[str, tuple[float, float, str] | None]


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


def label(model: str) -> str:
    return LABEL.get(model, model.split("/")[-1])


def corpus_seed(corpus: str) -> int:
    return SEED + CORPORA.index(corpus)


def post_stages(stages: tuple[Stage, ...] = STAGES) -> list[Stage]:
    return [s for s in stages if s.parent is not None]


def corpus_rows(wb: np.ndarray, wm: np.ndarray, counts: np.ndarray, names: tuple[str, ...], idx: np.ndarray,
                meta: Mapping[str, object]) -> tuple[list[dict[str, object]], dict[str, tuple]]:
    """Rows for one (checkpoint, corpus, scheme): per class and over all tokens, drift % and share % with 95% CIs.
    wb, wm: (n_windows, n_classes) base / post bit totals on identical windows; counts: tokens per class.
    Also returns the bootstrap statistics, for the fresh mean."""
    rows, stats = [], {}
    tb, tm, n = wb.sum(1), wm.sum(1), counts.sum()
    for ci, c in enumerate((*names, ALL)):
        b, m = (tb, tm) if c == ALL else (wb[:, ci], wm[:, ci])
        k = n if c == ALL else counts[ci]
        drift = paired_pct_change(b, m)
        share = None if c == ALL else paired_share(b, m, tb, tm)
        d, dlo, dhi = bootstrap_ci(drift, idx)
        s, slo, shi = bootstrap_ci(share, idx) if share else (100.0, np.nan, np.nan)
        stats[c] = (drift, share)
        rows.append({**meta, "class": c, "n_windows": len(tb), "n_tokens": int(k), "token_pct": 100 * k / n,
                     "base_bits": float(b.sum()), "post_bits": float(m.sum()),
                     "base_frac_pct": 100 * b.sum() / tb.sum(), "base_bits_per_token": b.sum() / k,
                     "post_bits_per_token": m.sum() / k, "drift_pct": d, "drift_lo": dlo, "drift_hi": dhi,
                     "share_pct": s, "share_lo": slo, "share_hi": shi})
    return rows, stats


def stratified_rows(stats: Sequence, idxs: Sequence[np.ndarray]) -> tuple[float, np.ndarray]:
    """(point, per-resample values) of the mean over corpora of a window statistic, windows resampled within each
    corpus (the rows behind cilab.stats.stratified_mean_ci)."""
    point = float(np.mean([s(np.arange(ix.shape[1])[None, :])[0] for s, ix in zip(stats, idxs, strict=True)]))
    return point, np.mean([s(ix) for s, ix in zip(stats, idxs, strict=True)], axis=0)


def ci95(rows: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(rows, [2.5, 97.5])
    return float(lo), float(hi)


def decomposition_table(wsums: Mapping[Key, np.ndarray], counts: Mapping[Key, np.ndarray],
                        stages: tuple[Stage, ...] = STAGES) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """Every post-trained checkpoint vs its family base, per scheme, class and corpus, plus fresh-mean rows.
    wsums[(model, corpus, scheme)]: (n_windows, n_classes) bits; counts[(model, corpus, scheme)]: tokens per class.
    Also returns the fresh-mean bootstrap rows of share_% per (model, class) for the pos scheme (H3)."""
    rows, share_boot = [], {}
    for s in post_stages(stages):
        base = family_base(s.family, stages)
        meta = {"model": s.model, "family": s.family, "stage": s.stage, "reference": base, "tokenizer": s.tokenizer}
        for scheme, names in SCHEMES.items():
            fresh: dict[str, list] = {c: [] for c in (*names, ALL)}
            idxs, per_corpus = [], []
            for corpus in CORPORA:
                wb, wm = wsums[(base, corpus, scheme)], wsums[(s.model, corpus, scheme)]
                if wb.shape != wm.shape:
                    raise ValueError(f"{s.model} vs {base} on {corpus}: windows differ")
                idx = resample_indices(wb.shape[0], N_BOOT_WIN, corpus_seed(corpus))
                r, st = corpus_rows(wb, wm, counts[(s.model, corpus, scheme)], names, idx,
                                    {**meta, "scheme": scheme, "corpus": corpus})
                rows += r
                if corpus in FRESH_CORPORA:
                    idxs.append(idx)
                    per_corpus.append(pd.DataFrame(r))
                    for c in fresh:
                        fresh[c].append(st[c])
            pc = pd.concat(per_corpus).groupby("class", sort=False)
            for c, sts in fresh.items():
                d, drows = stratified_rows([x[0] for x in sts], idxs)
                g = pc.get_group(c)
                row = {**meta, "scheme": scheme, "corpus": FRESH, "class": c, "n_windows": int(g.n_windows.sum()),
                       "n_tokens": int(g.n_tokens.sum()), "token_pct": g.token_pct.mean(),
                       "base_bits": g.base_bits.sum(), "post_bits": g.post_bits.sum(),
                       "base_frac_pct": g.base_frac_pct.mean(), "base_bits_per_token": g.base_bits_per_token.mean(),
                       "post_bits_per_token": g.post_bits_per_token.mean(), "drift_pct": d,
                       "drift_lo": ci95(drows)[0], "drift_hi": ci95(drows)[1], "share_pct": 100.0,
                       "share_lo": np.nan, "share_hi": np.nan}
                if c != ALL:
                    sh, srows = stratified_rows([x[1] for x in sts], idxs)
                    row.update({"share_pct": sh, "share_lo": ci95(srows)[0], "share_hi": ci95(srows)[1]})
                    if scheme == "pos":
                        share_boot[(s.model, c)] = srows
                rows.append(row)
    return pd.DataFrame(rows), share_boot


def predictor_table(decomp: pd.DataFrame, stages: tuple[Stage, ...] = STAGES) -> pd.DataFrame:
    """One row per post-trained checkpoint: fresh-mean drift per class and over all tokens, shares, the
    knowledge−style and rare−frequent contrasts, Δbpb_enwik8 and the memorisation excess."""
    rows = []
    for s in post_stages(stages):
        d = decomp[decomp.model == s.model]
        fr = d[d.corpus == FRESH].set_index(["scheme", "class"])
        en = d[(d.corpus == ENWIK8) & (d.scheme == "pos") & (d["class"] == ALL)].iloc[0]
        r = {"model": s.model, "family": s.family, "stage": s.stage,
             "dbpb_fresh": fr.loc[("pos", ALL), "drift_pct"], "dbpb_enwik8": en.drift_pct}
        for scheme, names in SCHEMES.items():
            for c in names:
                r[f"drift_{c}"] = fr.loc[(scheme, c), "drift_pct"]
                if scheme == "pos":
                    r[f"share_{c}"] = fr.loc[(scheme, c), "share_pct"]
                    r[f"base_frac_{c}"] = fr.loc[(scheme, c), "base_frac_pct"]
        r["contrast_ks"] = r["drift_knowledge"] - r["drift_style"]
        r["contrast_rf"] = r["drift_rare"] - r["drift_frequent"]
        r["excess"] = r["dbpb_enwik8"] - r["dbpb_fresh"]
        rows.append(r)
    return pd.DataFrame(rows)


def pos_share_table(tag_sums: Mapping[tuple[str, str], np.ndarray], stages: tuple[Stage, ...] = STAGES) -> pd.DataFrame:
    """Share (%) of the fresh-text Δbits per POS tag (point estimates, mean over the 3 fresh corpora), one column per
    post-trained checkpoint, plus `null` = the tag's mean % of base bits over the 12 comparisons.
    tag_sums[(model, corpus)]: total bits per UPOS index."""
    cols, nulls = {}, []
    for s in post_stages(stages):
        base = family_base(s.family, stages)
        sh, nl = [], []
        for c in FRESH_CORPORA:
            b, m = tag_sums[(base, c)], tag_sums[(s.model, c)]
            sh.append(100 * (m - b) / (m - b).sum())
            nl.append(100 * b / b.sum())
        cols[label(s.model)] = np.mean(sh, axis=0)
        nulls.append(np.mean(nl, axis=0))
    out = pd.DataFrame(cols, index=list(UPOS))
    out.insert(0, "null", np.mean(nulls, axis=0))
    return out


def outcome_table(scores: pd.DataFrame, stages: tuple[Stage, ...] = STAGES) -> pd.DataFrame:
    """exp04 outcomes per post-trained checkpoint: forgetting (%), IFEval chat gain (points), MMLU loss (% of the
    base and points), and exp04's own Δbpb predictors for the reproduction check."""
    sc = scores.set_index("model")
    rows = []
    for s in post_stages(stages):
        post, base = sc.loc[s.model], sc.loc[family_base(s.family, stages)]
        rows.append({"model": s.model, "forgetting": post.forgetting, "gain_ifeval_chat": post.gain_ifeval_chat,
                     "mmlu_loss": 100 * (base.mmlu - post.mmlu) / base.mmlu,
                     "mmlu_loss_pts": 100 * (base.mmlu - post.mmlu),
                     "exp04_dbpb_fresh": post.dbpb_fresh, "exp04_dbpb_enwik8": post.dbpb_enwik8})
    return pd.DataFrame(rows)


def reproduction_table(pred: pd.DataFrame, outc: pd.DataFrame) -> pd.DataFrame:
    """Drift over all tokens here vs exp04's Δbpb predictors; raises beyond REPRO_TOL_PP."""
    t = pred[["model", "dbpb_fresh", "dbpb_enwik8"]].merge(outc[["model", "exp04_dbpb_fresh", "exp04_dbpb_enwik8"]])
    t["diff_fresh_pp"] = t.dbpb_fresh - t.exp04_dbpb_fresh
    t["diff_enwik8_pp"] = t.dbpb_enwik8 - t.exp04_dbpb_enwik8
    worst = float(t[["diff_fresh_pp", "diff_enwik8_pp"]].abs().max().max())
    if not worst <= REPRO_TOL_PP:
        raise ValueError(f"total drift does not reproduce exp04: worst {worst:.2e} pp\n{t}")
    return t


def subsets(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {ALL12: table, NO_MATH: table[~table.stage.isin(MATH_STAGES)].reset_index(drop=True),
            CHAT8: table[table.stage.isin(CHAT_STAGES)].reset_index(drop=True)}


def corr_row(df: pd.DataFrame, x: str, y: str, subset: str) -> dict[str, object]:
    """Pearson and Spearman with checkpoint-bootstrap CIs and the leave-one-out range of Pearson r (exp04)."""
    d = df[[x, y, "model"]].dropna()
    xv, yv = d[x].to_numpy(dtype=float), d[y].to_numpy(dtype=float)
    p, plo, phi, pn = bootstrap_corr(xv, yv, pearson, N_BOOT_CORR, SEED)
    s, slo, shi, sn = bootstrap_corr(xv, yv, spearman, N_BOOT_CORR, SEED)
    loo = leave_one_out(xv, yv, pearson)
    worst = int(np.argmax(np.abs(loo - p)))
    return {"predictor": x, "outcome": y, "subset": subset, "n": len(d), "pearson": p, "pearson_lo": plo,
            "pearson_hi": phi, "spearman": s, "spearman_lo": slo, "spearman_hi": shi, "boot_valid": min(pn, sn),
            "loo_min": float(loo.min()), "loo_max": float(loo.max()),
            "loo_most_influential": label(d.model.iloc[worst]), "loo_r_without_it": float(loo[worst])}


def correlation_table(table: pd.DataFrame) -> pd.DataFrame:
    """Every predictor against the subset's outcomes, plus the channel pairs (knowledge vs style, rare vs frequent)
    on every subset."""
    sets = subsets(table)
    rows = [corr_row(sets[name], p, o, name) for name, outs in SUBSET_OUTCOMES.items() for p in PREDICTORS
            for o in outs]
    rows += [corr_row(sets[name], a, b, name) for name in sets for a, b in CHANNEL_PAIRS]
    return pd.DataFrame(rows)


def lookup(corr: pd.DataFrame, x: str, y: str, subset: str, stat: str) -> float:
    hit = corr[(corr.predictor == x) & (corr.outcome == y) & (corr.subset == subset)]
    if len(hit) != 1:
        raise KeyError(f"{x} vs {y} on {subset}: {len(hit)} rows")
    return float(hit[stat].iloc[0])


def verdict(ok: bool) -> str:
    return "pass" if ok else "fail"


def rule_table(corr: pd.DataFrame, decomp: pd.DataFrame, pred: pd.DataFrame,
               share_boot: Mapping[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    """The pre-registered tests (H1-H4, the one-dimensional fail rule), the frequency-variant and contrast readings,
    and the expectations E1-E2, each with its numbers and verdict."""
    def lk(x: str, y: str, sub: str, st: str) -> float:
        return lookup(corr, x, y, sub, st)

    rows = []
    for tag, k, s in (("", "drift_knowledge", "drift_style"), (" (freq variant)", "drift_rare", "drift_frequent")):
        rho_k, rho_t = lk(k, "forgetting", ALL12, "spearman"), lk("dbpb_fresh", "forgetting", ALL12, "spearman")
        r10 = lk(k, "forgetting", NO_MATH, "pearson")
        rows.append({"id": "H1" + tag, "rule": f"ρ({k}, forgetting) ≥ ρ(Δbpb_fresh, forgetting) and r without math "
                     f"≥ {H1_R_NO_MATH}", "observed": f"ρ = {rho_k:.3f} vs {rho_t:.3f}; r(n=10) = {r10:.3f}",
                     "verdict": verdict(rho_k >= rho_t and r10 >= H1_R_NO_MATH)})
        r_s, r_k = lk(s, "gain_ifeval_chat", CHAT8, "pearson"), lk(k, "gain_ifeval_chat", CHAT8, "pearson")
        rows.append({"id": "H2" + tag, "rule": f"r({s}, IFEval chat gain) ≥ {H2_STYLE_MIN} and r({k}, gain) < "
                     f"{H2_KNOWLEDGE_MAX}, n = 8", "observed": f"r_style = {r_s:.3f}; r_knowledge = {r_k:.3f}",
                     "verdict": verdict(r_s >= H2_STYLE_MIN and r_k < H2_KNOWLEDGE_MAX)})
        r_ks = lk(k, s, ALL12, "pearson")
        rows.append({"id": "one-dimensional" + tag, "rule": f"r({k}, {s}) over the 12 > {ONE_DIM_R} (the fail rule)",
                     "observed": f"r = {r_ks:.3f} (ρ = {lk(k, s, ALL12, 'spearman'):.3f}; without math r = "
                     f"{lk(k, s, NO_MATH, 'pearson'):.3f}; chat-tuned r = {lk(k, s, CHAT8, 'pearson'):.3f})",
                     "verdict": "triggered" if r_ks > ONE_DIM_R else "not triggered"})
    fr = decomp[(decomp.corpus == FRESH) & (decomp.scheme == "pos") & (decomp["class"] == "knowledge")]
    a, b = fr[fr.model == SMOL_INSTRUCT].iloc[0], fr[fr.model == OLMO_DPO].iloc[0]
    diff = share_boot[(SMOL_INSTRUCT, "knowledge")] - share_boot[(OLMO_DPO, "knowledge")]
    dlo, dhi = ci95(diff)
    rows.append({"id": "H3", "rule": "share_knowledge CI of SmolLM2-Instruct entirely above OLMo-2-DPO's",
                 "observed": f"{a.share_pct:.1f}% [{a.share_lo:.1f}, {a.share_hi:.1f}] vs {b.share_pct:.1f}% "
                 f"[{b.share_lo:.1f}, {b.share_hi:.1f}]; difference {a.share_pct - b.share_pct:+.1f} pp "
                 f"[{dlo:+.1f}, {dhi:+.1f}]", "verdict": verdict(a.share_lo > b.share_hi)})
    rho_e, rho_f = lk("excess", "mmlu_loss", ALL12, "spearman"), lk("dbpb_fresh", "mmlu_loss", ALL12, "spearman")
    rows.append({"id": "H4", "rule": "ρ(excess, MMLU loss) ≥ ρ(Δbpb_fresh, MMLU loss), n = 12",
                 "observed": f"ρ = {rho_e:.3f} vs {rho_f:.3f}", "verdict": verdict(rho_e >= rho_f)})
    for sub, out in ((ALL12, "forgetting"), (NO_MATH, "forgetting"), (CHAT8, "gain_ifeval_chat")):
        rows.append({"id": "contrast (reading)", "rule": f"contrast_ks vs {out}, {sub}",
                     "observed": f"r = {lk('contrast_ks', out, sub, 'pearson'):.3f}, "
                     f"ρ = {lk('contrast_ks', out, sub, 'spearman'):.3f}", "verdict": "reading"})
    p = pred.set_index("model")
    rows.append({"id": "E1", "rule": "the fail rule triggers", "observed": f"r = {lk('drift_knowledge', 'drift_style', ALL12, 'pearson'):.3f}",
                 "verdict": "met" if lk("drift_knowledge", "drift_style", ALL12, "pearson") > ONE_DIM_R else "not met"})
    heavy = all(p.loc[m, "drift_knowledge"] > p.loc[m, "drift_style"] for m in E2_KNOWLEDGE_HEAVY)
    olmo = p.loc[OLMO_DPO, "drift_style"] >= p.loc[OLMO_DPO, "drift_knowledge"]
    obs = "; ".join(f"{label(m)} K {p.loc[m, 'drift_knowledge']:+.1f} / S {p.loc[m, 'drift_style']:+.1f}"
                    for m in (*E2_KNOWLEDGE_HEAVY, OLMO_DPO))
    rows.append({"id": "E2", "rule": "drift_K > drift_S for Math, R1, SmolLM2-Instruct; drift_S ≥ drift_K for OLMo-2 DPO",
                 "observed": obs, "verdict": "met" if heavy and olmo else "not met"})
    return pd.DataFrame(rows)


def fmt(v: float, lo: float, hi: float, digits: int = 2) -> str:
    return f"{v:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def summary_markdown(tests: pd.DataFrame, decomp: pd.DataFrame, pred: pd.DataFrame, outc: pd.DataFrame,
                     corr: pd.DataFrame, repro: pd.DataFrame, pos_share: pd.DataFrame) -> str:
    out = ["# exp05 summary (auto-generated by analyze_decomposition.py analyze)", "",
           f"Per-token bits, RAW, each post-trained checkpoint vs its family base. Window-bootstrap 95% CIs "
           f"({N_BOOT_WIN} paired resamples, seed {SEED} + corpus index); correlations: {N_BOOT_CORR} checkpoint "
           f"resamples, seed {SEED}. 'fresh' = mean of the 3 fresh corpora.", "",
           "## Pre-registered tests, readings and expectations", "", "| id | rule | observed | verdict |",
           "|---|---|---|---|"]
    out += [f"| {r.id} | {r.rule} | {r.observed} | {r.verdict} |" for r in tests.itertuples()]
    fr = decomp[(decomp.corpus == FRESH) & (decomp.scheme == "pos")]
    out += ["", "## Where the drift sits (fresh mean, pos classes)", "",
            "share = % of the total Δbits; null = the class's % of base bits (share if every token's bits grew by "
            "one factor); drift = relative change of the class's bits.", "",
            "| checkpoint | Δbpb all | share K (null) | share M (null) | share S (null) | drift K | drift M | drift S |",
            "|---|---|---|---|---|---|---|---|"]
    for m in pred.model:
        g = fr[fr.model == m].set_index("class")
        cells = [f"{g.loc[c, 'share_pct']:.1f} [{g.loc[c, 'share_lo']:.1f}, {g.loc[c, 'share_hi']:.1f}] "
                 f"({g.loc[c, 'base_frac_pct']:.1f})" for c in POS_CLASSES]
        drifts = [fmt(g.loc[c, "drift_pct"], g.loc[c, "drift_lo"], g.loc[c, "drift_hi"]) for c in POS_CLASSES]
        out.append(f"| {label(m)} | {g.loc[ALL, 'drift_pct']:+.2f}% | " + " | ".join(cells + drifts) + " |")
    pos_cls = {tag: next((c for c in POS_CLASSES if c != "middle" and tag in CLASS_TAGS[c]), "knowledge/middle")
               for tag in UPOS}
    out += ["", "## Share of the fresh Δbits per POS tag (%, point estimates; null = mean % of base bits)", "",
            "NOUN/ADJ/VERB tokens split between knowledge (rare word) and middle (common word).", "",
            "| POS | class | " + " | ".join(pos_share.columns) + " |", "|---|---|" + "---|" * len(pos_share.columns)]
    out += [f"| {tag} | {pos_cls[tag]} | " + " | ".join(f"{v:.1f}" for v in row) + " |"
            for tag, row in pos_share.iterrows() if row.abs().max() >= 0.5]
    out += ["", "## Drift per class and corpus (pos and freq schemes, %)", "",
            "| checkpoint | scheme | class | " + " | ".join(CORPORA) + f" | {FRESH} |",
            "|---|---|---|" + "---|" * (len(CORPORA) + 1)]
    for m in pred.model:
        for scheme, names in SCHEMES.items():
            for c in (*names, ALL):
                if scheme == "freq" and c == ALL:
                    continue
                g = decomp[(decomp.model == m) & (decomp.scheme == scheme) & (decomp["class"] == c)].set_index("corpus")
                out.append(f"| {label(m)} | {scheme} | {c} | " + " | ".join(
                    fmt(g.loc[k, "drift_pct"], g.loc[k, "drift_lo"], g.loc[k, "drift_hi"]) for k in (*CORPORA, FRESH))
                    + " |")
    t = pred.merge(outc, on="model")
    out += ["", "## Per-checkpoint predictors and outcomes", "",
            "| checkpoint | Δbpb fresh | drift K | drift S | contrast K−S | drift rare | drift freq | excess (enwik8 − "
            "fresh) | forgetting % | MMLU loss % | IFEval chat gain |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    out += [f"| {label(r.model)} | {r.dbpb_fresh:+.2f} | {r.drift_knowledge:+.2f} | {r.drift_style:+.2f} | "
            f"{r.contrast_ks:+.2f} | {r.drift_rare:+.2f} | {r.drift_frequent:+.2f} | {r.excess:+.2f} | "
            f"{r.forgetting:+.2f} | {r.mmlu_loss:+.2f} | {r.gain_ifeval_chat:+.1f} |" for r in t.itertuples()]
    out += ["", "## Correlations", "", "| subset | predictor | outcome | n | Pearson [95% CI] | Spearman [95% CI] | "
            "LOO range of r | most influential (r without it) |", "|---|---|---|---|---|---|---|---|"]
    out += [f"| {r.subset} | {r.predictor} | {r.outcome} | {r.n} | {fmt(r.pearson, r.pearson_lo, r.pearson_hi, 3)} | "
            f"{fmt(r.spearman, r.spearman_lo, r.spearman_hi, 3)} | {r.loo_min:+.3f} to {r.loo_max:+.3f} | "
            f"{r.loo_most_influential} ({r.loo_r_without_it:+.3f}) |" for r in corr.itertuples()]
    out += ["", f"## Reproduction: drift over all tokens vs exp04's Δbpb (tolerance {REPRO_TOL_PP:g} pp)", "",
            "| checkpoint | Δbpb fresh here | exp04 | diff | Δbpb enwik8 here | exp04 | diff |",
            "|---|---|---|---|---|---|---|"]
    out += [f"| {label(r.model)} | {r.dbpb_fresh:.5f} | {r.exp04_dbpb_fresh:.5f} | {r.diff_fresh_pp:+.1e} | "
            f"{r.dbpb_enwik8:.5f} | {r.exp04_dbpb_enwik8:.5f} | {r.diff_enwik8_pp:+.1e} |" for r in repro.itertuples()]
    return "\n".join(out) + "\n"


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)


def scatter_panel(ax: plt.Axes, d: pd.DataFrame, x: str, y: str, color: str, xlabel: str, ylabel: str,
                  title: str, place: Place, groups: Sequence[tuple[str, float, float]] = (),
                  ymin: float | None = None) -> None:
    """One predictor against one outcome with selective direct labels (`place`), group labels for tight clusters
    ((text, x, y) in data units) and the OLS line for reference."""
    ax.axhline(0, color=INK2, linewidth=0.8)
    xv, yv = d[x].to_numpy(dtype=float), d[y].to_numpy(dtype=float)
    slope, icpt = np.polyfit(xv, yv, 1)
    grid = np.linspace(min(0.0, xv.min()), xv.max() * 1.04, 50)
    ax.plot(grid, slope * grid + icpt, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.scatter(xv, yv, s=46, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    for r in d.itertuples():
        spec = place.get(r.model, (5, 3, "left"))
        if spec is not None:
            ax.annotate(label(r.model), (getattr(r, x), getattr(r, y)), xytext=spec[:2], textcoords="offset points",
                        ha=spec[2], fontsize=7, color=INK2)
    for text, gx, gy in groups:
        ax.text(gx, gy, text, fontsize=7, color=INK2, ha="left", va="center")
    if ymin is not None:
        ax.set_ylim(ymin, ax.get_ylim()[1])
    ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    style_axis(ax)


def figure(decomp: pd.DataFrame, table: pd.DataFrame, corr: pd.DataFrame) -> Figure:
    """(a) share of the fresh-text Δbits per class per checkpoint, with the proportional-drift boundaries;
    (b) knowledge drift vs forgetting (12); (c) style drift vs IFEval chat gain (8 chat-tuned); (d) the secondary
    knowledge − style contrast vs forgetting (12)."""
    fig = plt.figure(figsize=(15, 10.6), facecolor=SURFACE)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.05], hspace=0.42, wspace=0.26)
    ax = fig.add_subplot(gs[0, :])
    axes = [ax, *(fig.add_subplot(gs[1, i]) for i in range(3))]
    fr = decomp[(decomp.corpus == FRESH) & (decomp.scheme == "pos")]
    order = list(table.model)
    w = 0.7
    for i, m in enumerate(order):
        g = fr[fr.model == m].set_index("class")
        up, down = 0.0, 0.0
        for c in POS_CLASSES:
            v = g.loc[c, "share_pct"]
            bottom = up if v >= 0 else down
            ax.bar(i, v, bottom=bottom, width=w, color=CLASS_COLOR[c], edgecolor=SURFACE, linewidth=1.5)
            up, down = (up + v, down) if v >= 0 else (up, down + v)
        k0 = g.loc["knowledge", "base_frac_pct"]
        for yb in (k0, k0 + g.loc["middle", "base_frac_pct"]):
            ax.plot([i - w / 2, i + w / 2], [yb, yb], color=INK, linewidth=1.3, solid_capstyle="butt", zorder=4)
        ax.annotate(f"{g.loc[ALL, 'drift_pct']:+.1f}%" if abs(g.loc[ALL, "drift_pct"]) >= 1
                    else f"{g.loc[ALL, 'drift_pct']:+.2f}%", (i, max(up, 100)), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label(m) for m in order], rotation=40, ha="right", fontsize=7.5, color=INK2)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("share of the Δbits on fresh text (%)", color=INK2, fontsize=8.5)
    ax.set_title("(a) Where the drift sits: share of Δbits by token class (label: total Δbpb)", fontsize=8.5,
                 color=INK, loc="left")
    style_axis(ax)
    handles = [Patch(facecolor=CLASS_COLOR[c], label=c) for c in POS_CLASSES]
    handles.append(Line2D([], [], color=INK, linewidth=1.3, label="class boundary if every token's bits\n"
                          "grew by one factor (base-bits split)"))
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False, labelcolor=INK2, ncol=2)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.22 * (yhi - ylo))
    rk, rk10 = (lookup(corr, "drift_knowledge", "forgetting", ALL12, "spearman"),
                lookup(corr, "drift_knowledge", "forgetting", NO_MATH, "pearson"))
    rt = lookup(corr, "dbpb_fresh", "forgetting", ALL12, "spearman")
    olmo, small = "OLMo-2 DPO, RLVR", "Qwen2.5 Inst ×3, SimpleRL ×2"
    hide = {m: None for m in ("allenai/OLMo-2-0425-1B-DPO", "allenai/OLMo-2-0425-1B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct",
                              "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
                              "hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo", "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo")}
    right = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": (-7, -3, "right"), "Qwen/Qwen2.5-Math-1.5B": (-7, 0, "right"),
             "allenai/OLMo-2-0425-1B-SFT": (0, 7, "center")}
    scatter_panel(axes[1], table, "drift_knowledge", "forgetting", CLASS_COLOR["knowledge"],
                  "knowledge-token drift, fresh mean (%)", "forgetting (% of base general score)",
                  f"(b) Knowledge drift vs forgetting, n = 12 (H1)\nρ = {rk:.2f} (total Δbpb: {rt:.2f}); "
                  f"r without math line = {rk10:.2f}", {**hide, **right},
                  ((olmo, 11.5, -2.9), (small, -1.0, -4.7)), ymin=-6.0)
    chat = table[table.stage.isin(CHAT_STAGES)]
    rs, rkc = (lookup(corr, "drift_style", "gain_ifeval_chat", CHAT8, "pearson"),
               lookup(corr, "drift_knowledge", "gain_ifeval_chat", CHAT8, "pearson"))
    scatter_panel(axes[2], chat, "drift_style", "gain_ifeval_chat", CLASS_COLOR["style"],
                  "style-token drift, fresh mean (%)", "IFEval gain, chat template (points)",
                  f"(c) Style drift vs IFEval chat gain, n = 8 chat-tuned (H2)\nr = {rs:.2f} "
                  f"(knowledge drift: r = {rkc:.2f})",
                  {"allenai/OLMo-2-0425-1B-Instruct": (-7, 0, "right"), "allenai/OLMo-2-0425-1B-DPO": (-7, 0, "right"),
                   "HuggingFaceTB/SmolLM2-1.7B-Instruct": (6, -9, "left")})
    rc, rc10 = (lookup(corr, "contrast_ks", "forgetting", ALL12, "pearson"),
                lookup(corr, "contrast_ks", "forgetting", NO_MATH, "pearson"))
    rcg = lookup(corr, "contrast_ks", "gain_ifeval_chat", CHAT8, "pearson")
    scatter_panel(axes[3], table, "contrast_ks", "forgetting", INK2,
                  "knowledge drift − style drift, fresh mean (pp)", "forgetting (% of base general score)",
                  f"(d) Knowledge − style contrast vs forgetting (secondary reading)\nr = {rc:.2f}; without math "
                  f"line {rc10:.2f}; vs IFEval chat gain r = {rcg:.2f}", {**hide, **right},
                  ((olmo, -4.0, -2.9), (small, -0.5, -4.7)), ymin=-6.0)
    fig.suptitle("exp05: post-training drift split by token class (12 checkpoints vs family base, RAW per-token bits,"
                 " 3 fresh 2026 corpora)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.93, bottom=0.07)
    return fig


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


def load_classes(out: Path) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """{(tokenizer group, corpus): {pos, freq, tag, lengths}} written by `align`."""
    res = {}
    for group in tokenizer_groups(STAGES):
        for corpus in CORPORA:
            with np.load(out / "classes" / f"{group}__{corpus}.npz") as z:
                res[(group, corpus)] = {k: z[k] for k in ("pos", "freq", "tag", "lengths")}
    return res


def cmd_analyze(out: Path) -> None:
    classes = load_classes(out)
    wsums, counts, tag_sums = {}, {}, {}
    for s in STAGES:
        for corpus in CORPORA:
            cls = classes[(s.tokenizer, corpus)]
            bits = load_token_bits(bits_dir(s.model), s.model, "raw", corpus)
            if bits.size != cls["pos"].size:
                raise ValueError(f"{s.model} on {corpus}: {bits.size} bits vs {cls['pos'].size} classified tokens")
            tag_sums[(s.model, corpus)] = class_sums(bits, cls["tag"], len(UPOS))
            for scheme, names in SCHEMES.items():
                wsums[(s.model, corpus, scheme)] = window_class_sums(bits, cls[scheme], list(cls["lengths"]), len(names))
                counts[(s.model, corpus, scheme)] = np.bincount(cls[scheme].astype(np.intp), minlength=len(names))
    decomp, share_boot = decomposition_table(wsums, counts)
    pred = predictor_table(decomp)
    outc = outcome_table(pd.read_csv(EXP04 / "results" / "scores.csv"))
    repro = reproduction_table(pred, outc)
    table = pred.merge(outc, on="model")
    corr = correlation_table(table)
    tests = rule_table(corr, decomp, pred, share_boot)
    decomp.to_csv(out / "decomposition.csv", index=False, float_format="%.5f")
    corr.to_csv(out / "correlations.csv", index=False, float_format="%.4f")
    (out / "summary.md").write_text(summary_markdown(tests, decomp, pred, outc, corr, repro, pos_share_table(tag_sums)))
    figure(decomp, table, corr).savefig(out / "fig_decomposition.png", dpi=160, facecolor=SURFACE)
    print((out / "summary.md").read_text())


def _dist_version(name: str) -> str:
    from importlib.metadata import version
    return version(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["align", "analyze"])
    ap.add_argument("--results", default=str(HERE / "results"))
    args = ap.parse_args()
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    {"align": cmd_align, "analyze": cmd_analyze}[args.cmd](out)


if __name__ == "__main__":
    main()

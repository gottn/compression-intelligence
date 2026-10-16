"""Vector figures for the technical report (paper/main.tex).

Reads the committed results of exp01–exp06 (CSV tables, plus exp02's per-window bit grids for the
temperature fit), turns them into the plotted numbers with pure functions, checks every plotted number
against the tables printed in the experiments' RESULTS.md (the EXPECTED_* constants below are copied from
those tables, with the rounding they are printed at), and writes four PDF figures.

Run from the repository root (writes paper/figures/fig{1,2,3,4}_*.pdf):
    .venv/bin/python -m paper.figures.make_figures

Pure above the line marked "effects"; file reads and writes only in `read_inputs` and `main`.
The figures are drawn at the report's text width (16 cm) so that LaTeX includes them unscaled and the
10 pt figure text stays 10 pt on the page.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from cilab.calib import T_GRID, recovered_stat
from cilab.pairs import FRESH_CORPORA, PAIRS, model_slug
from cilab.stages import EXP03_FAMILIES, STAGES, family_base
from cilab.stats import bootstrap_ci, paired_pct_change, pearson, resample_indices
from paper.derived import Inputs as DerivedInputs
from paper.derived import contrast_table
from paper.derived import read_inputs as read_derived

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- style (print, one palette)
TEXT_W_IN = 16.0 / 2.54  # \textwidth of main.tex: A4 with 2.5 cm side margins
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated all-pairs)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#efeeea"
KIND_COLOR = {"chat": BLUE, "rl": AQUA, "math": ORANGE}
RECIPE_COLOR = {"tulu": BLUE, "smoltalk": AQUA, "mathcot": ORANGE}
MARK_SHAPE = {"3": "o", "6": "s", "final": "^"}
MARK_SIZE = {"3": 5.5, "6": 5.5, "final": 6.5}  # points; a triangle needs more size to read as large as a circle
RC = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.handlelength": 1.4, "legend.columnspacing": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
}
N_BOOT, SEED = 5000, 0  # exp02's bootstrap (cilab.stats.resample_indices, seed 0)
T1 = T_GRID.index(1.0)

# ----------------------------------------------------------------------------- RESULTS.md values (checks)
# exp03 RESULTS.md, "Results per corpus" (fresh mean, 95% CI); SmolLM2 and Qwen3 from exp04 RESULTS.md's
# per-checkpoint table (their CIs come from exp05 decomposition.csv and are not printed in a RESULTS.md).
EXPECTED_F1_BASE: dict[str, tuple[float, ...]] = {
    "allenai/OLMo-2-0425-1B-SFT": (6.99, 6.61, 7.40),
    "allenai/OLMo-2-0425-1B-DPO": (14.10, 13.52, 14.69),
    "allenai/OLMo-2-0425-1B-Instruct": (14.66, 14.05, 15.28),
    "Qwen/Qwen2.5-0.5B-Instruct": (1.61, 1.49, 1.73),
    "hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo": (0.18, 0.16, 0.21),
    "Qwen/Qwen2.5-1.5B-Instruct": (0.91, 0.81, 1.01),
    "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo": (0.12, 0.09, 0.14),
    "Qwen/Qwen2.5-Math-1.5B": (23.84, 22.46, 25.41),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": (42.95, 41.07, 44.93),
    "Qwen/Qwen2.5-3B-Instruct": (2.08, 1.74, 2.47),
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": (3.57,),
    "Qwen/Qwen3-1.7B": (16.18,),
}
# exp03 RESULTS.md verdict table: steps against the parent stage.
EXPECTED_F1_PARENT: dict[str, tuple[float, float, float]] = {
    "allenai/OLMo-2-0425-1B-DPO": (6.65, 6.38, 6.93),
    "allenai/OLMo-2-0425-1B-Instruct": (0.49, 0.44, 0.55),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": (15.81, 14.35, 17.29),
}
# exp02 RESULTS.md: recovered share in % per fresh corpus (wiki, arxiv, gutenberg) and its fresh mean;
# gap at T = 1 and residual gap at own T* (fresh means, %).
EXPECTED_F2: dict[str, dict[str, tuple[float, ...]]] = {
    "qwen2.5": {"recovered": (3, 42, 1), "mean": (15,), "gap": (1.0,), "residual": (0.9,)},
    "smollm2": {"recovered": (3, 13, 2), "mean": (6,), "gap": (3.7,), "residual": (3.6,)},
    "qwen3": {"recovered": (39, 37, 31), "mean": (35,), "gap": (16.0,), "residual": (10.4,)},
    "qwen2.5-math": {"recovered": (19, 18, 20), "mean": (19,), "gap": (15.4,), "residual": (12.7,)},
}
EXPECTED_F2_QWEN3_CI = ((36, 42), (34, 39), (28, 33))  # exp02 RESULTS.md, per corpus
# exp04 results/summary.md (written by analyze_forgetting.py; RESULTS.md prints the same table with forgetting
# at 1 decimal): (Δbpb fresh %, forgetting %, IFEval chat gain in points).
EXPECTED_F3: dict[str, tuple[float, float, float]] = {
    "allenai/OLMo-2-0425-1B-SFT": (6.99, 0.01, 31.5),
    "allenai/OLMo-2-0425-1B-DPO": (14.10, -1.40, 43.5),
    "allenai/OLMo-2-0425-1B-Instruct": (14.66, -0.88, 50.5),
    "Qwen/Qwen2.5-0.5B-Instruct": (1.61, -0.22, 13.0),
    "hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo": (0.18, -0.15, 5.5),
    "Qwen/Qwen2.5-1.5B-Instruct": (0.91, -1.78, 18.0),
    "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo": (0.12, -1.57, -6.0),
    "Qwen/Qwen2.5-Math-1.5B": (23.84, 13.90, 1.5),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": (42.95, 21.61, 5.5),
    "Qwen/Qwen2.5-3B-Instruct": (2.08, -0.44, 34.0),
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": (3.57, 3.79, 29.5),
    "Qwen/Qwen3-1.7B": (16.18, 4.67, 46.5),
}
EXPECTED_F3_R = {"forgetting": 0.887, "ifeval_chat": 0.89}  # Gate 1 r; ifeval chat r over 8 chat-tuned
# exp06 RESULTS.md, checkpoint table: (recipe, update) -> (Δbpb fresh %, contrast K−S pp, forgetting %).
EXPECTED_F4: dict[tuple[str, int], tuple[float, float, float]] = {
    ("tulu", 1500): (2.02, 1.69, -1.45), ("tulu", 1760): (3.28, 2.47, -0.30),
    ("tulu", 2660): (6.01, 3.37, -1.74), ("tulu", 3489): (7.11, 4.35, -0.82),
    ("smoltalk", 920): (3.02, 1.55, -0.97), ("smoltalk", 1500): (3.36, 2.17, -0.83),
    ("smoltalk", 2500): (6.00, 2.38, -0.89), ("smoltalk", 3487): (8.97, 3.41, -0.65),
    ("mathcot", 1500): (2.50, 2.10, 4.24), ("mathcot", 1660): (3.06, 2.86, 4.07),
    ("mathcot", 2780): (6.06, 3.41, 4.10), ("mathcot", 3464): (7.14, 4.48, 4.41),
}
EXPECTED_F4_R = {"marks": 0.43, "all": 0.29}  # exp06 H1 r(contrast_ks, forgetting), 6 marks and all 12


def check_close(what: str, got: float, want: float, tol: float) -> None:
    """Raise AssertionError unless |got − want| ≤ tol (survives `python -O`, unlike `assert`)."""
    if not abs(float(got) - float(want)) <= tol:
        raise AssertionError(f"{what}: computed {got:.6g}, RESULTS.md says {want} (tolerance {tol})")


TOL2 = 0.005 + 1e-9  # half a unit of a value printed with 2 decimals in RESULTS.md
TOL1 = 0.05 + 1e-9  # ... with 1 decimal


# ----------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Inputs:
    exp01_bpb: pd.DataFrame
    exp02_runs: pd.DataFrame
    exp02_windows: pd.DataFrame
    exp02_grids: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]  # (model, corpus) -> (T grid, T* bits)
    exp03_deltas: pd.DataFrame
    exp04_scores: pd.DataFrame
    exp05_decomp: pd.DataFrame
    exp06_ckpts: pd.DataFrame
    exp06_traj: pd.DataFrame


# ----------------------------------------------------------------------------- F1: staged drift
SHORT = {
    "allenai/OLMo-2-0425-1B-SFT": "OLMo-2-1B SFT", "allenai/OLMo-2-0425-1B-DPO": "OLMo-2-1B DPO",
    "allenai/OLMo-2-0425-1B-Instruct": "OLMo-2-1B RLVR (Instruct)",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B Instruct",
    "hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo": "Qwen2.5-0.5B SimpleRL-Zoo",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B Instruct",
    "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo": "Qwen2.5-1.5B SimpleRL-Zoo",
    "Qwen/Qwen2.5-Math-1.5B": "Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "R1-Distill-Qwen-1.5B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2-1.7B Instruct",
    "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
}


def stage_kind(stage: str) -> str:
    """Colour class of the whole path from the family base to a checkpoint: 'rl' when the path is RL only
    (SimpleRL-Zoo), 'math' for the math line, else 'chat' (the path includes SFT or DPO)."""
    return "rl" if stage == "rl-zero" else "math" if stage in ("math-cpt", "r1-distill") else "chat"


def step_kind(stage: str) -> str:
    """Colour class of the single step from the parent stage: 'rl' for RLVR, 'math' for R1 distillation, else 'chat'."""
    return "rl" if stage == "rlvr" else "math" if stage == "r1-distill" else "chat"


def exp01_fresh_delta(bpb: pd.DataFrame, base: str, post: str) -> tuple[float, list[float]]:
    """(fresh mean, per-corpus) % change of total RAW bits, post vs base, from exp01's bpb.csv."""
    raw = bpb[bpb.condition == "raw"].set_index(["model", "corpus"]).total_bits
    per = [100.0 * (raw[(post, c)] - raw[(base, c)]) / raw[(base, c)] for c in FRESH_CORPORA]
    return float(np.mean(per)), [float(p) for p in per]


def f1_data(inp: Inputs) -> pd.DataFrame:
    """One row per post-trained checkpoint (cilab.stages order): Δ vs family base with its 95% CI, the
    step vs the parent stage where the parent is not the base, and the lowest per-corpus CI bound."""
    d, dec = inp.exp03_deltas, inp.exp05_decomp
    rows = []
    for s in STAGES:
        if s.parent is None:
            continue
        base = family_base(s.family)
        row: dict[str, object] = {"model": s.model, "label": SHORT[s.model], "family": s.family,
                                  "stage": s.stage, "kind": stage_kind(s.stage), "parent_kind": step_kind(s.stage)}
        if s.family in EXP03_FAMILIES:
            vb = d[(d.model == s.model) & (d.kind == "vs_base")].set_index("corpus")
            row.update(delta=vb.delta_pct["fresh_mean"], lo=vb.lo["fresh_mean"], hi=vb.hi["fresh_mean"],
                       enwik8=vb.delta_pct["enwik8_ctrl"], corpus_lo_min=vb.lo[list(FRESH_CORPORA)].min())
            vp = d[(d.model == s.model) & (d.kind == "vs_parent")].set_index("corpus")
            if len(vp):
                row.update(parent_delta=vp.delta_pct["fresh_mean"], parent_lo=vp.lo["fresh_mean"],
                           parent_hi=vp.hi["fresh_mean"], parent_corpus_lo_min=vp.lo[list(FRESH_CORPORA)].min())
        else:  # exp01 pairs: point from exp01's bpb.csv, CI from exp05's stratified window bootstrap
            point, _ = exp01_fresh_delta(inp.exp01_bpb, base, s.model)
            ci = dec[(dec.model == s.model) & (dec.scheme == "pos") & (dec["class"] == "all")].set_index("corpus")
            check_close(f"F1 {s.model}: exp01 point vs exp05 drift", point, ci.drift_pct["fresh_mean"], 1e-3)
            row.update(delta=point, lo=ci.drift_lo["fresh_mean"], hi=ci.drift_hi["fresh_mean"],
                       enwik8=ci.drift_pct["enwik8_ctrl"], corpus_lo_min=ci.drift_lo[list(FRESH_CORPORA)].min())
        rows.append(row)
    out = pd.DataFrame(rows)
    for r in out.itertuples():
        want = EXPECTED_F1_BASE[r.model]
        for got, w in zip((r.delta, r.lo, r.hi), want, strict=False):
            check_close(f"F1 {r.model} vs base", got, w, TOL2)
        if r.model in EXPECTED_F1_PARENT:
            for got, w in zip((r.parent_delta, r.parent_lo, r.parent_hi), EXPECTED_F1_PARENT[r.model], strict=True):
                check_close(f"F1 {r.model} vs parent", got, w, TOL2)
    return out


def f1_counts(t: pd.DataFrame) -> dict[str, int]:
    """Comparisons in F1 and how many have every fresh-corpus 95% CI above zero."""
    n_parent = int(t.parent_delta.notna().sum())
    pos_base = int((t.corpus_lo_min > 0).sum())
    pos_parent = int((t.parent_corpus_lo_min > 0).sum())
    return {"comparisons": len(t) + n_parent, "all_corpora_above_zero": pos_base + pos_parent}


def f1_figure(t: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(TEXT_W_IN, 2.95), layout="constrained")
    y, ys, prev = 0.0, [], None
    for r in t.itertuples():
        if prev is not None and r.family != prev:
            ax.axhline(y + 0.35, color=GRID, linewidth=0.5, zorder=0)
            y += 0.7
        ys.append(y)
        prev = r.family
        y += 1.0
    ys = np.array(ys)
    for yy, r in zip(ys, t.itertuples(), strict=True):
        c = KIND_COLOR[r.kind]
        ax.plot([r.lo, r.hi], [yy, yy], color=c, linewidth=1.3, zorder=2)
        ax.plot([r.delta], [yy], "o", color=c, markersize=5, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        if pd.notna(r.parent_delta):
            cp = KIND_COLOR[r.parent_kind]
            ax.plot([r.parent_lo, r.parent_hi], [yy, yy], color=cp, linewidth=0.8, zorder=2)
            ax.plot([r.parent_delta], [yy], "o", markerfacecolor="white", markeredgecolor=cp, markersize=5,
                    markeredgewidth=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.07, 60)
    ticks = [0.1, 0.3, 1, 3, 10, 30]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for v in ticks:
        ax.axvline(v, color=GRID, linewidth=0.5, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels(t.label)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(ys.max() + 0.7, ys.min() - 0.7)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"increase in code length on fresh text, $\Delta$ (%, log scale)")
    handles = [Line2D([], [], color=KIND_COLOR[k], marker="o", markersize=5, markeredgecolor="white",
                      linewidth=1.3, label=lab)
               for k, lab in (("chat", "includes SFT or DPO"), ("rl", "RL only"), ("math", "math line"))]
    handles.append(Line2D([], [], color=INK2, marker="o", markerfacecolor="white", markersize=5, linewidth=0.8,
                          label="step against the parent stage"))
    fig.legend(handles=handles, loc="outside upper center", ncol=4, handletextpad=0.4)
    return fig


# ----------------------------------------------------------------------------- F2: temperature
PAIR_LABEL = {"qwen2.5": "Qwen2.5-1.5B → Instruct", "smollm2": "SmolLM2-1.7B → Instruct",
              "qwen3": "Qwen3-1.7B-Base → Qwen3-1.7B", "qwen2.5-math": "Qwen2.5-Math-1.5B → R1-Distill"}
PAIR_KIND = {"qwen2.5": "chat", "smollm2": "chat", "qwen3": "chat", "qwen2.5-math": "math"}
F2_ORDER = ("qwen2.5", "smollm2", "qwen3", "qwen2.5-math")


def f2_data(inp: Inputs) -> pd.DataFrame:
    """Per (pair, fresh corpus), odd windows: gap at T = 1 (%), share of it removed by the post model's
    own T* with its 95% window-bootstrap CI, and the residual gap with both models at their own T* (%).
    The same statistic and resamples as exp02's analyze_temperature.py."""
    g, w = inp.exp02_grids, inp.exp02_windows
    rows = []
    for pair, base, post in PAIRS:
        for corpus in FRESH_CORPORA:
            (gb, tb), (gp, tp) = g[(base, corpus)], g[(post, corpus)]
            wb = w[(w.model == base) & (w.corpus == corpus)].sort_values("window")[["tokens", "bytes"]].to_numpy()
            wp = w[(w.model == post) & (w.corpus == corpus)].sort_values("window")[["tokens", "bytes"]].to_numpy()
            if not np.array_equal(wb, wp):
                raise ValueError(f"{pair} {corpus}: base and post windows differ")
            odd = np.arange(1, len(gb), 2)
            b1, p1 = gb[odd, T1], gp[odd, T1]
            idx = resample_indices(odd.size, N_BOOT, SEED)
            rec = bootstrap_ci(recovered_stat(b1, p1, tp[odd]), idx)
            res = bootstrap_ci(paired_pct_change(tb[odd], tp[odd]), idx)
            rows.append({"pair": pair, "corpus": corpus, "gap": 100 * (p1.sum() - b1.sum()) / b1.sum(),
                         "recovered": 100 * rec[0], "rec_lo": 100 * rec[1], "rec_hi": 100 * rec[2],
                         "residual": res[0], "res_lo": res[1], "res_hi": res[2]})
    out = pd.DataFrame(rows)
    for pair, exp in EXPECTED_F2.items():
        sub = out[out.pair == pair].set_index("corpus").loc[list(FRESH_CORPORA)]
        for c, got, want in zip(FRESH_CORPORA, sub.recovered, exp["recovered"], strict=True):
            check_close(f"F2 {pair} {c} recovered", got, want, 0.5)
        check_close(f"F2 {pair} mean recovered", sub.recovered.mean(), exp["mean"][0], 0.5)
        check_close(f"F2 {pair} gap", sub.gap.mean(), exp["gap"][0], TOL1)
        check_close(f"F2 {pair} residual", sub.residual.mean(), exp["residual"][0], TOL1)
    q3 = out[out.pair == "qwen3"].set_index("corpus").loc[list(FRESH_CORPORA)]
    for (lo, hi), a, b in zip(EXPECTED_F2_QWEN3_CI, q3.rec_lo, q3.rec_hi, strict=True):
        check_close("F2 qwen3 CI lo", a, lo, 0.5)
        check_close("F2 qwen3 CI hi", b, hi, 0.5)
    return out


def f2_means(t: pd.DataFrame) -> pd.DataFrame:
    return t.groupby("pair", sort=False)[["gap", "recovered", "residual"]].mean().loc[list(F2_ORDER)]


def f2_figure(t: pd.DataFrame) -> Figure:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_W_IN, 1.85), layout="constrained", sharey=True,
                                  gridspec_kw={"width_ratios": [1.45, 1]})
    means = f2_means(t)
    shapes = dict(zip(FRESH_CORPORA, ("o", "s", "^"), strict=True))
    offs = dict(zip(FRESH_CORPORA, (-0.2, 0.0, 0.2), strict=True))
    for i, pair in enumerate(F2_ORDER):
        c = KIND_COLOR[PAIR_KIND[pair]]
        for r in t[t.pair == pair].itertuples():
            yy = i + offs[r.corpus]
            ax.plot([r.rec_lo, r.rec_hi], [yy, yy], color=c, linewidth=1.0, zorder=2)
            ax.plot([r.recovered], [yy], shapes[r.corpus], color=c, markersize=4, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=3)
        ax.plot([means.recovered[pair]] * 2, [i - 0.36, i + 0.36], color=INK, linewidth=1.2, zorder=4)
        m = means.loc[pair]
        ax2.plot([m.residual, m.gap], [i, i], color=c, linewidth=1.3, zorder=2)
        ax2.plot([m.gap], [i], "o", markerfacecolor="white", markeredgecolor=c, markersize=5, markeredgewidth=1.0,
                 zorder=3)
        ax2.plot([m.residual], [i], "o", color=c, markersize=5, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    for v in (30, 70):
        ax.axvline(v, color=MUTED, linewidth=0.7, zorder=1)
        ax.text(v + 1, 3.55, f"{v}%", color=INK2, va="center", ha="left")
    for v in (0, 10, 20, 40, 50, 60):
        ax.axvline(v, color=GRID, linewidth=0.5, zorder=0)
    for v in (0, 5, 10, 15):
        ax2.axvline(v, color=GRID, linewidth=0.5, zorder=0)
    ax.set_xlim(0, 75)
    ax2.set_xlim(0, 17.5)
    ax.set_yticks(range(len(F2_ORDER)))
    ax.set_yticklabels([PAIR_LABEL[p] for p in F2_ORDER])
    ax.set_ylim(3.75, -0.5)
    for a in (ax, ax2):
        a.tick_params(axis="y", length=0)
        a.spines["left"].set_visible(False)
    ax.set_xlabel("share of the increase removed (%)")
    ax2.set_xlabel(r"$\Delta$ on fresh text (%)")
    ax2.set_xticks([0, 5, 10, 15])
    ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, ha="left", va="bottom")
    ax2.text(0.0, 1.0, "(b)", transform=ax2.transAxes, ha="left", va="bottom")
    corpus_handles = [Line2D([], [], color=INK2, marker=shapes[c], linestyle="", markersize=4, label=c.replace("2026", ""))
                      for c in FRESH_CORPORA]
    corpus_handles.append(Line2D([], [], color=INK, linewidth=1.2, label="mean"))
    ax.legend(handles=corpus_handles, loc="center left", bbox_to_anchor=(0.6, 0.36), ncol=1, handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.25)
    ax2.legend(handles=[Line2D([], [], color=INK2, marker="o", markerfacecolor="white", linestyle="", markersize=5,
                               label="both at $T = 1$"),
                        Line2D([], [], color=INK2, marker="o", linestyle="", markersize=5,
                               label="each at its own $T^*$")],
               loc="upper right", handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
    return fig


# ----------------------------------------------------------------------------- F3: benchmarks (exp04)
F3_LABEL = {"Qwen/Qwen2.5-Math-1.5B": ("Qwen2.5-Math-1.5B", (-6, 0), "right"),
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": ("R1-Distill-Qwen-1.5B", (-6, 0), "right"),
            "Qwen/Qwen3-1.7B": ("Qwen3-1.7B", (5, -6), "left"),
            "HuggingFaceTB/SmolLM2-1.7B-Instruct": ("SmolLM2-1.7B Instruct", (-3, 10), "left"),
            "allenai/OLMo-2-0425-1B-Instruct": ("OLMo-2 DPO, RLVR", (5, -7), "left")}
F3B_LABEL = {"allenai/OLMo-2-0425-1B-Instruct": ("OLMo-2 RLVR", (-5, 2), "right"),
             "Qwen/Qwen3-1.7B": ("Qwen3", (5, -2), "left"),
             "allenai/OLMo-2-0425-1B-DPO": ("OLMo-2 DPO", (-5, -1), "right"),
             "allenai/OLMo-2-0425-1B-SFT": ("OLMo-2 SFT", (5, -3), "left"),
             "Qwen/Qwen2.5-0.5B-Instruct": ("Qwen2.5-0.5B Instruct", (5, -3), "left")}
CHAT_STAGES = ("sft", "dpo", "rlvr", "instruct")


def f3_data(inp: Inputs) -> tuple[pd.DataFrame, dict[str, float]]:
    """The 12 post-trained checkpoints of exp04 with drift, forgetting, its noise floor and the IFEval
    chat gain; plus Pearson r for forgetting (all 12) and for the IFEval chat gain (8 chat-tuned)."""
    s = inp.exp04_scores
    t = s[s.stage != "base"][["model", "stage", "dbpb_fresh", "forgetting", "floor_pct", "within_floor",
                              "gain_ifeval_chat"]].reset_index(drop=True)
    t = t.assign(kind=t.stage.map(stage_kind), chat=t.stage.isin(CHAT_STAGES))
    for r in t.itertuples():
        d, f, g = EXPECTED_F3[r.model]
        check_close(f"F3 {r.model} drift", r.dbpb_fresh, d, TOL2)
        check_close(f"F3 {r.model} forgetting", r.forgetting, f, TOL2)
        check_close(f"F3 {r.model} ifeval chat gain", r.gain_ifeval_chat, g, TOL1)
    ch = t[t.chat]
    stats = {"r_forgetting": pearson(t.dbpb_fresh, t.forgetting),
             "r_ifeval_chat": pearson(ch.dbpb_fresh, ch.gain_ifeval_chat),
             "floor_median": float(t.floor_pct.median()), "n_within": int(t.within_floor.sum()),
             "n_chat": len(ch)}
    check_close("F3 r(forgetting)", stats["r_forgetting"], EXPECTED_F3_R["forgetting"], 5e-4)
    check_close("F3 r(ifeval chat)", stats["r_ifeval_chat"], EXPECTED_F3_R["ifeval_chat"], 5e-3)
    return t, stats


def dot(ax: plt.Axes, x: float, y: float, color: str, marker: str = "o", size: float = 5) -> None:
    ax.plot([x], [y], marker, color=color, markersize=size, markeredgecolor="white", markeredgewidth=0.8, zorder=3)


def f3_figure(t: pd.DataFrame, stats: dict[str, float]) -> Figure:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_W_IN, 2.3), layout="constrained",
                                  gridspec_kw={"width_ratios": [1.3, 1]})
    fl = stats["floor_median"]
    ax.axhspan(-fl, fl, color=BAND, zorder=0, linewidth=0)
    ax.text(44.5, -fl + 0.4, "noise floor", color=INK2, ha="right", va="bottom")
    ax.axhline(0, color=AXIS, linewidth=0.6, zorder=1)
    for r in t.itertuples():
        dot(ax, r.dbpb_fresh, r.forgetting, KIND_COLOR[r.kind])
        if r.model in F3_LABEL:
            text, off, ha = F3_LABEL[r.model]
            ax.annotate(text, (r.dbpb_fresh, r.forgetting), xytext=off, textcoords="offset points", ha=ha,
                        va="center", color=INK2)
    ax.set_xlim(-1, 46)
    ax.set_ylim(-4, 24)
    ax.set_xlabel(r"$\Delta$ on fresh text (%)")
    ax.set_ylabel("forgetting (% of base score)")
    ch = t[t.chat]
    for r in ch.itertuples():
        dot(ax2, r.dbpb_fresh, r.gain_ifeval_chat, KIND_COLOR[r.kind])
        if r.model in F3B_LABEL:
            text, off, ha = F3B_LABEL[r.model]
            ax2.annotate(text, (r.dbpb_fresh, r.gain_ifeval_chat), xytext=off, textcoords="offset points", ha=ha,
                         va="center", color=INK2)
    ax2.set_xlim(-0.5, 18)
    ax2.set_ylim(0, 58)
    ax2.set_xlabel(r"$\Delta$ on fresh text (%)")
    ax2.set_ylabel("IFEval gain, chat template (points)")
    for a, ys in ((ax, (0, 10, 20)), (ax2, (0, 20, 40))):
        for v in ys:
            a.axhline(v, color=GRID, linewidth=0.5, zorder=0)
    ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, ha="left", va="bottom")
    ax2.text(0.0, 1.0, "(b)", transform=ax2.transAxes, ha="left", va="bottom")
    handles = [Line2D([], [], color=KIND_COLOR[k], marker="o", linestyle="", markersize=5, markeredgecolor="white",
                      label=lab)
               for k, lab in (("chat", "path from base includes SFT or DPO"), ("rl", "RL only"),
                              ("math", "math line"))]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, handletextpad=0.3)
    return fig


# ----------------------------------------------------------------------------- F4: matched recipes (exp06)
def f4_data(inp: Inputs) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """exp06 drift trajectories (passes over the SFT subset, Δbpb fresh) and the 12 benchmarked checkpoints
    with contrast K − S and forgetting; plus H1's r over the 6 marks and all 12, and the forgetting ranges."""
    traj = inp.exp06_traj.sort_values(["recipe", "step"])[["recipe", "stage", "step", "epochs", "dbpb_fresh"]]
    ck = inp.exp06_ckpts.copy()
    ck["shape"] = ck.marks.astype(str)
    for r in ck.itertuples():
        d, k, f = EXPECTED_F4[(r.recipe, int(r.step))]
        check_close(f"F4 {r.recipe}@{r.step} drift", r.dbpb_fresh, d, TOL2)
        check_close(f"F4 {r.recipe}@{r.step} contrast", r.contrast_ks, k, TOL2)
        check_close(f"F4 {r.recipe}@{r.step} forgetting", r.forgetting, f, TOL2)
    marks = ck[ck.is_mark]
    chat = ck[ck.recipe != "mathcot"]
    math = ck[ck.recipe == "mathcot"]
    stats = {"r_marks": pearson(marks.contrast_ks, marks.forgetting), "r_all": pearson(ck.contrast_ks, ck.forgetting),
             "r_drift_all": pearson(ck.dbpb_fresh, ck.forgetting), "floor_median": float(ck.floor_pct.median()),
             "math_forget_min": float(math.forgetting.min()), "math_forget_max": float(math.forgetting.max()),
             "math_drift_min": float(math.dbpb_fresh.min()), "math_drift_max": float(math.dbpb_fresh.max()),
             "chat_forget_max": float(chat.forgetting.max()), "chat_drift_max": float(chat.dbpb_fresh.max())}
    check_close("F4 H1 r (marks)", stats["r_marks"], EXPECTED_F4_R["marks"], TOL2)
    check_close("F4 H1 r (all 12)", stats["r_all"], EXPECTED_F4_R["all"], TOL2)
    return traj, ck, stats


def f4_figure(traj: pd.DataFrame, ck: pd.DataFrame, stats: dict[str, float], ci: pd.DataFrame) -> Figure:
    """ci: 95% window-bootstrap interval of K − S per checkpoint (paper.derived.contrast_table, contrast 'ks')."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_W_IN, 2.35), layout="constrained",
                                  gridspec_kw={"width_ratios": [1.3, 1]})
    for v in (1, 2, 3, 4):
        ax.axvline(v, color=GRID, linewidth=0.5, zorder=0)
    for v in (3, 6):
        ax.axhline(v, color=MUTED, linewidth=0.6, zorder=1)
        ax.text(0.04, v + 0.15, f"{v}% mark", color=INK2, ha="left", va="bottom")
    for recipe in RECIPE_COLOR:
        tr = traj[traj.recipe == recipe]
        ax.plot(tr.epochs, tr.dbpb_fresh, color=RECIPE_COLOR[recipe], linewidth=1.1, zorder=2)
    for r in ck.itertuples():  # open marks in (a), where tulu and mathcot reach 6% at the same place
        ax.plot([r.epochs], [r.dbpb_fresh], MARK_SHAPE[r.shape], markerfacecolor="white",
                markeredgecolor=RECIPE_COLOR[r.recipe], markersize=MARK_SIZE[r.shape], markeredgewidth=1.0, zorder=3)
        lo, hi = ci.loc[(r.recipe, int(r.step)), ["lo", "hi"]]
        ax2.plot([lo, hi], [r.forgetting, r.forgetting], color=RECIPE_COLOR[r.recipe], linewidth=0.9, zorder=2)
        dot(ax2, r.contrast_ks, r.forgetting, RECIPE_COLOR[r.recipe], MARK_SHAPE[r.shape], MARK_SIZE[r.shape])
    ax.set_xlim(0, 4.15)
    ax.set_ylim(-0.3, 9.6)
    ax.set_xlabel("passes over the 45M-token SFT subset")
    ax.set_ylabel(r"$\Delta$ on fresh text (%)")
    fl = stats["floor_median"]
    ax2.axhspan(-fl, fl, color=BAND, zorder=0, linewidth=0)
    ax2.text(5.35, -fl + 0.15, "noise floor", color=INK2, ha="right", va="bottom")
    ax2.axhline(0, color=AXIS, linewidth=0.6, zorder=1)
    ax2.set_xlim(0.9, 5.4)
    ax2.set_ylim(-2.9, 5.6)
    ax2.set_xlabel("contrast K − S (percentage points)")
    ax2.set_ylabel("forgetting (% of base score)")
    ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, ha="left", va="bottom")
    ax2.text(0.0, 1.0, "(b)", transform=ax2.transAxes, ha="left", va="bottom")
    handles = [Line2D([], [], color=RECIPE_COLOR[k], linewidth=1.1, label=k) for k in RECIPE_COLOR]
    handles += [Line2D([], [], color=INK2, marker=MARK_SHAPE[k], linestyle="", markersize=5, label=lab)
                for k, lab in (("3", "3% mark"), ("6", "6% mark"), ("final", "final of a stage"))]
    fig.legend(handles=handles, loc="outside upper center", ncol=6, handletextpad=0.4)
    return fig


# ----------------------------------------------------------------------------- effects
def read_inputs(repo: Path) -> Inputs:
    """Read every table and array the figures use (the only reads in this module)."""
    e2 = repo / "exp02_temperature" / "results"
    runs = pd.read_csv(e2 / "runs.csv")
    grids = {(m, c): (np.load(e2 / "grid" / f"{model_slug(m)}__{c}.npy"),
                      np.load(e2 / "grid" / f"{model_slug(m)}__{c}.tstar.npy"))
             for m, c in sorted(set(zip(runs.model, runs.corpus, strict=True)))}
    e6 = repo / "exp06_matched_drift" / "results"
    return Inputs(
        exp01_bpb=pd.read_csv(repo / "exp01_posttraining_bpc" / "results" / "bpb.csv"),
        exp02_runs=runs, exp02_windows=pd.read_csv(e2 / "windows.csv"), exp02_grids=grids,
        exp03_deltas=pd.read_csv(repo / "exp03_staged_drift" / "results" / "stage_deltas.csv"),
        exp04_scores=pd.read_csv(repo / "exp04_benchmarks" / "results" / "scores.csv"),
        exp05_decomp=pd.read_csv(repo / "exp05_decomposition" / "results" / "decomposition.csv"),
        exp06_ckpts=pd.read_csv(e6 / "checkpoints.csv"), exp06_traj=pd.read_csv(e6 / "trajectories.csv"))


def contrast_cis(derived_inputs: DerivedInputs) -> pd.DataFrame:
    """K − S intervals per (recipe, update), from paper.derived."""
    t, _ = contrast_table(derived_inputs)
    return t[t.contrast == "ks"].set_index(["recipe", "step"])


def save(fig: Figure, path: Path) -> None:
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(RC)
    inp = read_inputs(REPO)
    f1 = f1_data(inp)
    f2 = f2_data(inp)
    f3, s3 = f3_data(inp)
    traj, ck, s4 = f4_data(inp)
    ci = contrast_cis(read_derived(REPO))
    save(f1_figure(f1), OUT / "fig1_staged_drift.pdf")
    save(f2_figure(f2), OUT / "fig2_temperature.pdf")
    save(f3_figure(f3, s3), OUT / "fig3_benchmarks.pdf")
    save(f4_figure(traj, ck, s4, ci), OUT / "fig4_matched.pdf")
    print("F1", f1_counts(f1))
    print("F2", f2_means(f2).round(2).to_dict("index"))
    print("F3", {k: round(v, 4) for k, v in s3.items()})
    print("F4", {k: round(v, 4) for k, v in s4.items()})


if __name__ == "__main__":
    main()

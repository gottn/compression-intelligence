"""exp03 analysis: Δbpb of every post-training stage vs its family base and vs its parent, with paired
window-bootstrap 95% CIs; the pre-registered expectations; the exp01 reproduction check; the figure.

    .venv/bin/python exp03_staged_drift/analyze_stages.py [--results R] [--exp01-results E]

Reads   results/{bpb.csv, windows.csv, tokens/*.npy, tokcheck.json, model_checks.jsonl} (score_stages.py, AAU)
        ../exp01_posttraining_bpc/results/{bpb.csv, tokens/*.npy} (Mac, MPS) for the reproduction check
Writes  results/stage_deltas.csv, results/summary.md, results/fig_stages.png
Pure functions above the effects marker; file I/O only in load_inputs() and main().

Δbpb% = 100 (Σ bits_model − Σ bits_ref) / Σ bits_ref over the same windows (identical token ids within a
family, asserted per window), which equals the bpb change because the bytes are the same text.
CIs: 5000 paired resamples of windows per corpus, seed SEED + corpus index (so every comparison on a
corpus uses the same resamples); the mean over the 3 fresh corpora resamples within each corpus.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cilab`

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from cilab.io import load_token_bits
from cilab.pairs import CORPORA, FRESH_CORPORA
from cilab.stages import FAMILIES, RL_STAGES, STAGES, comparisons, family_base, stage_of
from cilab.stats import bootstrap_ci, paired_pct_change, resample_indices, stratified_mean_ci, window_totals

N_BOOT, SEED = 5000, 0
FRESH = "fresh_mean"
REPRO_TOL_BPB = 1e-3
OLMO = "allenai/OLMo-2-0425-1B"
EXPECTATIONS: tuple[tuple[str, str, str, float, float], ...] = (  # (id, model, reference, lo %, hi %)
    ("E1", f"{OLMO}-SFT", OLMO, 3.0, 8.0),
    ("E2", f"{OLMO}-DPO", f"{OLMO}-SFT", 0.0, 2.0),
    ("E3", f"{OLMO}-Instruct", f"{OLMO}-DPO", -1.0, 1.0),
    ("E4", "hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo", "Qwen/Qwen2.5-0.5B", 0.5, 3.0),
    ("E5", "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo", "Qwen/Qwen2.5-1.5B", 0.5, 3.0),
)
STAGE_LABEL = {"sft": "SFT", "dpo": "DPO", "rlvr": "RLVR\n(Instruct)", "instruct": "Instruct",
               "rl-zero": "SimpleRL-Zoo\n(GRPO from base)", "math-cpt": "Math\n(cont. pretrain)",
               "r1-distill": "R1-Distill"}
FAMILY_LABEL = {"olmo2-1b": "OLMo-2-0425-1B", "qwen2.5-0.5b": "Qwen2.5-0.5B", "qwen2.5-1.5b": "Qwen2.5-1.5B",
                "qwen2.5-3b": "Qwen2.5-3B"}
OTHER_C, RL_C = "#2a78d6", "#eb6834"  # dataviz reference palette slots 1, 2 (as in exp01/exp02)
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"

Key = tuple[str, str]  # (model, corpus)


# ----------------------------------------------------------------------------- pure
def short(model: str) -> str:
    return model.split("/")[-1]


def corpus_seed(corpus: str) -> int:
    return SEED + CORPORA.index(corpus)


def delta_table(wbits: Mapping[Key, np.ndarray], lengths: Mapping[Key, tuple[int, ...]],
                comps: list[tuple[str, str, str]], corpora: tuple[str, ...] = CORPORA) -> pd.DataFrame:
    """One row per (model, reference, corpus) plus a fresh-mean row per comparison: Δbpb % with 95% CI.

    wbits: per-window bit totals; lengths: per-window token counts (must match between model and ref).
    Comparisons with a missing (model, corpus) are skipped; the fresh mean needs all 3 fresh corpora."""
    rows = []
    for model, ref, kind in comps:
        st = stage_of(model)
        meta = {"model": model, "reference": ref, "kind": kind, "family": st.family, "stage": st.stage}
        fresh_stats, fresh_idx = [], []
        for c in corpora:
            if (model, c) not in wbits or (ref, c) not in wbits:
                continue
            if lengths[(model, c)] != lengths[(ref, c)]:
                raise ValueError(f"{model} vs {ref} on {c}: windows differ")
            stat = paired_pct_change(wbits[(ref, c)], wbits[(model, c)])
            idx = resample_indices(len(lengths[(model, c)]), N_BOOT, corpus_seed(c))
            point, lo, hi = bootstrap_ci(stat, idx)
            rows.append({**meta, "corpus": c, "n_windows": idx.shape[1], "delta_pct": point, "lo": lo, "hi": hi})
            if c in FRESH_CORPORA:
                fresh_stats.append(stat)
                fresh_idx.append(idx)
        if len(fresh_stats) == len(FRESH_CORPORA):
            point, lo, hi = stratified_mean_ci(fresh_stats, fresh_idx)
            rows.append({**meta, "corpus": FRESH, "n_windows": sum(i.shape[1] for i in fresh_idx),
                         "delta_pct": point, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def step_rows(deltas: pd.DataFrame) -> pd.DataFrame:
    """Each stage against its own parent (its predecessor in the recipe)."""
    parent = {s.model: s.parent for s in STAGES}
    return deltas[deltas.reference == deltas.model.map(parent)].reset_index(drop=True)


def lookup(deltas: pd.DataFrame, model: str, ref: str, corpus: str = FRESH) -> pd.Series | None:
    hit = deltas[(deltas.model == model) & (deltas.reference == ref) & (deltas.corpus == corpus)]
    return hit.iloc[0] if len(hit) else None


def expectation_table(deltas: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered expectations: observed fresh-mean Δbpb, CI and whether the point is in range.
    E1 also needs SFT to be the largest |step| of the OLMo-2 chain."""
    steps = step_rows(deltas)
    olmo = steps[(steps.family == "olmo2-1b") & (steps.corpus == FRESH)]
    rows = []
    for eid, model, ref, lo, hi in EXPECTATIONS:
        r = lookup(deltas, model, ref)
        if r is None:
            rows.append({"id": eid, "model": model, "reference": ref, "range": f"{lo:+g} to {hi:+g}%", "met": "n/a"})
            continue
        met = lo <= r.delta_pct <= hi
        note = ""
        if eid == "E1" and len(olmo) == 3:
            largest = olmo.loc[olmo.delta_pct.abs().idxmax(), "model"] == model
            met, note = met and largest, f"largest OLMo-2 step: {'yes' if largest else 'no'}"
        rows.append({"id": eid, "model": model, "reference": ref, "range": f"{lo:+g} to {hi:+g}%",
                     "delta_pct": r.delta_pct, "lo": r.lo, "hi": r.hi, "met": "met" if met else "not met",
                     "note": note})
    return pd.DataFrame(rows)


def rl_headline(deltas: pd.DataFrame) -> pd.DataFrame:
    """RL stages vs their predecessor (fresh mean and per corpus), flagged when the whole CI is below 0."""
    steps = step_rows(deltas)
    rl = steps[steps.stage.isin(RL_STAGES)].copy()
    rl["below_predecessor"] = rl.hi < 0
    return rl.reset_index(drop=True)


def model_bpb_table(bpb: pd.DataFrame) -> pd.DataFrame:
    """bpb per model (rows, STAGES order) and corpus (columns), plus the mean of the 3 fresh corpora."""
    t = bpb.pivot(index="model", columns="corpus", values="bpb")
    t = t.reindex([s.model for s in STAGES if s.model in t.index])[[c for c in CORPORA if c in t.columns]]
    t[FRESH] = t[[c for c in FRESH_CORPORA if c in t.columns]].mean(axis=1)
    return t


def reproduction_table(bpb: pd.DataFrame, exp01_bpb: pd.DataFrame,
                       tokens: Mapping[Key, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """exp03 (AAU) vs exp01 (Mac) RAW bpb for models scored in both; tokens holds (exp03, exp01) per-token bits."""
    e1 = exp01_bpb[exp01_bpb.condition == "raw"].drop_duplicates(["model", "corpus"], keep="last")
    e1 = e1.set_index(["model", "corpus"])
    rows = []
    for r in bpb.itertuples():
        if (r.model, r.corpus) not in e1.index:
            continue
        old = e1.loc[(r.model, r.corpus)]
        a, b = tokens[(r.model, r.corpus)]
        rows.append({"model": r.model, "corpus": r.corpus, "bpb_aau": r.bpb, "bpb_mac": float(old.bpb),
                     "diff_bpb": r.bpb - float(old.bpb), "total_bits_rel": (r.total_bits - old.total_bits) / old.total_bits,
                     "max_token_abs_bits": float(np.abs(a - b).max()) if a.shape == b.shape else float("nan"),
                     "within_tol": abs(r.bpb - float(old.bpb)) <= REPRO_TOL_BPB})
    return pd.DataFrame(rows)


def fmt_ci(r: pd.Series | object) -> str:
    return f"{r.delta_pct:+.2f}% [{r.lo:+.2f}, {r.hi:+.2f}]"


def summary_markdown(deltas: pd.DataFrame, exps: pd.DataFrame, rl: pd.DataFrame, mt: pd.DataFrame,
                     repro: pd.DataFrame, checks: pd.DataFrame, tokcheck: Mapping[str, object]) -> str:
    out = ["# exp03 summary (auto-generated by analyze_stages.py)", "",
           f"Δbpb RAW, paired window bootstrap 95% CIs ({N_BOOT} resamples, seed {SEED} + corpus index). "
           "'fresh_mean' = mean of the 3 fresh corpora, resampled within each corpus.", "",
           "## Pre-registered expectations (fresh mean)", "",
           "| id | comparison | expected | observed [95% CI] | verdict | note |", "|---|---|---|---|---|---|"]
    for r in exps.itertuples():
        obs = fmt_ci(r) if r.met != "n/a" else "missing"
        out.append(f"| {r.id} | {short(r.model)} vs {short(r.reference)} | {r.range} | {obs} | {r.met} | "
                   f"{getattr(r, 'note', '') or ''} |")
    out += ["", "## RL stages vs their predecessor (headline rule: whole CI below 0)", "",
            "| model | vs | corpus | Δbpb [95% CI] | below predecessor |", "|---|---|---|---|---|"]
    out += [f"| {short(r.model)} | {short(r.reference)} | {r.corpus} | {fmt_ci(r)} | "
            f"{'**yes**' if r.below_predecessor else 'no'} |" for r in rl.itertuples()]
    r1 = deltas[deltas.stage == "r1-distill"]
    out += ["", "## R1-Distill vs both references (Qwen2.5-Math-1.5B parent, plain Qwen2.5-1.5B)", "",
            "| vs | " + " | ".join(CORPORA) + f" | {FRESH} |", "|---|" + "---|" * (len(CORPORA) + 1)]
    for ref, g in r1.groupby("reference", sort=False):
        cells = [fmt_ci(g[g.corpus == c].iloc[0]) if (g.corpus == c).any() else "–" for c in (*CORPORA, FRESH)]
        out.append(f"| {short(ref)} | " + " | ".join(cells) + " |")
    for kind, title in (("vs_base", "Every stage vs its family base"), ("step", "Every stage vs its parent (step)")):
        d = step_rows(deltas) if kind == "step" else deltas[deltas.kind == "vs_base"]
        out += ["", f"## {title}", "", "| family | stage | model | vs | " + " | ".join(CORPORA) + f" | {FRESH} |",
                "|---|---|---|---|" + "---|" * (len(CORPORA) + 1)]
        for (model, ref), g in d.groupby(["model", "reference"], sort=False):
            cells = [fmt_ci(g[g.corpus == c].iloc[0]) if (g.corpus == c).any() else "–" for c in (*CORPORA, FRESH)]
            st = stage_of(model)
            out.append(f"| {st.family} | {st.stage} | {short(model)} | {short(ref)} | " + " | ".join(cells) + " |")
    out += ["", "## bpb per model", "", "| model | " + " | ".join(mt.columns) + " |",
            "|---|" + "---|" * len(mt.columns)]
    out += [f"| {short(m)} | " + " | ".join(f"{v:.5f}" for v in row) + " |" for m, row in mt.iterrows()]
    out += ["", f"## Reproduction: AAU (exp03) vs Mac (exp01), tolerance {REPRO_TOL_BPB:g} bpb", "",
            "| model | corpus | bpb AAU | bpb Mac | diff | total bits rel | max per-token |Δ| bits | within tol |",
            "|---|---|---|---|---|---|---|---|"]
    out += [f"| {short(r.model)} | {r.corpus} | {r.bpb_aau:.5f} | {r.bpb_mac:.5f} | {r.diff_bpb:+.1e} | "
            f"{r.total_bits_rel:+.1e} | {r.max_token_abs_bits:.1e} | {r.within_tol} |" for r in repro.itertuples()]
    out += ["", "## Per-model checks (job)", "",
            "| model | gpu | doc start | dtype | fp16 vs fp32 window 0 | deterministic | load s | model total s |",
            "|---|---|---|---|---|---|---|---|"]
    out += [f"| {short(r.model)} | {r.gpu} | {r.doc_start} ({r.doc_start_id}) | {r.dtype} | "
            f"{r.fp16_vs_fp32_rel:.1e} | {r.deterministic} | {r.load_s} | {r.total_model_s} |"
            for r in checks.itertuples()]
    groups = tokcheck.get("groups", {})
    out += ["", "## Tokenizer identity (job)", ""]
    out += [f"- {g}: {len(v['models'])} models, identical ids and exact round-trip on all corpora: {v['all_identical']}"
            for g, v in groups.items()]
    return "\n".join(out) + "\n"


def pct_label(x: float) -> str:
    """Signed percent, two decimals below 1% so small RL steps stay readable."""
    return f"{x:+.2f}%" if abs(x) < 1 else f"{x:+.1f}%"


def step_note(deltas: pd.DataFrame, model: str, parent: str | None, base: str) -> str:
    """Second label line for a stage trained on top of a non-base parent: its own step on the fresh mean."""
    if parent is None or parent == base:
        return ""
    r = lookup(deltas, model, parent)
    return "" if r is None else f"\n({pct_label(r.delta_pct)} vs {STAGE_LABEL[stage_of(parent).stage].split(chr(10))[0]})"


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def figure(deltas: pd.DataFrame) -> Figure:
    """Per family: Δbpb vs the family base for every stage; solid = fresh mean, hatched = enwik8 control.
    Stages trained on a non-base parent also carry their own step (vs that parent) in the label."""
    base_rows = deltas[deltas.kind == "vs_base"]
    widths = [max(1.6, len([s for s in STAGES if s.family == f and s.parent])) for f in FAMILIES]
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(14, 4.9), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": widths})
    w = 0.36
    for ax, fam in zip(axes, FAMILIES, strict=True):
        stages = [s for s in STAGES if s.family == fam and s.parent]
        for i, s in enumerate(stages):
            color = RL_C if s.stage in RL_STAGES else OTHER_C
            pair = {}
            for dx, corpus, hatch in ((-w / 2, FRESH, None), (w / 2, "enwik8_ctrl", "////")):
                hit = base_rows[(base_rows.model == s.model) & (base_rows.corpus == corpus)]
                if hit.empty:
                    continue
                r = pair[corpus] = hit.iloc[0]
                ax.bar(i + dx, r.delta_pct, width=w * 0.92, color=SURFACE if hatch else color,
                       edgecolor=color if hatch else SURFACE, hatch=hatch, linewidth=1 if hatch else 2)
                ax.errorbar(i + dx, r.delta_pct, yerr=[[r.delta_pct - r.lo], [r.hi - r.delta_pct]],
                            color=INK2, linewidth=1, capsize=2)
            if FRESH in pair:  # fresh-mean label centred over the pair, clear of both whiskers
                r = pair[FRESH]
                up = r.delta_pct >= 0
                y = max(max(p.hi for p in pair.values()), 0) if up else min(min(p.lo for p in pair.values()), 0)
                label = pct_label(r.delta_pct) + step_note(deltas, s.model, s.parent, family_base(fam))
                ax.annotate(label, (i, y), xytext=(0, 4 if up else -4), textcoords="offset points", ha="center",
                            va="bottom" if up else "top", fontsize=7.5, color=INK)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([STAGE_LABEL[s.stage] for s in stages], fontsize=7.5, color=INK2)
        ax.set_xlim(-0.6, len(stages) - 0.4)
        lo, hi = ax.get_ylim()
        span = hi - min(lo, -0.5)
        ax.set_ylim(min(lo, -0.5) - (0.06 * span if lo < 0 else 0), hi + 0.16 * span)
        ax.set_title(FAMILY_LABEL[fam], fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        style_axis(ax)
    axes[0].set_ylabel("Δ bits per byte vs family base (%)", color=INK2, fontsize=8.5)
    handles = [Patch(facecolor=OTHER_C, label="SFT / DPO / Instruct / continued pretraining / distillation"),
               Patch(facecolor=RL_C, label="stage that ends in pure RL (RLVR, SimpleRL-Zoo)"),
               Patch(facecolor=SURFACE, edgecolor=INK2, hatch="////", label="enwik8 control (2006 text)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("How far each post-training stage moves fresh-text bpb (mean of 3 fresh 2026 corpora, "
                 "paired window bootstrap 95% CI)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    return fig


# ----------------------------------------------------------------------------- effects
def load_inputs(res: Path, exp01: Path) -> dict[str, object]:
    bpb = pd.read_csv(res / "bpb.csv").drop_duplicates(["model", "condition", "corpus"], keep="last")
    bpb = bpb[bpb.condition == "raw"].reset_index(drop=True)
    win = pd.read_csv(res / "windows.csv").drop_duplicates(["model", "corpus", "window"], keep="last")
    lengths = {(m, c): tuple(g.sort_values("window").tokens) for (m, c), g in win.groupby(["model", "corpus"])}
    tok = {(r.model, r.corpus): load_token_bits(res, r.model, "raw", r.corpus) for r in bpb.itertuples()}
    wbits = {k: window_totals(v, list(lengths[k])) for k, v in tok.items()}
    exp01_bpb = pd.read_csv(exp01 / "bpb.csv")
    shared = set(exp01_bpb.model) & set(bpb.model)
    repro_tokens = {(r.model, r.corpus): (tok[(r.model, r.corpus)], load_token_bits(exp01, r.model, "raw", r.corpus))
                    for r in bpb.itertuples() if r.model in shared}
    checks = [json.loads(line) for line in (res / "model_checks.jsonl").read_text().splitlines() if line.strip()]
    checks = pd.DataFrame([c for c in checks if "dtype" in c]).drop_duplicates("model", keep="last")
    tokcheck = json.loads((res / "tokcheck.json").read_text()) if (res / "tokcheck.json").exists() else {}
    return {"bpb": bpb, "lengths": lengths, "wbits": wbits, "exp01_bpb": exp01_bpb, "repro_tokens": repro_tokens,
            "checks": checks, "tokcheck": tokcheck}


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(here / "results"))
    ap.add_argument("--exp01-results", default=str(here.parent / "exp01_posttraining_bpc" / "results"))
    args = ap.parse_args()
    res = Path(args.results)
    d = load_inputs(res, Path(args.exp01_results))
    deltas = delta_table(d["wbits"], d["lengths"], comparisons())
    exps, rl = expectation_table(deltas), rl_headline(deltas)
    repro = reproduction_table(d["bpb"], d["exp01_bpb"], d["repro_tokens"])
    deltas.to_csv(res / "stage_deltas.csv", index=False, float_format="%.5f")
    (res / "summary.md").write_text(summary_markdown(deltas, exps, rl, model_bpb_table(d["bpb"]), repro,
                                                     d["checks"], d["tokcheck"]))
    figure(deltas).savefig(res / "fig_stages.png", dpi=160, facecolor=SURFACE)
    print((res / "summary.md").read_text())


if __name__ == "__main__":
    main()

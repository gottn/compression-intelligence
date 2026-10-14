"""exp04 analysis: does fresh-text drift predict benchmark forgetting but not target-task gain?

    .venv/bin/python exp04_benchmarks/analyze_forgetting.py [--results R]

Reads   results/lm_eval/<model slug>/summary.json            (run_eval.py on AAU, fetched by run_all.sh fetch)
        ../exp03_staged_drift/results/stage_deltas.csv        (Δbpb vs family base, AAU T4)
        ../exp01_posttraining_bpc/results/bpb.csv             (SmolLM2 and Qwen3 pairs, Mac MPS)
Writes  results/scores.csv, results/correlations.csv, results/summary.md, results/fig_forgetting.png
Pure functions above the effects marker; file I/O only in load_inputs() and main().

Definitions are the pre-registered ones in RESULTS.md: general = mean of 6 primary accuracies (acc_norm where
reported, else acc); forgetting = 100 (general_base − general_post) / general_base; gain = gsm8k
flexible-extract, post − base, in points; Δbpb = 100 (Σ bits_post − Σ bits_base) / Σ bits_base, RAW, mean of
the 3 fresh corpora (and the enwik8 control). Correlations over the 12 post-trained checkpoints: Pearson
(gate statistic) and Spearman, 10,000 checkpoint-bootstrap resamples (seed 0), leave-one-out range.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from cilab.pairs import FRESH_CORPORA, model_slug
from cilab.stages import STAGES, Stage, family_base
from cilab.stats import bootstrap_corr, bootstrap_line_band, leave_one_out, ols_line, pearson, spearman

N_BOOT, N_BOOT_BAND, SEED = 10_000, 2_000, 0
GENERAL: tuple[tuple[str, str, float], ...] = (  # (task, primary metric key, chance accuracy)
    ("arc_easy", "acc_norm,none", 0.25), ("arc_challenge", "acc_norm,none", 0.25),
    ("hellaswag", "acc_norm,none", 0.25), ("piqa", "acc_norm,none", 0.5),
    ("winogrande", "acc,none", 0.5), ("mmlu", "acc,none", 0.25),
)
TARGET: tuple[tuple[str, str, str], ...] = (  # (column, task, metric key)
    ("gsm8k_flex", "gsm8k", "exact_match,flexible-extract"), ("gsm8k_strict", "gsm8k", "exact_match,strict-match"),
    ("ifeval_prompt_strict", "ifeval", "prompt_level_strict_acc,none"),
    ("ifeval_inst_strict", "ifeval", "inst_level_strict_acc,none"),
)
CHAT_STAGES: tuple[str, ...] = ("sft", "dpo", "rlvr", "instruct")
PREDICTORS: tuple[str, ...] = ("dbpb_fresh", "dbpb_enwik8")
OUTCOMES: tuple[str, ...] = ("forgetting", "forgetting_chance", "gain_gsm8k", "gain_gsm8k_strict", "gain_ifeval")
GATE_PASS, GATE_SOFT, ONE_SIDED_MAX = 0.8, 0.6, 0.3
STAGE_LABEL = {"sft": "SFT", "dpo": "DPO", "rlvr": "RLVR", "instruct": "Instruct", "rl-zero": "SimpleRL",
               "math-cpt": "Math", "r1-distill": "R1-Distill"}
FAMILY_LABEL = {"olmo2-1b": "OLMo-2-1B", "qwen2.5-0.5b": "Qwen2.5-0.5B", "qwen2.5-1.5b": "Qwen2.5-1.5B",
                "qwen2.5-3b": "Qwen2.5-3B", "smollm2": "SmolLM2-1.7B", "qwen3": "Qwen3-1.7B"}
# 3 categorical hues (the all-pairs cap for a scatter) + marker shape, so the 6 families never rest on colour
# alone; every point also carries a direct label. Dataviz reference palette slots 1-3, light mode.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FAMILY_STYLE = {"olmo2-1b": (BLUE, "o"), "qwen2.5-0.5b": (ORANGE, "v"), "qwen2.5-1.5b": (ORANGE, "o"),
                "qwen2.5-3b": (ORANGE, "^"), "smollm2": (AQUA, "s"), "qwen3": (AQUA, "D")}
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"


# ----------------------------------------------------------------------------- pure
def short(model: str) -> str:
    return model.split("/")[-1]


def point_label(st: Stage) -> str:
    """Stage name, plus the family size where the same stage appears in several Qwen2.5 sizes."""
    lab = STAGE_LABEL[st.stage]
    if st.family.startswith("qwen2.5") and st.stage in ("instruct", "rl-zero"):
        lab += " " + st.family.split("-")[1].upper()
    if st.family in ("smollm2", "qwen3"):
        lab = FAMILY_LABEL[st.family].split("-")[0] + " " + lab
    return lab


def score_row(summary: Mapping[str, object]) -> dict[str, object]:
    """Flat per-model scores from a run_eval.py summary.json: primary accuracy per general task, target
    metrics, the GPU, total wall time and the groups that did not finish."""
    tasks, wall, bad = {}, 0.0, []
    for g, rec in summary["groups"].items():
        wall += float(rec.get("wall_s", 0.0))
        if rec.get("status") != "ok":
            bad.append(g)
        tasks.update(rec.get("tasks", {}))
    row: dict[str, object] = {"model": summary["model"], "gpu": summary.get("gpu", ""), "eval_wall_s": wall,
                              "failed_groups": ",".join(bad)}
    for task, key, _ in GENERAL:
        row[task] = tasks.get(task, {}).get("metrics", {}).get(key, np.nan)
    for col, task, key in TARGET:
        row[col] = tasks.get(task, {}).get("metrics", {}).get(key, np.nan)
    return row


def general_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Adds `general` (mean primary accuracy) and `general_chance` (mean of (acc − c) / (1 − c))."""
    out = scores.copy()
    out["general"] = out[[t for t, _, _ in GENERAL]].mean(axis=1, skipna=False)
    adj = [(out[t] - c) / (1 - c) for t, _, c in GENERAL]
    out["general_chance"] = pd.concat(adj, axis=1).mean(axis=1, skipna=False)
    return out


def exp01_deltas(bpb: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Δbpb (%) vs the base from exp01's RAW rows: per corpus 100 (Σ bits_post − Σ bits_base) / Σ bits_base,
    fresh = mean of the 3 fresh corpora (exp03's point estimate on identical windows)."""
    raw = bpb[bpb.condition == "raw"].drop_duplicates(["model", "corpus"], keep="last").set_index(["model", "corpus"])
    rows = []
    for post, base in pairs:
        d = {}
        for c in (*FRESH_CORPORA, "enwik8_ctrl"):
            p, b = raw.loc[(post, c)], raw.loc[(base, c)]
            if (p.bytes, p.tokens) != (b.bytes, b.tokens):
                raise ValueError(f"{post} vs {base} on {c}: different bytes or tokens")
            d[c] = 100 * (p.total_bits - b.total_bits) / b.total_bits
        rows.append({"model": post, "dbpb_fresh": float(np.mean([d[c] for c in FRESH_CORPORA])),
                     "dbpb_enwik8": float(d["enwik8_ctrl"]), "bpb_source": "exp01 (Mac MPS)"})
    return pd.DataFrame(rows)


def predictor_table(stage_deltas: pd.DataFrame, exp01_bpb: pd.DataFrame,
                    stages: tuple[Stage, ...] = STAGES) -> pd.DataFrame:
    """Δbpb_fresh and Δbpb_enwik8 vs the family base for every post-trained checkpoint: exp03 rows where
    present, exp01 for the rest."""
    vb = stage_deltas[stage_deltas.kind == "vs_base"]
    piv = vb.pivot(index="model", columns="corpus", values="delta_pct")
    e3 = pd.DataFrame({"model": piv.index, "dbpb_fresh": piv["fresh_mean"].values,
                       "dbpb_enwik8": piv["enwik8_ctrl"].values, "bpb_source": "exp03 (AAU T4)"})
    post = [s for s in stages if s.parent is not None]
    rest = [(s.model, family_base(s.family, stages)) for s in post if s.model not in set(e3.model)]
    out = pd.concat([e3, exp01_deltas(exp01_bpb, rest)], ignore_index=True)
    order = [s.model for s in post]
    return out.set_index("model").reindex(order).reset_index()


def checkpoint_table(scores: pd.DataFrame, preds: pd.DataFrame, stages: tuple[Stage, ...] = STAGES) -> pd.DataFrame:
    """All checkpoints in table order with scores; post-trained rows also get predictors, forgetting and gain
    against their family base."""
    sc = general_scores(scores).set_index("model")
    pr = preds.set_index("model")
    rows = []
    for s in stages:
        if s.model not in sc.index:
            continue
        r = {"model": s.model, "family": s.family, "stage": s.stage, "base": family_base(s.family, stages),
             **sc.loc[s.model].to_dict()}
        if s.parent is not None and s.model in pr.index and r["base"] in sc.index:
            b = sc.loc[r["base"]]
            r.update(pr.loc[s.model].to_dict())
            r["forgetting"] = 100 * (b.general - r["general"]) / b.general
            r["forgetting_chance"] = 100 * (b.general_chance - r["general_chance"]) / b.general_chance
            r["gain_gsm8k"] = 100 * (r["gsm8k_flex"] - b.gsm8k_flex)
            r["gain_gsm8k_strict"] = 100 * (r["gsm8k_strict"] - b.gsm8k_strict)
            r["gain_ifeval"] = 100 * (r["ifeval_prompt_strict"] - b.ifeval_prompt_strict)
        rows.append(r)
    return pd.DataFrame(rows)


def post_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[table.stage != "base"].reset_index(drop=True)


def corr_row(df: pd.DataFrame, pred: str, out: str, subset: str) -> dict[str, object]:
    """Pearson and Spearman with checkpoint-bootstrap CIs and the leave-one-out range of Pearson r."""
    d = df[[pred, out, "model"]].dropna()
    x, y = d[pred].to_numpy(), d[out].to_numpy()
    p, plo, phi, pn = bootstrap_corr(x, y, pearson, N_BOOT, SEED)
    s, slo, shi, sn = bootstrap_corr(x, y, spearman, N_BOOT, SEED)
    loo = leave_one_out(x, y, pearson)
    worst = int(np.argmax(np.abs(loo - p)))
    return {"predictor": pred, "outcome": out, "subset": subset, "n": len(d), "pearson": p, "pearson_lo": plo,
            "pearson_hi": phi, "spearman": s, "spearman_lo": slo, "spearman_hi": shi,
            "boot_valid": min(pn, sn), "loo_min": float(loo.min()), "loo_max": float(loo.max()),
            "loo_most_influential": short(d.model.iloc[worst]), "loo_r_without_it": float(loo[worst])}


def correlation_table(table: pd.DataFrame) -> pd.DataFrame:
    """Every predictor × outcome over the 12 post-trained checkpoints; ifeval gain also over the chat-style 9."""
    post = post_rows(table)
    rows = [corr_row(post, p, o, "all post-trained") for p in PREDICTORS for o in OUTCOMES]
    chat = post[post.stage.isin(CHAT_STAGES)]
    rows += [corr_row(chat, p, "gain_ifeval", "chat-style") for p in PREDICTORS]
    return pd.DataFrame(rows)


def family_slopes(table: pd.DataFrame, min_points: int = 3) -> pd.DataFrame:
    """Within-family OLS slope (outcome units per Δbpb_fresh %) and Pearson r, families with ≥ min_points
    post-trained checkpoints."""
    post = post_rows(table)
    rows = []
    for fam, g in post.groupby("family", sort=False):
        if len(g) < min_points:
            continue
        for out in ("forgetting", "gain_gsm8k"):
            d = g[["dbpb_fresh", out]].dropna()
            if len(d) < min_points:
                continue
            slope, icpt = ols_line(d.dbpb_fresh.to_numpy(), d[out].to_numpy())
            rows.append({"family": fam, "outcome": out, "n": len(d), "slope_per_dbpb_pct": slope,
                         "intercept": icpt, "pearson": pearson(d.dbpb_fresh.to_numpy(), d[out].to_numpy())})
    return pd.DataFrame(rows)


def gate_verdict(r: float) -> str:
    if not np.isfinite(r):
        return "undefined"
    return "pass" if r >= GATE_PASS else "soft pass" if r >= GATE_SOFT else "fail"


def lookup_r(corr: pd.DataFrame, pred: str, out: str, subset: str = "all post-trained") -> float:
    hit = corr[(corr.predictor == pred) & (corr.outcome == out) & (corr.subset == subset)]
    return float(hit.pearson.iloc[0]) if len(hit) else float("nan")


def rule_table(corr: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered rules and expectations (RESULTS.md) against the outcome."""
    r_f, r_g = lookup_r(corr, "dbpb_fresh", "forgetting"), lookup_r(corr, "dbpb_fresh", "gain_gsm8k")
    r_e = lookup_r(corr, "dbpb_enwik8", "forgetting")
    gate = gate_verdict(r_f)
    one_sided = abs(r_g) < ONE_SIDED_MAX and gate in ("pass", "soft pass")
    post = post_rows(table).dropna(subset=["dbpb_fresh", "forgetting"])
    slope, icpt = ols_line(post.dbpb_fresh.to_numpy(), post.forgetting.to_numpy())
    r1 = post[post.stage == "r1-distill"]
    r1_resid = float(r1.forgetting.iloc[0] - (slope * r1.dbpb_fresh.iloc[0] + icpt)) if len(r1) else float("nan")
    r1_largest = bool(len(r1)) and r1.dbpb_fresh.iloc[0] == post.dbpb_fresh.max()
    return pd.DataFrame([
        {"rule": "Gate 1", "test": "r(Δbpb_fresh, forgetting) ≥ 0.8 pass, ≥ 0.6 soft", "observed": f"r = {r_f:.3f}",
         "verdict": gate},
        {"rule": "One-sidedness", "test": "abs r(Δbpb_fresh, gsm8k gain) < 0.3 and Gate 1 ≥ soft pass",
         "observed": f"r = {r_g:.3f}; gate {gate}", "verdict": "holds" if one_sided else "not shown"},
        {"rule": "E1", "test": "0.6 ≤ r(fresh, forgetting) < 0.8", "observed": f"r = {r_f:.3f}",
         "verdict": "met" if GATE_SOFT <= r_f < GATE_PASS else "not met"},
        {"rule": "E2", "test": "r(fresh, gsm8k gain) < 0.3", "observed": f"r = {r_g:.3f}",
         "verdict": "met" if r_g < ONE_SIDED_MAX else "not met"},
        {"rule": "E3", "test": "r(enwik8, forgetting) ≥ r(fresh, forgetting) − 0.05",
         "observed": f"enwik8 r = {r_e:.3f} vs fresh r = {r_f:.3f}",
         "verdict": "met" if r_e >= r_f - 0.05 else "not met"},
        {"rule": "E4", "test": "R1-Distill largest Δbpb_fresh and below the all-12 OLS line",
         "observed": f"largest: {r1_largest}; residual {r1_resid:+.2f} points",
         "verdict": "met" if r1_largest and r1_resid < 0 else "not met"},
    ])


def fmt(x: float, nd: int = 2) -> str:
    return "–" if not np.isfinite(x) else f"{x:+.{nd}f}"


def summary_markdown(table: pd.DataFrame, corr: pd.DataFrame, slopes: pd.DataFrame, rules: pd.DataFrame,
                     meta: Mapping[str, object]) -> str:
    out = ["# exp04 summary (auto-generated by analyze_forgetting.py)", "",
           f"Correlations over post-trained checkpoints; 95% CIs from {N_BOOT} checkpoint-bootstrap resamples "
           f"(seed {SEED}); LOO = leave-one-out range of Pearson r.", "",
           "## Rules and expectations (pre-registered in RESULTS.md)", "",
           "| rule | test | observed | verdict |", "|---|---|---|---|"]
    out += [f"| {r.rule} | {r.test} | {r.observed} | **{r.verdict}** |" for r in rules.itertuples()]
    out += ["", "## Correlations", "",
            "| predictor | outcome | subset | n | Pearson r [95% CI] | Spearman ρ [95% CI] | LOO r range | most influential (r without it) |",
            "|---|---|---|---|---|---|---|---|"]
    out += [f"| {r.predictor} | {r.outcome} | {r.subset} | {r.n} | {r.pearson:+.3f} [{r.pearson_lo:+.2f}, "
            f"{r.pearson_hi:+.2f}] | {r.spearman:+.3f} [{r.spearman_lo:+.2f}, {r.spearman_hi:+.2f}] | "
            f"{r.loo_min:+.2f} to {r.loo_max:+.2f} | {r.loo_most_influential} ({r.loo_r_without_it:+.2f}) |"
            for r in corr.itertuples()]
    out += ["", "## Within-family slopes (post-trained checkpoints only)", "",
            "| family | outcome | n | slope per Δbpb % | Pearson r |", "|---|---|---|---|---|"]
    out += [f"| {r.family} | {r.outcome} | {r.n} | {r.slope_per_dbpb_pct:+.3f} | {r.pearson:+.3f} |"
            for r in slopes.itertuples()]
    gen = [t for t, _, _ in GENERAL]
    out += ["", "## Per-checkpoint scores (accuracy %, primary metric per task)", "",
            "| family | stage | model | " + " | ".join(gen) + " | general | gsm8k flex | gsm8k strict | ifeval prompt-strict |",
            "|---|---|---|" + "---|" * (len(gen) + 4)]
    for r in table.itertuples():
        cells = [f"{100 * getattr(r, t):.1f}" for t in gen]
        out.append(f"| {r.family} | {r.stage} | {short(r.model)} | " + " | ".join(cells) +
                   f" | {100 * r.general:.2f} | {100 * r.gsm8k_flex:.1f} | {100 * r.gsm8k_strict:.1f} | "
                   f"{100 * r.ifeval_prompt_strict:.1f} |")
    post = post_rows(table)
    out += ["", "## Post-trained checkpoints: predictors and outcomes", "",
            "| family | stage | model | Δbpb fresh % | Δbpb enwik8 % | forgetting % | forgetting (chance-adj.) % | "
            "gsm8k gain (flex) pts | gsm8k gain (strict) pts | ifeval gain pts | bpb source |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    out += [f"| {r.family} | {r.stage} | {short(r.model)} | {fmt(r.dbpb_fresh)} | {fmt(r.dbpb_enwik8)} | "
            f"{fmt(r.forgetting)} | {fmt(r.forgetting_chance)} | {fmt(r.gain_gsm8k, 1)} | {fmt(r.gain_gsm8k_strict, 1)} | "
            f"{fmt(r.gain_ifeval, 1)} | {r.bpb_source} |" for r in post.itertuples()]
    out += ["", "## Runs", "", "| model | gpu | eval wall s | failed groups |", "|---|---|---|---|"]
    out += [f"| {short(r.model)} | {r.gpu} | {r.eval_wall_s:.0f} | {r.failed_groups or '–'} |" for r in table.itertuples()]
    out += ["", "## Harness", ""] + [f"- {k}: {v}" for k, v in meta.items()]
    return "\n".join(out) + "\n"


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)


def scatter_panel(ax: plt.Axes, post: pd.DataFrame, out: str, ylabel: str, r_row: pd.Series) -> None:
    d = post.dropna(subset=["dbpb_fresh", out])
    x, y = d.dbpb_fresh.to_numpy(), d[out].to_numpy()
    grid = np.linspace(min(0.0, x.min()), x.max() * 1.04, 120)
    slope, icpt = ols_line(x, y)
    lo, hi = bootstrap_line_band(x, y, grid, N_BOOT_BAND, SEED)
    ax.fill_between(grid, lo, hi, color=MUTED, alpha=0.18, linewidth=0)
    ax.plot(grid, slope * grid + icpt, color=INK2, linewidth=2)
    ax.axhline(0, color=INK2, linewidth=1)
    for r in d.itertuples():
        color, marker = FAMILY_STYLE[r.family]
        ax.scatter(r.dbpb_fresh, getattr(r, out), s=58, color=color, marker=marker, edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        st = next(s for s in STAGES if s.model == r.model)
        ax.annotate(point_label(st), (r.dbpb_fresh, getattr(r, out)), xytext=(5, 4), textcoords="offset points",
                    fontsize=7, color=INK2)
    ax.set_xlabel("Δ bits per byte on fresh 2026 text vs family base (%)", color=INK2, fontsize=8.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)
    ax.set_title(f"Pearson r = {r_row.pearson:+.2f} [{r_row.pearson_lo:+.2f}, {r_row.pearson_hi:+.2f}],  "
                 f"Spearman ρ = {r_row.spearman:+.2f}  (n = {r_row.n})", fontsize=8.5, color=INK, loc="left")
    style_axis(ax)


def r_panel(ax: plt.Axes, corr: pd.DataFrame) -> None:
    """Pearson r with 95% CI for each predictor, for forgetting and for gsm8k gain."""
    allp = corr[corr.subset == "all post-trained"]
    items = [("forgetting", "forgetting"), ("gain_gsm8k", "gsm8k gain")]
    for i, (out, _name) in enumerate(items):
        for dy, pred, color, lab in ((-0.14, "dbpb_fresh", INK, "fresh 2026 text"),
                                     (0.14, "dbpb_enwik8", MUTED, "enwik8 control")):
            r = allp[(allp.predictor == pred) & (allp.outcome == out)].iloc[0]
            ax.errorbar(r.pearson, i + dy, xerr=[[r.pearson - r.pearson_lo], [r.pearson_hi - r.pearson]],
                        fmt="o", color=color, markersize=6, capsize=2, linewidth=1.5,
                        label=lab if i == 0 else None)
    # rule zones in data coordinates: Gate 1 on the forgetting row, the one-sidedness band on the gain row
    ax.add_patch(Rectangle((GATE_SOFT, -0.45), GATE_PASS - GATE_SOFT, 0.9, color=GRID, linewidth=0))
    ax.add_patch(Rectangle((GATE_PASS, -0.45), 1.02 - GATE_PASS, 0.9, color=GRID, alpha=0.5, linewidth=0))
    ax.add_patch(Rectangle((-ONE_SIDED_MAX, 0.55), 2 * ONE_SIDED_MAX, 0.9, color=GRID, linewidth=0))
    ax.annotate("soft | pass", (GATE_SOFT, -0.45), xytext=(2, 2), textcoords="offset points", fontsize=6.5,
                color=INK2, va="bottom")
    ax.annotate("one-sided zone", (-ONE_SIDED_MAX, 0.55), xytext=(2, 2), textcoords="offset points", fontsize=6.5,
                color=INK2, va="bottom")
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for _, n in items], fontsize=8, color=INK2)
    ax.set_ylim(len(items) - 0.4, -0.6)  # forgetting on top
    ax.set_xlim(-1.02, 1.02)
    ax.set_xlabel("Pearson r with Δbpb (95% CI)", color=INK2, fontsize=8.5)
    ax.set_title("Predictor: fresh text vs enwik8", fontsize=8.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower left")
    style_axis(ax)


def figure(table: pd.DataFrame, corr: pd.DataFrame) -> Figure:
    post = post_rows(table)
    allp = corr[(corr.subset == "all post-trained") & (corr.predictor == "dbpb_fresh")].set_index("outcome")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1, 0.62]})
    scatter_panel(axes[0], post, "forgetting", "forgetting: drop in mean general accuracy vs base (% of base)",
                  allp.loc["forgetting"])
    scatter_panel(axes[1], post, "gain_gsm8k", "gain: gsm8k (flexible-extract) post − base (points)",
                  allp.loc["gain_gsm8k"])
    r_panel(axes[2], corr)
    handles = [Line2D([], [], color=c, marker=m, linestyle="", markersize=7, label=FAMILY_LABEL[f])
               for f, (c, m) in FAMILY_STYLE.items()]
    handles.append(Line2D([], [], color=INK2, linewidth=2, label="OLS fit, 95% bootstrap band"))
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Does fresh-text drift predict what post-training loses (left) but not what it gains (middle)? "
                 "12 post-trained checkpoints, no chat template", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig


# ----------------------------------------------------------------------------- effects
def load_inputs(res: Path, exp03: Path, exp01: Path) -> dict[str, object]:
    summaries = {}
    for s in STAGES:
        p = res / "lm_eval" / model_slug(s.model) / "summary.json"
        if p.exists():
            summaries[s.model] = json.loads(p.read_text())
    return {"summaries": summaries, "stage_deltas": pd.read_csv(exp03 / "stage_deltas.csv"),
            "exp01_bpb": pd.read_csv(exp01 / "bpb.csv")}


def harness_meta(summaries: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Versions, task versions and few-shot counts as recorded by the jobs (all models must agree)."""
    vers, tasks = set(), {}
    for sm in summaries.values():
        vers.add(json.dumps(sm.get("versions", {}), sort_keys=True))
        for rec in sm["groups"].values():
            for t, v in rec.get("tasks", {}).items():
                tasks.setdefault(t, set()).add((str(v.get("version")), str(v.get("leaf_versions")),
                                                str(v.get("n_shot")), v.get("n_effective")))
    meta: dict[str, object] = {"environments": sorted(vers)}
    for t, v in sorted(tasks.items()):
        meta[t] = "; ".join(f"version {a} (leaves {b}), n-shot {c}, docs {d}" for a, b, c, d in sorted(v))
    return meta


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(here / "results"))
    args = ap.parse_args()
    res = Path(args.results)
    d = load_inputs(res, here.parent / "exp03_staged_drift" / "results", here.parent / "exp01_posttraining_bpc" / "results")
    scores = pd.DataFrame([score_row(s) for s in d["summaries"].values()])
    preds = predictor_table(d["stage_deltas"], d["exp01_bpb"])
    table = checkpoint_table(scores, preds)
    corr, slopes = correlation_table(table), family_slopes(table)
    rules = rule_table(corr, table)
    table.to_csv(res / "scores.csv", index=False, float_format="%.5f")
    corr.to_csv(res / "correlations.csv", index=False, float_format="%.4f")
    (res / "summary.md").write_text(summary_markdown(table, corr, slopes, rules, harness_meta(d["summaries"])))
    figure(table, corr).savefig(res / "fig_forgetting.png", dpi=160, facecolor=SURFACE)
    print((res / "summary.md").read_text())


if __name__ == "__main__":
    main()

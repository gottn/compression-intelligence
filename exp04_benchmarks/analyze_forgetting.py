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
OUTCOMES: tuple[str, ...] = ("forgetting", "forgetting_chance", "gain_gsm8k", "gain_gsm8k_strict", "gain_ifeval",
                             "gain_gsm8k_chat", "gain_gsm8k_chat_strict", "gain_ifeval_chat")
GATE_PASS, GATE_SOFT, ONE_SIDED_MAX = 0.8, 0.6, 0.3
Z95 = 1.96
MATH_STAGES: tuple[str, ...] = ("math-cpt", "r1-distill")
ALL12 = "all 12 (pre-registered)"
NO_MATH = "without Qwen2.5-Math and R1-Distill"
R1_PARENT = "R1-Distill vs its Math parent"
NO_INSTRUCT = "without the 5 instruct-stage checkpoints"
CHAT = "chat-style (sft/dpo/rlvr/instruct)"
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


def score_row(summary: Mapping[str, object], chat: Mapping[str, object] | None = None) -> dict[str, object]:
    """Flat per-model scores from a run_eval.py summary.json: primary accuracy per general task, target
    metrics, the GPU, total wall time and the groups that did not finish; with the chat-condition summary
    (summary_chat.json), the same target metrics as *_chat columns."""
    tasks, wall, bad = {}, 0.0, []
    for g, rec in summary["groups"].items():
        wall += float(rec.get("wall_s", 0.0))
        if rec.get("status") != "ok":
            bad.append(g)
        tasks.update(rec.get("tasks", {}))
    row: dict[str, object] = {"model": summary["model"], "gpu": summary.get("gpu", ""), "eval_wall_s": wall,
                              "failed_groups": ",".join(bad)}
    for task, key, _ in GENERAL:
        m = tasks.get(task, {}).get("metrics", {})
        row[task] = m.get(key, np.nan)
        row[f"{task}_se"] = m.get(key.replace(",", "_stderr,", 1), np.nan)
    for col, task, key in TARGET:
        row[col] = tasks.get(task, {}).get("metrics", {}).get(key, np.nan)
    ctasks, cwall = {}, 0.0
    for g, rec in (chat or {}).get("groups", {}).items():
        cwall += float(rec.get("wall_s", 0.0))
        if rec.get("status") != "ok":
            bad.append(g)
        ctasks.update(rec.get("tasks", {}))
    for col, task, key in TARGET:
        row[f"{col}_chat"] = ctasks.get(task, {}).get("metrics", {}).get(key, np.nan)
    row["chat_wall_s"], row["failed_groups"] = cwall if chat else np.nan, ",".join(bad)
    return row


def general_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Adds `general` (mean primary accuracy) and `general_chance` (mean of (acc − c) / (1 − c))."""
    out = scores.copy()
    out["general"] = out[[t for t, _, _ in GENERAL]].mean(axis=1, skipna=False)
    adj = [(out[t] - c) / (1 - c) for t, _, c in GENERAL]
    out["general_chance"] = pd.concat(adj, axis=1).mean(axis=1, skipna=False)
    se = out[[f"{t}_se" for t, _, _ in GENERAL]]
    out["general_se"] = np.sqrt((se**2).sum(axis=1, skipna=False)) / len(GENERAL)
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


def bpb_cost(bpb: pd.DataFrame, hardware: str) -> pd.DataFrame:
    """Measured wall time (s) to score the 3 fresh corpora per model (RAW rows; one forward pass per window)."""
    raw = bpb[(bpb.condition == "raw") & bpb.corpus.isin(FRESH_CORPORA)].drop_duplicates(["model", "corpus"], keep="last")
    out = raw.groupby("model", sort=False).wall_s.sum().rename("bpb_fresh_wall_s").reset_index()
    out["bpb_hardware"] = hardware
    return out


def cost_table(table: pd.DataFrame, e3_bpb: pd.DataFrame, e1_bpb: pd.DataFrame) -> pd.DataFrame:
    """Per checkpoint: benchmark wall time (this run) vs fresh-text scoring wall time (exp03 T4, else exp01 Mac)."""
    c3, c1 = bpb_cost(e3_bpb, "T4 (exp03)"), bpb_cost(e1_bpb, "M2 Pro MPS (exp01)")
    cost = pd.concat([c3, c1[~c1.model.isin(c3.model)]], ignore_index=True).set_index("model")
    t = table[["model", "family", "stage", "gpu", "eval_wall_s"]].set_index("model").join(cost, how="left")
    t["ratio"] = t.eval_wall_s / t.bpb_fresh_wall_s
    return t.reset_index()


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


def outcome_fields(post: Mapping[str, float], ref: Mapping[str, float]) -> dict[str, float]:
    """Forgetting (% of the reference's general score), its 95% noise floor, and gains (points) of a checkpoint
    against a reference. The floor treats the two scores as independent (1.96 × the root of the summed squared
    standard errors of the 6-task means), which overstates it a little because both models answer the same items."""
    floor = 100 * Z95 * np.sqrt(post["general_se"] ** 2 + ref["general_se"] ** 2) / ref["general"]
    forgetting = 100 * (ref["general"] - post["general"]) / ref["general"]
    return {"forgetting": forgetting, "floor_pct": floor, "within_floor": bool(abs(forgetting) <= floor),
            "forgetting_chance": 100 * (ref["general_chance"] - post["general_chance"]) / ref["general_chance"],
            "gain_gsm8k": 100 * (post["gsm8k_flex"] - ref["gsm8k_flex"]),
            "gain_gsm8k_strict": 100 * (post["gsm8k_strict"] - ref["gsm8k_strict"]),
            "gain_ifeval": 100 * (post["ifeval_prompt_strict"] - ref["ifeval_prompt_strict"]),
            # chat condition: the post-trained model with its chat template vs the reference as prompted above
            "gain_gsm8k_chat": 100 * (post.get("gsm8k_flex_chat", np.nan) - ref["gsm8k_flex"]),
            "gain_gsm8k_chat_strict": 100 * (post.get("gsm8k_strict_chat", np.nan) - ref["gsm8k_strict"]),
            "gain_ifeval_chat": 100 * (post.get("ifeval_prompt_strict_chat", np.nan) - ref["ifeval_prompt_strict"])}


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
             "reference": "", **sc.loc[s.model].to_dict()}
        if s.parent is not None and s.model in pr.index and r["base"] in sc.index:
            r["reference"] = r["base"]
            r.update(pr.loc[s.model].to_dict())
            r.update(outcome_fields(sc.loc[s.model], sc.loc[r["base"]]))
        rows.append(r)
    return pd.DataFrame(rows)


def parent_reference_row(table: pd.DataFrame, stage_deltas: pd.DataFrame, model: str,
                         stages: tuple[Stage, ...] = STAGES) -> dict[str, object]:
    """The same checkpoint measured against its parent instead of the family base (R1-Distill vs Qwen2.5-Math):
    Δbpb from exp03's vs_parent rows, forgetting and gain from this run's scores."""
    parent = next(s.parent for s in stages if s.model == model)
    t = table.set_index("model")
    d = stage_deltas[(stage_deltas.model == model) & (stage_deltas.reference == parent)].set_index("corpus").delta_pct
    row = {**t.loc[model].to_dict(), "model": model, "reference": parent, "dbpb_fresh": float(d["fresh_mean"]),
           "dbpb_enwik8": float(d["enwik8_ctrl"]), "bpb_source": "exp03 (AAU T4), vs parent"}
    row.update(outcome_fields(t.loc[model], t.loc[parent]))
    return row


def post_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[table.stage != "base"].reset_index(drop=True)


def corr_row(df: pd.DataFrame, pred: str, out: str, subset: str) -> dict[str, object]:
    """Pearson and Spearman with checkpoint-bootstrap CIs and the leave-one-out range of Pearson r."""
    d = df[[pred, out, "model"]].dropna()
    if len(d) < 3:
        return {"predictor": pred, "outcome": out, "subset": subset, "n": len(d)}
    x, y = d[pred].to_numpy(), d[out].to_numpy()
    p, plo, phi, pn = bootstrap_corr(x, y, pearson, N_BOOT, SEED)
    s, slo, shi, sn = bootstrap_corr(x, y, spearman, N_BOOT, SEED)
    loo = leave_one_out(x, y, pearson)
    worst = int(np.argmax(np.abs(loo - p)))
    return {"predictor": pred, "outcome": out, "subset": subset, "n": len(d), "pearson": p, "pearson_lo": plo,
            "pearson_hi": phi, "spearman": s, "spearman_lo": slo, "spearman_hi": shi,
            "boot_valid": min(pn, sn), "loo_min": float(loo.min()), "loo_max": float(loo.max()),
            "loo_most_influential": short(d.model.iloc[worst]), "loo_r_without_it": float(loo[worst])}


def subsets(table: pd.DataFrame, r1_parent: Mapping[str, object] | None) -> dict[str, pd.DataFrame]:
    """The pre-registered set (12 post-trained vs family base) and the sensitivity sets (not pre-registered)."""
    post = post_rows(table)
    out = {ALL12: post, NO_MATH: post[~post.stage.isin(MATH_STAGES)].reset_index(drop=True),
           NO_INSTRUCT: post[post.stage != "instruct"].reset_index(drop=True),
           CHAT: post[post.stage.isin(CHAT_STAGES)].reset_index(drop=True)}
    if r1_parent is not None:
        swapped = post[post.stage != "r1-distill"]
        out[R1_PARENT] = pd.concat([swapped, pd.DataFrame([r1_parent])], ignore_index=True)
    return out


SUBSET_OUTCOMES: dict[str, tuple[str, ...]] = {
    ALL12: OUTCOMES, NO_MATH: ("forgetting", "gain_gsm8k", "gain_gsm8k_chat"),
    R1_PARENT: ("forgetting", "gain_gsm8k", "gain_gsm8k_chat"),
    NO_INSTRUCT: ("gain_gsm8k", "gain_gsm8k_chat"), CHAT: ("gain_ifeval", "gain_ifeval_chat"),
}


def correlation_table(table: pd.DataFrame, r1_parent: Mapping[str, object] | None = None) -> pd.DataFrame:
    """Every predictor × outcome on the pre-registered 12, plus the sensitivity subsets in SUBSET_OUTCOMES."""
    sets = subsets(table, r1_parent)
    return pd.DataFrame([corr_row(sets[name], p, o, name) for name, outs in SUBSET_OUTCOMES.items()
                         if name in sets for p in PREDICTORS for o in outs])


def threshold_reading(post: pd.DataFrame) -> dict[str, object]:
    """Where forgetting leaves the benchmark noise floor along Δbpb_fresh. `contiguous`: largest Δbpb up to which
    every checkpoint is inside its own floor; `largest_inside`: the largest Δbpb of any checkpoint inside its
    floor, so every checkpoint above it is outside; `exceptions`: checkpoints at or below largest_inside that are
    outside their floor. Rows are (model, Δbpb %, forgetting %, floor %)."""
    d = post.dropna(subset=["dbpb_fresh", "forgetting"]).sort_values("dbpb_fresh").reset_index(drop=True)
    k = 0
    while k < len(d) and d.within_floor.iloc[k]:
        k += 1
    inside = d[d.within_floor.astype(bool)]
    top = float(inside.dbpb_fresh.max()) if len(inside) else float("nan")

    def rows(x: pd.DataFrame) -> list[tuple[str, float, float, float]]:
        return [(short(r.model), float(r.dbpb_fresh), float(r.forgetting), float(r.floor_pct)) for r in x.itertuples()]

    return {"n": len(d), "contiguous": float(d.dbpb_fresh.iloc[k - 1]) if k else float("nan"), "n_contiguous": k,
            "largest_inside": top, "n_inside": len(inside),
            "max_abs_forgetting_inside": float(inside.forgetting.abs().max()) if len(inside) else float("nan"),
            "above": rows(d[d.dbpb_fresh > top]), "exceptions": rows(d[(d.dbpb_fresh <= top) & ~d.within_floor.astype(bool)]),
            "median_floor": float(d.floor_pct.median())}


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


def lookup_r(corr: pd.DataFrame, pred: str, out: str, subset: str = ALL12, stat: str = "pearson") -> float:
    hit = corr[(corr.predictor == pred) & (corr.outcome == out) & (corr.subset == subset)]
    return float(hit[stat].iloc[0]) if len(hit) else float("nan")


def rule_table(corr: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered rules and expectations (RESULTS.md) against the outcome."""
    r_f, r_g = lookup_r(corr, "dbpb_fresh", "forgetting"), lookup_r(corr, "dbpb_fresh", "gain_gsm8k")
    r_e = lookup_r(corr, "dbpb_enwik8", "forgetting")
    rho_f = lookup_r(corr, "dbpb_fresh", "forgetting", stat="spearman")
    r_nm = lookup_r(corr, "dbpb_fresh", "forgetting", NO_MATH)
    rho_nm = lookup_r(corr, "dbpb_fresh", "forgetting", NO_MATH, "spearman")
    gate = gate_verdict(r_f)
    one_sided = abs(r_g) < ONE_SIDED_MAX and gate in ("pass", "soft pass")
    r_gc = lookup_r(corr, "dbpb_fresh", "gain_gsm8k_chat")
    rho_gc = lookup_r(corr, "dbpb_fresh", "gain_gsm8k_chat", stat="spearman")
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
        {"rule": "One-sidedness, chat-condition gain (gain measure changed after the run)",
         "test": "abs r(Δbpb_fresh, gsm8k gain, chat) < 0.3 and Gate 1 ≥ soft pass",
         "observed": f"r = {r_gc:.3f}; ρ = {rho_gc:.3f}" if np.isfinite(r_gc) else "chat condition not run",
         "verdict": ("holds" if abs(r_gc) < ONE_SIDED_MAX and gate in ("pass", "soft pass") else "not shown")
         if np.isfinite(r_gc) else "n/a"},
        {"rule": "Gate 1 on Spearman (not pre-registered)", "test": "same thresholds on Spearman ρ",
         "observed": f"ρ = {rho_f:.3f}", "verdict": gate_verdict(rho_f)},
        {"rule": "Gate 1 without Qwen2.5-Math and R1-Distill (not pre-registered)",
         "test": "same thresholds on Pearson r, n = 10", "observed": f"r = {r_nm:.3f}; ρ = {rho_nm:.3f}",
         "verdict": gate_verdict(r_nm)},
    ])


def fmt(x: float, nd: int = 2) -> str:
    return "–" if not np.isfinite(x) else f"{x:+.{nd}f}"


def summary_markdown(table: pd.DataFrame, corr: pd.DataFrame, slopes: pd.DataFrame, rules: pd.DataFrame,
                     cost: pd.DataFrame, thr: Mapping[str, object], r1_parent: Mapping[str, object] | None,
                     meta: Mapping[str, object]) -> str:
    out = ["# exp04 summary (auto-generated by analyze_forgetting.py)", "",
           f"Correlations over post-trained checkpoints; 95% CIs from {N_BOOT} checkpoint-bootstrap resamples "
           f"(seed {SEED}); LOO = leave-one-out range of Pearson r.", "",
           "## Rules and expectations (pre-registered in RESULTS.md)", "",
           "| rule | test | observed | verdict |", "|---|---|---|---|"]
    out += [f"| {r.rule} | {r.test} | {r.observed} | **{r.verdict}** |" for r in rules.itertuples()]
    def fmt_rows(rows: list[tuple[str, float, float, float]]) -> str:
        return "; ".join(f"{m} (Δbpb {d:+.1f}%, forgetting {f:+.1f}%, floor ±{fl:.1f}%)" for m, d, f, fl in rows) or "none"

    out += ["", "## Threshold reading (not pre-registered)", "",
            f"Noise floor per checkpoint = 1.96 × √(SE_post² + SE_base²) of the 6-task mean, as % of the base score "
            f"(median ±{thr['median_floor']:.1f}%; conservative, it ignores that both models answer the same items).", "",
            f"- {thr['n_inside']} of {thr['n']} checkpoints are inside their floor (largest |forgetting| "
            f"{thr['max_abs_forgetting_inside']:.1f}%); the largest drift still inside is Δbpb_fresh "
            f"{thr['largest_inside']:+.1f}%.",
            f"- Above it, every checkpoint is outside: {fmt_rows(thr['above'])}.",
            f"- At or below it, outside the floor anyway: {fmt_rows(thr['exceptions'])}.",
            f"- Contiguous reading: every checkpoint up to Δbpb {thr['contiguous']:+.1f}% "
            f"({thr['n_contiguous']} of {thr['n']}) is inside."]
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
            "| family | stage | model | " + " | ".join(gen) + " | general | gsm8k flex | gsm8k strict | ifeval prompt-strict "
            "| gsm8k flex, chat | gsm8k strict, chat | ifeval prompt-strict, chat |",
            "|---|---|---|" + "---|" * (len(gen) + 7)]
    for r in table.itertuples():
        cells = [f"{100 * getattr(r, t):.1f}" for t in gen]
        out.append(f"| {r.family} | {r.stage} | {short(r.model)} | " + " | ".join(cells) +
                   f" | {100 * r.general:.2f} | {100 * r.gsm8k_flex:.1f} | {100 * r.gsm8k_strict:.1f} | "
                   f"{100 * r.ifeval_prompt_strict:.1f} | " + " | ".join(
                       "–" if not np.isfinite(v) else f"{100 * v:.1f}"
                       for v in (r.gsm8k_flex_chat, r.gsm8k_strict_chat, r.ifeval_prompt_strict_chat)) + " |")
    post = post_rows(table)
    out += ["", "## Post-trained checkpoints: predictors and outcomes (last row: R1-Distill against its parent, "
            "a second reference for the same model, not one of the 12)", "",
            "| family | stage | model | Δbpb fresh % | Δbpb enwik8 % | forgetting % | forgetting (chance-adj.) % | "
            "gsm8k gain (flex) pts | gsm8k gain (strict) pts | ifeval gain pts | gsm8k gain, chat (flex) pts | "
            "ifeval gain, chat pts | bpb source |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out += [f"| {r.family} | {r.stage} | {short(r.model)} | {fmt(r.dbpb_fresh)} | {fmt(r.dbpb_enwik8)} | "
            f"{fmt(r.forgetting)} | {fmt(r.forgetting_chance)} | {fmt(r.gain_gsm8k, 1)} | {fmt(r.gain_gsm8k_strict, 1)} | "
            f"{fmt(r.gain_ifeval, 1)} | {fmt(r.gain_gsm8k_chat, 1)} | {fmt(r.gain_ifeval_chat, 1)} | {r.bpb_source} |"
            for r in post.itertuples()]
    if r1_parent is not None:
        r = r1_parent
        out.append(f"| {r['family']} | {r['stage']} vs parent | {short(r['model'])} vs {short(r['reference'])} | "
                   f"{fmt(r['dbpb_fresh'])} | {fmt(r['dbpb_enwik8'])} | {fmt(r['forgetting'])} | "
                   f"{fmt(r['forgetting_chance'])} | {fmt(r['gain_gsm8k'], 1)} | {fmt(r['gain_gsm8k_strict'], 1)} | "
                   f"{fmt(r['gain_ifeval'], 1)} | {fmt(r['gain_gsm8k_chat'], 1)} | {fmt(r['gain_ifeval_chat'], 1)} | "
                   f"{r['bpb_source']} |")
    out += ["", "Noise floor (± % of base) per row above: " + ", ".join(
        f"{short(r.model)} {r.floor_pct:.1f}" for r in post.itertuples()) + "."]
    out += ["", "## Runs", "", "| model | gpu | eval wall s | failed groups |", "|---|---|---|---|"]
    out += [f"| {short(r.model)} | {r.gpu} | {r.eval_wall_s:.0f} | {r.failed_groups or '–'} |" for r in table.itertuples()]
    out += ["", "## Cost per checkpoint (measured wall time)", "",
            f"Median benchmark pass {cost.eval_wall_s.median() / 60:.1f} min (A10) vs median fresh-text scoring "
            f"{cost.bpb_fresh_wall_s.median():.0f} s (3 corpora, ~300 KB); median ratio {cost.ratio.median():.0f}×. "
            "Benchmark time excludes the ~2.5 min of job setup (imports, task index, model load).", "",
            "| model | benchmark s (gpu) | fresh-text bpb s (hardware) | ratio |", "|---|---|---|---|"]
    out += [f"| {short(r.model)} | {r.eval_wall_s:.0f} ({r.gpu}) | {r.bpb_fresh_wall_s:.0f} ({r.bpb_hardware}) | "
            f"{r.ratio:.0f}× |" for r in cost.itertuples()]
    out += ["", "## Harness", ""] + [f"- {k}: {v}" for k, v in meta.items()]
    return "\n".join(out) + "\n"


def nudge_labels(anchors: list[tuple[float, float]], widths: list[float], height: float,
                 dx: float = 5.0, dy: float = 4.0, marker: float = 0.0) -> list[tuple[float, float]]:
    """Label offsets (points) that keep label boxes from overlapping each other and every marker (a square of
    side 2 × marker around each anchor). anchors are point positions in points; each label starts up-right of
    its point and moves up by one line height until its box is clear. Labels are placed bottom-up, so the
    result depends only on the inputs."""
    order = sorted(range(len(anchors)), key=lambda i: (anchors[i][1], anchors[i][0]))
    boxes: list[tuple[float, float, float, float]] = [
        (x - marker, y - marker, x + marker, y + marker) for x, y in anchors] if marker > 0 else []
    out: list[tuple[float, float]] = [(dx, dy)] * len(anchors)
    for i in order:
        x0 = anchors[i][0] + dx
        y0 = anchors[i][1] + dy
        while any(x0 < bx1 and bx0 < x0 + widths[i] and y0 < by1 and by0 < y0 + height for bx0, by0, bx1, by1 in boxes):
            y0 += height
        boxes.append((x0, y0, x0 + widths[i], y0 + height))
        out[i] = (dx, y0 - anchors[i][1])
    return out


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)


def scatter_label(st: Stage) -> str:
    """Point label; the two math-line points are named in full because they carry the fit."""
    return {"math-cpt": "Qwen2.5-Math", "r1-distill": "R1-Distill vs base"}.get(st.stage, point_label(st))


Label = tuple[str, float, float, str]  # (text, x, y, colour)


def scatter_panel(ax: plt.Axes, post: pd.DataFrame, out: str, ylabel: str, r_row: pd.Series,
                  r1_parent: Mapping[str, object] | None, floor: float | None) -> list[Label]:
    """Draws the points, fit and band; returns the point labels, placed by place_labels after layout."""
    d = post.dropna(subset=["dbpb_fresh", out])
    x, y = d.dbpb_fresh.to_numpy(), d[out].to_numpy()
    grid = np.linspace(min(0.0, x.min()), x.max() * 1.04, 120)
    slope, icpt = ols_line(x, y)
    lo, hi = bootstrap_line_band(x, y, grid, N_BOOT_BAND, SEED)
    if floor is not None:
        ax.axhspan(-floor, floor, color=AQUA, alpha=0.10, linewidth=0)
        ax.annotate(f"benchmark noise floor (median ±{floor:.1f}%)", (grid[-1], floor), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=6.5, color=INK2)
    ax.fill_between(grid, lo, hi, color=MUTED, alpha=0.18, linewidth=0)
    ax.plot(grid, slope * grid + icpt, color=INK2, linewidth=2)
    ax.axhline(0, color=INK2, linewidth=1)
    if r1_parent is not None and np.isfinite(r1_parent[out]):
        color, marker = FAMILY_STYLE[r1_parent["family"]]
        ax.scatter(r1_parent["dbpb_fresh"], r1_parent[out], s=58, facecolor=SURFACE, edgecolor=color,
                   marker=marker, linewidth=1.8, zorder=3)

    labels = []
    for r in d.itertuples():
        color, marker = FAMILY_STYLE[r.family]
        ax.scatter(r.dbpb_fresh, getattr(r, out), s=58, color=color, marker=marker, edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        st = next(s for s in STAGES if s.model == r.model)
        labels.append((scatter_label(st), r.dbpb_fresh, getattr(r, out), INK if st.stage in MATH_STAGES else INK2))
    if r1_parent is not None and np.isfinite(r1_parent[out]):
        labels.append(("R1-Distill vs Math parent (not in fit)", r1_parent["dbpb_fresh"], r1_parent[out], INK2))
    ax.set_xlabel("Δ bits per byte on fresh 2026 text vs family base (%)", color=INK2, fontsize=8.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.set_title(f"Pearson r = {r_row.pearson:+.2f} [{r_row.pearson_lo:+.2f}, {r_row.pearson_hi:+.2f}],  "
                 f"Spearman ρ = {r_row.spearman:+.2f}  (n = {r_row.n})", fontsize=8.5, color=INK, loc="left")
    style_axis(ax)
    return labels


def place_labels(ax: plt.Axes, labels: list[Label]) -> None:
    """Annotate points with non-overlapping labels; call after the figure layout is final."""
    to_pt = 72.0 / ax.figure.dpi
    anchors = [tuple(v * to_pt for v in ax.transData.transform((x, y))) for _, x, y, _ in labels]
    offsets = nudge_labels(anchors, [len(t) * 7 * 0.55 for t, _, _, _ in labels], height=9.0, marker=4.5)
    for (text, x, y, color), (ox, oy) in zip(labels, offsets, strict=True):
        ax.annotate(text, (x, y), xytext=(ox, oy), textcoords="offset points", fontsize=7, color=color,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6} if oy > 8 else None)


def r_panel(ax: plt.Axes, corr: pd.DataFrame) -> None:
    """Pearson r with 95% CI for each predictor, for forgetting and for gsm8k gain."""
    allp = corr[corr.subset == ALL12]
    items = [("forgetting", "forgetting"), ("gain_gsm8k", "gsm8k gain,\nbase prompting")]
    if allp[allp.outcome == "gain_gsm8k_chat"].pearson.notna().any():
        items.insert(1, ("gain_gsm8k_chat", "gsm8k gain,\nchat template"))
    for i, (out, _name) in enumerate(items):
        for dy, pred, color, lab in ((-0.14, "dbpb_fresh", INK, "fresh 2026 text"),
                                     (0.14, "dbpb_enwik8", MUTED, "enwik8 control")):
            r = allp[(allp.predictor == pred) & (allp.outcome == out)].iloc[0]
            ax.errorbar(r.pearson, i + dy, xerr=[[r.pearson - r.pearson_lo], [r.pearson_hi - r.pearson]],
                        fmt="o", color=color, markersize=6, capsize=2, linewidth=1.5,
                        label=f"Pearson r, {lab}" if i == 0 else None)
            ax.scatter(r.spearman, i + dy, s=34, marker="D", facecolor=SURFACE, edgecolor=color, linewidth=1.3,
                       zorder=3, label=f"Spearman ρ, {lab}" if i == 0 else None)
    # rule zones in data coordinates: Gate 1 on the forgetting row, the one-sidedness band on the gain row
    ax.add_patch(Rectangle((GATE_SOFT, -0.45), GATE_PASS - GATE_SOFT, 0.9, color=GRID, linewidth=0))
    ax.add_patch(Rectangle((GATE_PASS, -0.45), 1.02 - GATE_PASS, 0.9, color=GRID, alpha=0.5, linewidth=0))
    ax.add_patch(Rectangle((-ONE_SIDED_MAX, 0.55), 2 * ONE_SIDED_MAX, len(items) - 1 - 0.1, color=GRID, linewidth=0))
    ax.annotate("soft | pass", (GATE_SOFT, -0.45), xytext=(2, 2), textcoords="offset points", fontsize=6.5,
                color=INK2, va="bottom")
    ax.annotate("one-sided zone", (-ONE_SIDED_MAX, 0.55), xytext=(2, 2), textcoords="offset points", fontsize=6.5,
                color=INK2, va="bottom")
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for _, n in items], fontsize=8, color=INK2)
    ax.set_ylim(len(items) - 0.4, -0.6)  # forgetting on top
    ax.set_xlim(-1.02, 1.02)
    ax.set_xlabel("Pearson r (95% CI), Spearman ρ (diamond)", color=INK2, fontsize=8.5)
    ax.set_title("Predictor: fresh text vs enwik8", fontsize=8.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=6.8, labelcolor=INK2, loc="upper left")
    style_axis(ax)


def figure(table: pd.DataFrame, corr: pd.DataFrame, r1_parent: Mapping[str, object] | None,
           floor: float) -> Figure:
    post = post_rows(table)
    allp = corr[(corr.subset == ALL12) & (corr.predictor == "dbpb_fresh")].set_index("outcome")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.0), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1, 0.62]})
    lab0 = scatter_panel(axes[0], post, "forgetting", "forgetting: drop in mean general accuracy vs base (% of base)",
                         allp.loc["forgetting"], r1_parent, floor)
    chat = "gain_gsm8k_chat" in allp.index and np.isfinite(allp.loc["gain_gsm8k_chat"].get("pearson", np.nan))
    gain = "gain_gsm8k_chat" if chat else "gain_gsm8k"
    ylab = ("gain: gsm8k flexible-extract, post with its chat template − base (points)" if chat
            else "gain: gsm8k (flexible-extract) post − base, base prompting (points)")
    lab1 = scatter_panel(axes[1], post, gain, ylab, allp.loc[gain], r1_parent, None)
    r_panel(axes[2], corr)
    handles = [Line2D([], [], color=c, marker=m, linestyle="", markersize=7, label=FAMILY_LABEL[f])
               for f, (c, m) in FAMILY_STYLE.items()]
    handles.append(Line2D([], [], color=INK2, linewidth=2, label="OLS fit, 95% bootstrap band"))
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=8, labelcolor=INK2)
    cond = ("general tasks without chat template; gain with each model's chat template" if chat
            else "no chat template")
    fig.suptitle("Does fresh-text drift predict what post-training loses (left) but not what it gains (middle)? "
                 f"12 post-trained checkpoints; {cond}", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    place_labels(axes[0], lab0)
    place_labels(axes[1], lab1)
    return fig


# ----------------------------------------------------------------------------- effects
def load_inputs(res: Path, exp03: Path, exp01: Path) -> dict[str, object]:
    summaries, chats = {}, {}
    for s in STAGES:
        p = res / "lm_eval" / model_slug(s.model) / "summary.json"
        if p.exists():
            summaries[s.model] = json.loads(p.read_text())
        c = p.with_name("summary_chat.json")
        if c.exists():
            chats[s.model] = json.loads(c.read_text())
    return {"summaries": summaries, "chats": chats, "stage_deltas": pd.read_csv(exp03 / "stage_deltas.csv"),
            "exp03_bpb": pd.read_csv(exp03 / "bpb.csv"), "exp01_bpb": pd.read_csv(exp01 / "bpb.csv")}


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
    scores = pd.DataFrame([score_row(s, d["chats"].get(m)) for m, s in d["summaries"].items()])
    preds = predictor_table(d["stage_deltas"], d["exp01_bpb"])
    table = checkpoint_table(scores, preds)
    r1 = next(s.model for s in STAGES if s.stage == "r1-distill")
    r1_parent = parent_reference_row(table, d["stage_deltas"], r1) if r1 in set(table.model) else None
    corr, slopes = correlation_table(table, r1_parent), family_slopes(table)
    rules, thr = rule_table(corr, table), threshold_reading(post_rows(table))
    cost = cost_table(table, d["exp03_bpb"], d["exp01_bpb"])
    table.to_csv(res / "scores.csv", index=False, float_format="%.5f")
    corr.to_csv(res / "correlations.csv", index=False, float_format="%.4f")
    (res / "summary.md").write_text(summary_markdown(table, corr, slopes, rules, cost, thr, r1_parent,
                                                     harness_meta(d["summaries"])))
    figure(table, corr, r1_parent, thr["median_floor"]).savefig(res / "fig_forgetting.png", dpi=160, facecolor=SURFACE)
    print((res / "summary.md").read_text())


if __name__ == "__main__":
    main()

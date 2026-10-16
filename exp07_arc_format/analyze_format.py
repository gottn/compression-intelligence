"""exp07 analysis: is the ARC "forgetting" of exp04/exp06 knowledge loss or an answer-format effect?

    .venv/bin/python exp07_arc_format/analyze_format.py [--results R]

Reads   results/lm_eval/<label>/summary.json and <folder>/{done.json, items.json}   (run_battery.py on AAU A10s)
        results/train/<recipe>/drift.csv                       (exp07's retraining of exp06 stage 1, AAU A40)
        logs/setup/data/<recipe>.prov.json                     (the re-made data; checked against exp06's)
        ../exp04_benchmarks/results/scores.csv                 (exp04: the forgetting in question, PIQA/WinoGrande)
        ../exp06_matched_drift/results/{checkpoints.csv, train/<recipe>/drift.csv}, logs/setup/data/*.prov.json
Writes  results/scores_long.csv, results/format_table.csv, results/verdicts.csv, results/reproduction.csv,
        results/sanity.csv, results/summary.md, results/fig_format.png
Pure functions above the effects marker; file I/O only below it.

Definitions are the pre-registered ones in RESULTS.md: losses are 100 × (base − checkpoint) points, gains
100 × (checkpoint − base); the base under B3 is the run with the checkpoint's own template; unpaired floors are
1.96 × √(se_b² + se_c²) × 100 (exp04); paired CIs are mean per-item differences ± 1.96 standard errors over the same
documents; forgetting = 100 (g_base − g) / g_base on the 6-task general score and on the means without ARC-E (5 tasks)
and without ARC (4 tasks).
"""
from __future__ import annotations

import argparse
import json
import math
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

from cilab.bench import GENERAL
from cilab.pairs import model_slug
from cilab.stages import STAGES, stage_of
from exp07_arc_format.run_battery import CHAT_GROUP, PUBLIC, RETRAINED_BASE, RETRAINED_TEMPLATE, chat_folder

HERE = Path(__file__).resolve().parent
EXP04_SCORES = HERE.parent / "exp04_benchmarks" / "results" / "scores.csv"
EXP06 = HERE.parent / "exp06_matched_drift"
Z95, K_MIN, F_MAX = 1.96, 5.0, 2.2  # z; Rule 1 thresholds, ARC-E acc_norm points
FORMATS: tuple[tuple[str, str], ...] = (("b1", "b1_arc_0shot"), ("b2", "b2_arc_25shot"), ("b3", CHAT_GROUP))
METRICS: tuple[tuple[str, str], ...] = (("norm", "acc_norm"), ("acc", "acc"))
ARC: tuple[tuple[str, str], ...] = (("arce", "arc_easy"), ("arcc", "arc_challenge"))
MMLU: tuple[tuple[str, str, str], ...] = (("letter", "b4_mmlu_letter", "mmlu"), ("cont", "b4_mmlu_cont", "mmlu_continuation"))
RULE2_STAGES: tuple[str, ...] = ("sft", "dpo", "rlvr", "instruct")  # exp04's chat-style checkpoints
RULE1_PUBLIC = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
RETRAINED_RECIPES: tuple[str, ...] = ("tulu", "mathcot")
TARGET_STEP = 1500
SIGMA_WINDOW: tuple[int, int] = (760, 1480)  # scored updates whose exp07 − exp06 drift difference gives σ_run
PRIMARY_SOURCE: dict[str, tuple[str, str]] = {  # retrained checkpoints: battery group and metric of exp04's 6 tasks
    "arc_easy": ("b1_arc_0shot", "acc_norm"), "arc_challenge": ("b1_arc_0shot", "acc_norm"),
    "hellaswag": ("b5_hellaswag", "acc_norm"), "piqa": ("b0_piqa_wino", "acc_norm"),
    "winogrande": ("b0_piqa_wino", "acc"), "mmlu": ("b4_mmlu_letter", "acc"),
}
NO_ARCE: tuple[str, ...] = ("arc_challenge", "hellaswag", "piqa", "winogrande", "mmlu")
NO_ARC: tuple[str, ...] = ("hellaswag", "piqa", "winogrande", "mmlu")
SANITY: tuple[tuple[str, str, str], ...] = (  # (exp04 column, battery folder, metric) for the B1/B4/B5 re-run check
    ("arc_easy", "b1_arc_0shot", "acc_norm"), ("arc_challenge", "b1_arc_0shot", "acc_norm"),
    ("mmlu", "b4_mmlu_letter", "acc"), ("hellaswag", "b5_hellaswag", "acc_norm"),
)
PROV_FIELDS: tuple[str, ...] = ("dataset", "config", "revision", "kept", "candidates_scanned", "drops")
# dataviz reference palette, categorical slots 1-4 (validated, light surface); the exp04 default format is slot 1
FORMAT_COLOR = {"b1": "#2a78d6", "b2": "#eb6834", "b3": "#1baf7a", "letter": "#2a78d6", "cont": "#eda100"}
FORMAT_NAME = {"b1": "B1 0-shot (exp04 default)", "b2": "B2 25-shot", "b3": "B3 0-shot, chat template",
               "letter": "letter choice (exp04 default)", "cont": "answer-text continuation"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------- pure: scores
def scores_long(done: Mapping[tuple[str, str], Mapping[str, object]]) -> pd.DataFrame:
    """One row per (label, folder, task, metric) of every finished group: value, standard error, documents."""
    rows = []
    for (label, folder), rec in sorted(done.items()):
        if rec.get("status") != "ok":
            continue
        for task, t in sorted(rec["tasks"].items()):
            m = t["metrics"]
            for key in sorted(m):
                name, filt = key.split(",", 1)
                if name.endswith("_stderr") or filt != "none":
                    continue
                rows.append({"label": label, "folder": folder, "task": task, "metric": name, "value": m[key],
                             "se": m.get(f"{name}_stderr,none", np.nan), "n": t.get("n_effective")})
    return pd.DataFrame(rows, columns=["label", "folder", "task", "metric", "value", "se", "n"])


def score_index(sl: pd.DataFrame) -> dict[tuple[str, str, str, str], tuple[float, float]]:
    return {(r.label, r.folder, r.task, r.metric): (float(r.value), float(r.se)) for r in sl.itertuples()}


NAN2 = (math.nan, math.nan)


def diff_pts(base: tuple[float, float], ckpt: tuple[float, float], sign: int) -> tuple[float, float]:
    """(100 × sign × (base − ckpt), unpaired 95% floor in points); sign +1 gives a loss, −1 a gain."""
    return 100 * sign * (base[0] - ckpt[0]), 100 * Z95 * math.sqrt(base[1] ** 2 + ckpt[1] ** 2)


def leaves(items: Mapping[str, object], task: str) -> list[str]:
    """Leaf tasks of `task` in one group's items (MMLU: every subject leaf of the group)."""
    if task in ("mmlu", "mmlu_continuation"):
        return sorted(items)
    return [task] if task in items else []


def paired_ci(base_items: Mapping[str, Mapping[str, object]] | None,
              ckpt_items: Mapping[str, Mapping[str, object]] | None, task: str, metric: str,
              sign: int) -> tuple[float, float, float, int]:
    """(mean, lo, hi, n) of 100 × sign × (base − ckpt) per item, over documents both runs scored, with a normal
    95% interval; NaNs when either side has no items."""
    if not base_items or not ckpt_items:
        return math.nan, math.nan, math.nan, 0
    d: list[np.ndarray] = []
    for leaf in leaves(ckpt_items, task):
        b, c = base_items.get(leaf), ckpt_items.get(leaf)
        if not b or not c or metric not in b or metric not in c:
            continue
        bv = dict(zip(b["doc_id"], b[metric], strict=True))
        cv = dict(zip(c["doc_id"], c[metric], strict=True))
        common = sorted(set(bv) & set(cv))
        d.append(np.array([int(bv[k]) - int(cv[k]) for k in common], dtype=float))
    if not d or not sum(len(x) for x in d):
        return math.nan, math.nan, math.nan, 0
    x = 100 * sign * np.concatenate(d)
    se = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else math.nan
    m = float(x.mean())
    return m, m - Z95 * se, m + Z95 * se, len(x)


def mean_se(parts: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Mean of task accuracies and its standard error (independent tasks, exp04's general_se)."""
    v = [p[0] for p in parts]
    s = [p[1] for p in parts]
    return float(np.mean(v)), float(math.sqrt(sum(x**2 for x in s)) / len(s))


def rel_forgetting(base: tuple[float, float], post: tuple[float, float]) -> tuple[float, float]:
    """(forgetting % of the base, unpaired 95% floor %), exp04's definition."""
    return 100 * (base[0] - post[0]) / base[0], 100 * Z95 * math.sqrt(base[1] ** 2 + post[1] ** 2) / base[0]


def exp04_parts(row: Mapping[str, object]) -> dict[str, tuple[float, float]]:
    """{task: (primary accuracy, se)} of an exp04 scores.csv row."""
    return {t: (float(row[t]), float(row[f"{t}_se"])) for t, _, _ in GENERAL}


def battery_parts(idx: Mapping[tuple[str, str, str, str], tuple[float, float]],
                  label: str) -> dict[str, tuple[float, float]]:
    """{task: (primary accuracy, se)} of a retrained checkpoint, from its battery groups (exp04's primary metrics)."""
    return {t: idx.get((label, folder, t, metric), NAN2) for t, (folder, metric) in PRIMARY_SOURCE.items()}


def forgetting_cols(base: Mapping[str, tuple[float, float]], post: Mapping[str, tuple[float, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, tasks in (("forget6", tuple(t for t, _, _ in GENERAL)), ("forget5_no_arce", NO_ARCE),
                        ("forget4_no_arc", NO_ARC)):
        f, floor = rel_forgetting(mean_se([base[t] for t in tasks]), mean_se([post[t] for t in tasks]))
        out[name], out[f"{name}_floor"] = f, floor
    out["general"] = mean_se([post[t] for t, _, _ in GENERAL])[0]
    out["general_se"] = mean_se([post[t] for t, _, _ in GENERAL])[1]
    return out


# ----------------------------------------------------------------------------- pure: the per-checkpoint table
def base_label_of(entry: Mapping[str, object]) -> str:
    return model_slug(str(entry["base"]))


def is_retrained(entry: Mapping[str, object]) -> bool:
    return entry["recipe"] != PUBLIC


def b3_reference(entry: Mapping[str, object]) -> str:
    return chat_folder(RETRAINED_TEMPLATE if is_retrained(entry) else str(entry["model"]))


def display_name(entry: Mapping[str, object]) -> str:
    if is_retrained(entry):
        return f"{entry['recipe']} @{int(entry['step'])} (exp07)"
    return str(entry["model"]).split("/", 1)[1]


def stage_name(entry: Mapping[str, object]) -> str:
    return "sft (exp07)" if is_retrained(entry) else stage_of(str(entry["model"])).stage


def rule1_applies(entry: Mapping[str, object]) -> bool:
    return entry["model"] == RULE1_PUBLIC or (entry["recipe"] == "mathcot" and int(entry["step"]) == TARGET_STEP)


def rule2_applies(entry: Mapping[str, object]) -> bool:
    if is_retrained(entry):
        return entry["recipe"] == "tulu" and int(entry["step"]) == TARGET_STEP
    return stage_of(str(entry["model"])).stage in RULE2_STAGES


def rule1_verdict(loss_b2: float, loss_b3: float, k_min: float = K_MIN, f_max: float = F_MAX) -> str:
    """K if the ARC-E loss is ≥ k_min under both B2 and B3, F if ≤ f_max under either, else mixed."""
    if math.isnan(loss_b2) or math.isnan(loss_b3):
        return "missing"
    if loss_b2 >= k_min and loss_b3 >= k_min:
        return "K"
    if loss_b2 <= f_max or loss_b3 <= f_max:
        return "F"
    return "mixed"


def rule2_verdict(letter: float, letter_floor: float, cont: float, cont_floor: float) -> str:
    """format: the letter gain exceeds its floor and the continuation gain does not; survives: both do; no gain."""
    if any(math.isnan(x) for x in (letter, letter_floor, cont, cont_floor)):
        return "missing"
    if letter <= letter_floor:
        return "no gain"
    return "survives" if cont > cont_floor else "format"


def checkpoint_row(entry: Mapping[str, object], idx: Mapping[tuple[str, str, str, str], tuple[float, float]],
                   items: Mapping[tuple[str, str], Mapping[str, Mapping[str, object]]],
                   templates: Mapping[tuple[str, str], str | None],
                   truncation: Mapping[tuple[str, str], int], exp04: pd.DataFrame) -> dict[str, object]:
    """Every pre-registered number of one post-trained or retrained checkpoint against its family base."""
    lab, base = str(entry["label"]), base_label_of(entry)
    row: dict[str, object] = {"label": lab, "name": display_name(entry), "stage": stage_name(entry),
                              "recipe": entry["recipe"], "step": int(entry["step"]), "base": base}
    for fk, folder in FORMATS:
        bfolder = b3_reference(entry) if fk == "b3" else folder
        for ak, task in ARC:
            for mk, metric in METRICS:
                b, c = idx.get((base, bfolder, task, metric), NAN2), idx.get((lab, folder, task, metric), NAN2)
                loss, floor = diff_pts(b, c, +1)
                col = f"{ak}_loss_{fk}_{mk}"
                row[col], row[f"{col}_floor"] = loss, floor
                m, lo, hi, n = paired_ci(items.get((base, bfolder)), items.get((lab, folder)), task, metric, +1)
                row[f"{col}_pmean"], row[f"{col}_plo"], row[f"{col}_phi"], row[f"{col}_pn"] = m, lo, hi, n
            row[f"{ak}_normgap_{fk}"] = 100 * (idx.get((lab, folder, task, "acc_norm"), NAN2)[0]
                                               - idx.get((lab, folder, task, "acc"), NAN2)[0])
            row[f"{ak}_normgap_{fk}_base"] = 100 * (idx.get((base, bfolder, task, "acc_norm"), NAN2)[0]
                                                    - idx.get((base, bfolder, task, "acc"), NAN2)[0])
            row[f"{ak}_amp_{fk}"] = row[f"{ak}_loss_{fk}_norm"] - row[f"{ak}_loss_{fk}_acc"]
    for mk, folder, task in MMLU:
        b, c = idx.get((base, folder, task, "acc"), NAN2), idx.get((lab, folder, task, "acc"), NAN2)
        gain, floor = diff_pts(b, c, -1)
        row[f"mmlu_gain_{mk}"], row[f"mmlu_gain_{mk}_floor"] = gain, floor
        _, lo, hi, _ = paired_ci(items.get((base, folder)), items.get((lab, folder)), task, "acc", -1)
        row[f"mmlu_gain_{mk}_plo"], row[f"mmlu_gain_{mk}_phi"] = lo, hi
    letter = row["mmlu_gain_letter"]
    row["mmlu_cont_over_letter"] = row["mmlu_gain_cont"] / letter if letter and not math.isnan(letter) else math.nan
    for mk, metric in METRICS:
        b, c = idx.get((base, "b5_hellaswag", "hellaswag", metric), NAN2), idx.get((lab, "b5_hellaswag", "hellaswag", metric), NAN2)
        row[f"hs_loss_{mk}"], row[f"hs_loss_{mk}_floor"] = diff_pts(b, c, +1)
    base_row = exp04[exp04.model == RETRAINED_BASE if is_retrained(entry) else exp04.model == entry["base"]]
    post_parts = battery_parts(idx, lab) if is_retrained(entry) else exp04_parts(exp04[exp04.model == entry["model"]].iloc[0])
    row.update(forgetting_cols(exp04_parts(base_row.iloc[0]), post_parts))
    row["template_md5"], row["template_md5_base"] = templates.get((lab, CHAT_GROUP)), templates.get((base, b3_reference(entry)))
    row["template_matched"] = bool(row["template_md5"]) and row["template_md5"] == row["template_md5_base"]
    row["b2_truncated_docs"] = max(truncation.get((lab, "b2_arc_25shot"), 0), truncation.get((base, "b2_arc_25shot"), 0))
    row["rule1"] = rule1_verdict(row["arce_loss_b2_norm"], row["arce_loss_b3_norm"]) if rule1_applies(entry) else ""
    row["rule1_acc"] = rule1_verdict(row["arce_loss_b2_acc"], row["arce_loss_b3_acc"]) if rule1_applies(entry) else ""
    row["rule2"] = (rule2_verdict(letter, row["mmlu_gain_letter_floor"], row["mmlu_gain_cont"],
                                  row["mmlu_gain_cont_floor"]) if rule2_applies(entry) else "")
    return row


def entry_order(entry: Mapping[str, object]) -> tuple[int, str, int]:
    if is_retrained(entry):
        return len(STAGES), str(entry["recipe"]), int(entry["step"])
    return [s.model for s in STAGES].index(str(entry["model"])), "", 0


def format_table(entries: Sequence[Mapping[str, object]], idx: Mapping[tuple[str, str, str, str], tuple[float, float]],
                 items: Mapping[tuple[str, str], Mapping[str, Mapping[str, object]]],
                 templates: Mapping[tuple[str, str], str | None], truncation: Mapping[tuple[str, str], int],
                 exp04: pd.DataFrame) -> pd.DataFrame:
    """One row per post-trained or retrained checkpoint, in stage-table order, retrained last."""
    rows = [checkpoint_row(e, idx, items, templates, truncation, exp04)
            for e in sorted(entries, key=entry_order) if not e["is_base"]]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- pure: reproduction and sanity
def drift_difference(d07: pd.DataFrame, d06: pd.DataFrame) -> pd.DataFrame:
    """Δbpb_fresh of both runs at the updates both scored, and d = exp07 − exp06 (pp)."""
    m = d07[["step", "dbpb_fresh"]].merge(d06[["step", "dbpb_fresh"]], on="step", suffixes=("_exp07", "_exp06"))
    return m.assign(d=m.dbpb_fresh_exp07 - m.dbpb_fresh_exp06).sort_values("step").reset_index(drop=True)


def reproduction_row(recipe: str, diff: pd.DataFrame, g07: tuple[float, float], g06: tuple[float, float],
                     g_base: float, window: tuple[int, int] = SIGMA_WINDOW, step: int = TARGET_STEP) -> dict[str, object]:
    """Rule 3: |d at `step`| ≤ 2 σ_run (RMS of d over the window) and the general score inside the unpaired floor."""
    w = diff[(diff.step >= window[0]) & (diff.step <= window[1])]
    sigma = float(np.sqrt((w.d**2).mean())) if len(w) else math.nan
    at = diff[diff.step == step]
    d = float(at.d.iloc[0]) if len(at) else math.nan
    gd = 100 * (g07[0] - g06[0]) / g_base
    gfloor = 100 * Z95 * math.sqrt(g07[1] ** 2 + g06[1] ** 2) / g_base
    ok_drift = bool(abs(d) <= 2 * sigma) if not (math.isnan(d) or math.isnan(sigma)) else False
    ok_general = bool(abs(gd) < gfloor) if not math.isnan(gd) else False
    return {"recipe": recipe, "dbpb_exp06": float(at.dbpb_fresh_exp06.iloc[0]) if len(at) else math.nan,
            "dbpb_exp07": float(at.dbpb_fresh_exp07.iloc[0]) if len(at) else math.nan, "d_1500": d,
            "sigma_run": sigma, "n_sigma": len(w), "max_abs_d": float(diff.d.abs().max()) if len(diff) else math.nan,
            "r_trajectory": float(np.corrcoef(diff.dbpb_fresh_exp07, diff.dbpb_fresh_exp06)[0, 1]) if len(diff) > 2
            else math.nan, "general_exp07": g07[0], "general_exp06": g06[0], "general_diff_pct": gd,
            "general_floor_pct": gfloor, "ok_drift": ok_drift, "ok_general": ok_general,
            "reproduced": ok_drift and ok_general}


def prov_check(new: Mapping[str, object], old: Mapping[str, object]) -> dict[str, object]:
    """Field-by-field equality of a re-made subset's provenance with exp06's (plus token and loss-token sums)."""
    fields = {f: new.get(f) == old.get(f) for f in PROV_FIELDS}
    fields["tokens_sum"] = new["tokens"]["sum"] == old["tokens"]["sum"]
    fields["loss_tokens_sum"] = new["loss_tokens"]["sum"] == old["loss_tokens"]["sum"]
    return {"recipe": new.get("recipe"), **fields, "identical": all(fields.values())}


def sanity_table(idx: Mapping[tuple[str, str, str, str], tuple[float, float]], exp04: pd.DataFrame,
                 labels: Sequence[str]) -> pd.DataFrame:
    """Battery re-run of exp04's settings (B1, B4-letter, B5) minus exp04's scores, in points, per public checkpoint."""
    rows = []
    for r in exp04.itertuples():
        lab = model_slug(r.model)
        if lab not in labels:
            continue
        for col, folder, metric in SANITY:
            v = idx.get((lab, folder, col, metric), NAN2)[0]
            rows.append({"label": lab, "task": col, "exp04": getattr(r, col), "exp07": v,
                         "diff_pts": 100 * (v - getattr(r, col))})
    return pd.DataFrame(rows, columns=["label", "task", "exp04", "exp07", "diff_pts"])


def expectations(table: pd.DataFrame, repro: pd.DataFrame) -> pd.DataFrame:
    """E1–E3 as scored in RESULTS.md."""
    rows = []
    mc = table[(table.recipe == "mathcot") & (table.step == TARGET_STEP)]
    if len(mc):
        b2, b3 = float(mc.arce_loss_b2_norm.iloc[0]), float(mc.arce_loss_b3_norm.iloc[0])
        rows.append({"id": "E1", "expected": "mathcot: B2 loss ≥ 5 and 2.2 < B3 loss < 5 (mixed)",
                     "observed": f"B2 {b2:+.2f}, B3 {b3:+.2f} ({mc.rule1.iloc[0]})",
                     "met": bool(b2 >= K_MIN and F_MAX < b3 < K_MIN)})
    tu = table[(table.recipe == "tulu") & (table.step == TARGET_STEP)]
    if len(tu):
        lg, cg = float(tu.mmlu_gain_letter.iloc[0]), float(tu.mmlu_gain_cont.iloc[0])
        rows.append({"id": "E2", "expected": "tulu: MMLU continuation gain ≤ 0.5 × letter gain",
                     "observed": f"letter {lg:+.2f}, continuation {cg:+.2f}", "met": bool(lg > 0 and cg <= 0.5 * lg)})
    if len(repro):
        rows.append({"id": "E3", "expected": "both retrained recipes reproduce exp06 (Rule 3)",
                     "observed": ", ".join(f"{r.recipe} {'yes' if r.reproduced else 'no'}" for r in repro.itertuples()),
                     "met": bool(repro.reproduced.all() and len(repro) == len(RETRAINED_RECIPES))})
    return pd.DataFrame(rows, columns=["id", "expected", "observed", "met"])


def verdict_table(table: pd.DataFrame) -> pd.DataFrame:
    cols = ["name", "arce_loss_b1_norm", "arce_loss_b2_norm", "arce_loss_b3_norm", "arce_loss_b2_acc",
            "arce_loss_b3_acc", "rule1", "rule1_acc", "mmlu_gain_letter", "mmlu_gain_letter_floor", "mmlu_gain_cont",
            "mmlu_gain_cont_floor", "mmlu_cont_over_letter", "rule2"]
    return table[(table.rule1 != "") | (table.rule2 != "")][cols].reset_index(drop=True)


# ----------------------------------------------------------------------------- pure: report
def fmt(v: object, digits: int = 1, sign: bool = True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, bool | str):
        return str(v)
    return f"{float(v):+.{digits}f}" if sign else f"{float(v):.{digits}f}"


def summary_markdown(table: pd.DataFrame, repro: pd.DataFrame, prov: pd.DataFrame, sanity: pd.DataFrame,
                     exps: pd.DataFrame) -> str:
    out = ["# exp07 summary (auto-generated by analyze_format.py)", "",
           "Losses are base − checkpoint and gains checkpoint − base, in accuracy points; forgetting is % of the base.",
           "", "## Per-checkpoint table", "",
           "| checkpoint | stage | ARC-E B1 acc_norm | B1 acc | B2 25-shot | B3 chat | MMLU letter gain | MMLU continuation "
           "gain | HellaSwag loss | forgetting 6-task | 5-task no ARC-E | 4-task no ARC | Rule 1 | Rule 2 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        out.append(f"| {r.name} | {r.stage} | {fmt(r.arce_loss_b1_norm)} | {fmt(r.arce_loss_b1_acc)} | "
                   f"{fmt(r.arce_loss_b2_norm)} | {fmt(r.arce_loss_b3_norm)} | {fmt(r.mmlu_gain_letter)} | "
                   f"{fmt(r.mmlu_gain_cont)} | {fmt(r.hs_loss_norm)} | {fmt(r.forget6)} ({fmt(r.forget6_floor, sign=False)}) | "
                   f"{fmt(r.forget5_no_arce)} ({fmt(r.forget5_no_arce_floor, sign=False)}) | "
                   f"{fmt(r.forget4_no_arc)} ({fmt(r.forget4_no_arc_floor, sign=False)}) | {r.rule1} | {r.rule2} |")
    out += ["", "Floors in brackets. B3 compares with the base under the same template.", "",
            "## ARC-E paired 95% CIs and normalisation", "",
            "| checkpoint | B1 loss [paired CI] | B2 loss [paired CI] | B3 loss [paired CI] | norm gap B1 (ckpt / base) | "
            "amplification B1 / B2 / B3 | template matched | B2 truncated docs |", "|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        cis = " | ".join(f"{fmt(getattr(r, f'arce_loss_{f}_norm'))} [{fmt(getattr(r, f'arce_loss_{f}_norm_plo'))}, "
                         f"{fmt(getattr(r, f'arce_loss_{f}_norm_phi'))}]" for f, _ in FORMATS)
        out.append(f"| {r.name} | {cis} | {fmt(r.arce_normgap_b1)} / {fmt(r.arce_normgap_b1_base)} | "
                   f"{fmt(r.arce_amp_b1)} / {fmt(r.arce_amp_b2)} / {fmt(r.arce_amp_b3)} | {r.template_matched} | "
                   f"{r.b2_truncated_docs} |")
    out += ["", "## ARC-C (reported, not a rule)", "", "| checkpoint | B1 | B2 | B3 |", "|---|---|---|---|"]
    for r in table.itertuples():
        out.append(f"| {r.name} | {fmt(r.arcc_loss_b1_norm)} | {fmt(r.arcc_loss_b2_norm)} | {fmt(r.arcc_loss_b3_norm)} |")
    out += ["", "## Rule 2 detail (MMLU)", "",
            "| checkpoint | letter gain (floor) [paired CI] | continuation gain (floor) [paired CI] | ratio | Rule 2 |",
            "|---|---|---|---|---|"]
    for r in table.itertuples():
        out.append(f"| {r.name} | {fmt(r.mmlu_gain_letter)} ({fmt(r.mmlu_gain_letter_floor, sign=False)}) "
                   f"[{fmt(r.mmlu_gain_letter_plo)}, {fmt(r.mmlu_gain_letter_phi)}] | {fmt(r.mmlu_gain_cont)} "
                   f"({fmt(r.mmlu_gain_cont_floor, sign=False)}) [{fmt(r.mmlu_gain_cont_plo)}, {fmt(r.mmlu_gain_cont_phi)}] | "
                   f"{fmt(r.mmlu_cont_over_letter, 2, sign=False)} | {r.rule2 or '–'} |")
    out += ["", "## Reproduction of exp06 stage 1 (Rule 3)", ""]
    if len(repro):
        out += ["| recipe | Δbpb exp06 | Δbpb exp07 | d at 1500 | σ_run (n) | max \\|d\\| | r trajectory | general exp06 → "
                "exp07 | diff % (floor) | reproduced |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in repro.itertuples():
            out.append(f"| {r.recipe} | {fmt(r.dbpb_exp06, 3)} | {fmt(r.dbpb_exp07, 3)} | {fmt(r.d_1500, 3)} | "
                       f"{fmt(r.sigma_run, 3, sign=False)} ({r.n_sigma}) | {fmt(r.max_abs_d, 3, sign=False)} | "
                       f"{fmt(r.r_trajectory, 4, sign=False)} | {r.general_exp06:.4f} → {r.general_exp07:.4f} | "
                       f"{fmt(r.general_diff_pct, 2)} ({fmt(r.general_floor_pct, 2, sign=False)}) | {r.reproduced} |")
    else:
        out.append("No retraining logs.")
    if len(prov):
        out += ["", "Re-made data vs exp06 provenance: " + "; ".join(
            f"{r.recipe} {'identical' if r.identical else 'DIFFERS'}" for r in prov.itertuples())]
    out += ["", "## Sanity: battery re-run of exp04's settings", ""]
    if len(sanity):
        g = sanity.groupby("task").diff_pts.agg(lambda x: float(np.nanmax(np.abs(x))))
        out.append("Max |exp07 − exp04| over the public checkpoints, points: " +
                   ", ".join(f"{t} {v:.2f}" for t, v in sorted(g.items())))
    out += ["", "## Expectations", "", "| id | expected | observed | met |", "|---|---|---|---|"]
    out += [f"| {r.id} | {r.expected} | {r.observed} | {r.met} |" for r in exps.itertuples()]
    return "\n".join(out) + "\n"


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def hbar_panel(ax: plt.Axes, names: Sequence[str], series: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
               lines: Sequence[tuple[float, str]]) -> None:
    """Horizontal grouped bars (one group per checkpoint, top to bottom) with paired-CI whiskers and reference lines."""
    n, k = len(names), len(series)
    h = 0.8 / k
    y = np.arange(n)[::-1].astype(float)
    for j, (key, val, lo, hi) in enumerate(series):
        yy = y + 0.4 - h * (j + 0.5)
        ax.barh(yy, val, height=h * 0.86, color=FORMAT_COLOR[key], label=FORMAT_NAME[key], zorder=2)
        err = np.vstack([np.clip(val - lo, 0, None), np.clip(hi - val, 0, None)])
        ok = ~np.isnan(lo)
        ax.errorbar(val[ok], yy[ok], xerr=err[:, ok], fmt="none", ecolor=INK, elinewidth=0.7, capsize=0, zorder=3)
    for x, label in lines:
        ax.axvline(x, color=MUTED if x else INK, linewidth=0.8 if x else 0.9, linestyle=(0, (3, 3)) if x else "-",
                   zorder=1)
        if x:
            ax.text(x, n - 0.35, label, color=MUTED, fontsize=7, ha="center", va="bottom")
    ax.set_yticks(y, names, fontsize=8, color=INK)
    ax.set_ylim(-0.6, n + 0.1)
    style_axis(ax)


def figure(table: pd.DataFrame) -> Figure:
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 7.2), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    t = table.reset_index(drop=True)
    names = [f"{n}  ◆" if r1 else n for n, r1 in zip(t.name, t.rule1, strict=True)]
    hbar_panel(a, names, [(f, t[f"arce_loss_{f}_norm"].to_numpy(float), t[f"arce_loss_{f}_norm_plo"].to_numpy(float),
                           t[f"arce_loss_{f}_norm_phi"].to_numpy(float)) for f, _ in FORMATS],
               [(0.0, ""), (F_MAX, "2.2 (F)"), (K_MIN, "5 (K)")])
    a.set_xlabel("ARC-Easy loss vs base, acc_norm points (positive = loss); whiskers: paired 95% CI",
                 fontsize=8, color=MUTED)
    a.set_title("ARC-Easy loss under three answer formats  (◆ Rule 1 checkpoints)", fontsize=10, color=INK, loc="left")
    a.legend(fontsize=7.5, frameon=False, loc="lower right")
    c = t[t.rule2 != ""].reset_index(drop=True)
    hbar_panel(b, list(c.name), [(m, c[f"mmlu_gain_{m}"].to_numpy(float), c[f"mmlu_gain_{m}_plo"].to_numpy(float),
                                  c[f"mmlu_gain_{m}_phi"].to_numpy(float)) for m, _, _ in MMLU], [(0.0, "")])
    b.set_xlabel("MMLU gain vs base, acc points (40 per subject); whiskers: paired 95% CI", fontsize=8, color=MUTED)
    b.set_title("Chat-style checkpoints: MMLU gain by answer format", fontsize=10, color=INK, loc="left")
    b.legend(fontsize=7.5, frameon=False, loc="lower right")
    for ax in (a, b):
        ax.set_facecolor("#fcfcfb")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- effects
def read_runs(lm_dir: Path) -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]],
                                     dict[tuple[str, str], dict[str, object]]]:
    """(entries from summary.json, done.json records, items.json contents), keyed by (label, folder)."""
    entries, done, items = [], {}, {}
    for d in sorted(p for p in lm_dir.iterdir() if p.is_dir()):
        s = d / "summary.json"
        if s.exists():
            entries.append({**json.loads(s.read_text())["entry"]})
        for g in sorted(p for p in d.iterdir() if p.is_dir()):
            if (g / "done.json").exists():
                done[(d.name, g.name)] = json.loads((g / "done.json").read_text())
            if (g / "items.json").exists():
                items[(d.name, g.name)] = json.loads((g / "items.json").read_text())
    return entries, done, items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(HERE / "results"))
    args = ap.parse_args()
    res = Path(args.results)
    entries, done, items = read_runs(res / "lm_eval")
    sl = scores_long(done)
    idx = score_index(sl)
    templates = {k: v.get("template_md5") for k, v in done.items()}
    trunc = {k: max((t.get("n_over", 0) for t in v.get("truncation", {}).values()), default=0) for k, v in done.items()}
    exp04 = pd.read_csv(EXP04_SCORES)
    table = format_table(entries, idx, items, templates, trunc, exp04)
    exp06 = pd.read_csv(EXP06 / "results" / "checkpoints.csv")
    base_g = float(exp04[exp04.model == RETRAINED_BASE].general.iloc[0])
    repro_rows, prov_rows = [], []
    for r in RETRAINED_RECIPES:
        d07 = res / "train" / r / "drift.csv"
        lab = table[(table.recipe == r) & (table.step == TARGET_STEP)]
        if d07.exists() and len(lab):
            e6 = exp06[(exp06.recipe == r) & (exp06.step == TARGET_STEP)].iloc[0]
            diff = drift_difference(pd.read_csv(d07), pd.read_csv(EXP06 / "results" / "train" / r / "drift.csv"))
            repro_rows.append(reproduction_row(r, diff, (float(lab.general.iloc[0]), float(lab.general_se.iloc[0])),
                                               (float(e6.general), float(e6.general_se)), base_g))
        new = HERE / "logs" / "setup" / "data" / f"{r}.prov.json"
        if new.exists():
            old = json.loads((EXP06 / "logs" / "setup" / "data" / f"{r}.prov.json").read_text())
            prov_rows.append(prov_check(json.loads(new.read_text()), old))
    repro, prov = pd.DataFrame(repro_rows), pd.DataFrame(prov_rows)
    sanity = sanity_table(idx, exp04, sorted({lab for lab, _ in done}))
    exps = expectations(table, repro)
    sl.to_csv(res / "scores_long.csv", index=False, float_format="%.6f")
    table.to_csv(res / "format_table.csv", index=False, float_format="%.5f")
    verdict_table(table).to_csv(res / "verdicts.csv", index=False, float_format="%.5f")
    repro.to_csv(res / "reproduction.csv", index=False, float_format="%.5f")
    sanity.to_csv(res / "sanity.csv", index=False, float_format="%.5f")
    (res / "summary.md").write_text(summary_markdown(table, repro, prov, sanity, exps))
    fig = figure(table)
    fig.savefig(res / "fig_format.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print((res / "summary.md").read_text())


if __name__ == "__main__":
    main()

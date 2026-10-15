"""exp06 analysis: at matched drift, does the knowledge − style contrast predict forgetting and ignore the gain?

    .venv/bin/python exp06_matched_drift/analyze_matched.py [--results R]

Reads   results/train/<recipe>[-attemptK]/{drift.csv, done.json, base_check.json, train.csv}   (train_sft.py, AAU A40)
        results/train/checkpoints.txt                                    (one line per saved checkpoint)
        results/lm_eval/<label>/summary.json                             (run_eval.py on AAU A10; label exp06__base
                                                                          is the base with the Instruct template)
        ../exp04_benchmarks/results/scores.csv                           (the base's general score, exp04)
Writes  results/checkpoints.csv, results/trajectories.csv, results/correlations.csv, results/tests.csv,
        results/summary.md, results/fig_matched.png
Pure functions above the effects marker; file I/O only below it.

Definitions are the pre-registered ones in RESULTS.md: forgetting = 100 (g_base − g) / g_base with g_base = exp04's
base general score; IFEval gain = 100 × (prompt-level strict, chat template, checkpoint − base with the same template);
primary set = the mark checkpoints (first scored update at or above 3/6/10% Δbpb_fresh); Pearson r with 10,000
checkpoint-bootstrap resamples (seed 0), Spearman ρ, leave-one-recipe-out r.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cilab`

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from cilab.bench import general_scores, score_row
from cilab.sft import parse_checkpoint_line
from cilab.stats import bootstrap_corr, leave_one_out, ols_line, pearson, spearman

HERE = Path(__file__).resolve().parent
EXP04_SCORES = HERE.parent / "exp04_benchmarks" / "results" / "scores.csv"
BASE_MODEL = "allenai/OLMo-2-0425-1B"
BASE_LABEL = "exp06__base"
RECIPES: tuple[str, ...] = ("tulu", "smoltalk", "mathcot")
MARKS: tuple[str, ...] = ("3", "6", "10")
N_BOOT, SEED, Z95 = 10_000, 0, 1.96
H1_PASS, H1_SOFT, H2_MAX, FAIL_R, MIN_N = 0.8, 0.6, 0.3, 0.9, 6
H4_ORDER: tuple[str, ...] = ("mathcot", "smoltalk", "tulu")  # expected descending contrast and forgetting at 6%
MARK_SET, ALL_SET = "mark checkpoints", "mark checkpoints + finals"
CONTRASTS: tuple[str, ...] = ("contrast_ks", "contrast_rf")
OUTCOMES: tuple[str, ...] = ("forgetting", "gain_ifeval", "gain_gsm8k", "arc_loss", "mmlu_loss")
# dataviz reference palette, categorical slots 1-3 (validated all-pairs, light mode); marks by marker shape
RECIPE_COLOR = {"tulu": "#2a78d6", "smoltalk": "#eb6834", "mathcot": "#1baf7a"}
MARK_SHAPE = {"3": "o", "6": "s", "10": "D", "final": "^"}
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"


# ----------------------------------------------------------------------------- pure
def eval_scores(summary: Mapping[str, object]) -> dict[str, object]:
    """exp04's score row from one exp06 summary.json: loglik as the main groups, gsm8k_chat/ifeval_chat as chat."""
    groups = summary["groups"]
    main = {"model": summary["model"], "gpu": summary.get("gpu", ""),
            "groups": {g: r for g, r in groups.items() if not g.endswith("_chat")}}
    chat = {"groups": {g: r for g, r in groups.items() if g.endswith("_chat")}}
    return score_row(main, chat)


def mark_tags(marks: str) -> list[str]:
    return [] if marks in ("", "-") else marks.split(",")


def checkpoint_table(lines: Sequence[str], drift: pd.DataFrame, evals: Mapping[str, Mapping[str, object]],
                     base: Mapping[str, float], base_chat: Mapping[str, float]) -> pd.DataFrame:
    """One row per checkpoint: its drift-log row, marks, benchmark scores, forgetting (vs exp04's base general) with
    its noise floor, and gains against the base evaluated with the same chat template."""
    sc = general_scores(pd.DataFrame([{**eval_scores(s), "label": lab} for lab, s in sorted(evals.items())]))
    sc = sc.set_index("label")
    rows = []
    for line in lines:
        c = parse_checkpoint_line(line)
        step = int(c["step"])
        d = drift[(drift.recipe == c["recipe"]) & (drift.step == step)]
        if len(d) != 1:
            raise ValueError(f"{c['label']}: {len(d)} drift rows")
        r = {"label": c["label"], "recipe": c["recipe"], "step": step, "marks": c["marks"],
             "is_mark": any(t in MARKS for t in mark_tags(c["marks"])), "is_final": "final" in mark_tags(c["marks"]),
             **{k: d.iloc[0][k] for k in ("token_positions", "nonpad_tokens", "loss_tokens", "epochs", "dbpb_fresh",
                                          "dbpb_enwik8_ctrl", "drift_knowledge", "drift_middle", "drift_style",
                                          "drift_rare", "drift_frequent", "contrast_ks", "contrast_rf",
                                          "train_loss")}}
        if c["label"] in sc.index:
            s = sc.loc[c["label"]]
            r.update({k: s[k] for k in ("general", "general_se", "arc_easy", "arc_challenge", "hellaswag", "piqa",
                                        "winogrande", "mmlu", "ifeval_prompt_strict_chat", "gsm8k_flex_chat",
                                        "eval_wall_s", "chat_wall_s", "failed_groups", "gpu")})
            r["forgetting"] = 100 * (base["general"] - s.general) / base["general"]
            r["floor_pct"] = 100 * Z95 * np.sqrt(s.general_se**2 + base["general_se"] ** 2) / base["general"]
            r["arc_loss"] = 100 * (base["arc"] - (s.arc_easy + s.arc_challenge) / 2) / base["arc"]
            r["mmlu_loss"] = 100 * (base["mmlu"] - s.mmlu) / base["mmlu"]
            r["gain_ifeval"] = 100 * (s.ifeval_prompt_strict_chat - base_chat["ifeval"])
            r["gain_ifeval_exp04"] = 100 * (s.ifeval_prompt_strict_chat - base["ifeval_plain"])
            r["gain_gsm8k"] = 100 * (s.gsm8k_flex_chat - base_chat["gsm8k"])
        rows.append(r)
    order = {n: i for i, n in enumerate(RECIPES)}
    return pd.DataFrame(rows).sort_values(["recipe", "step"], key=lambda col: col.map(order) if col.name == "recipe"
                                          else col).reset_index(drop=True)


def per_mark(table: pd.DataFrame) -> pd.DataFrame:
    """Mark checkpoints exploded by mark: a checkpoint that crossed two marks at once appears under both."""
    rows = [{**r, "mark": t} for r in table.to_dict("records") for t in mark_tags(r["marks"]) if t in MARKS]
    return pd.DataFrame(rows)


def nan_extreme(values: np.ndarray, pick: object) -> float:
    """min or max of the finite values, nan if there are none."""
    v = values[np.isfinite(values)]
    return float(pick(v)) if v.size else float("nan")


def corr_row(df: pd.DataFrame, x: str, y: str, subset: str) -> dict[str, object]:
    """Pearson and Spearman with checkpoint-bootstrap CIs, leave-one-out range and leave-one-recipe-out r."""
    d = df[[x, y, "recipe", "label"]].dropna()
    out: dict[str, object] = {"x": x, "y": y, "subset": subset, "n": len(d)}
    if len(d) < 3:
        return out
    xs, ys = d[x].to_numpy(float), d[y].to_numpy(float)
    p, plo, phi, pn = bootstrap_corr(xs, ys, pearson, N_BOOT, SEED)
    s, slo, shi, sn = bootstrap_corr(xs, ys, spearman, N_BOOT, SEED)
    loo = leave_one_out(xs, ys, pearson)
    lor = {}
    for rec in RECIPES:
        k = (d.recipe != rec).to_numpy()
        lor[rec] = pearson(xs[k], ys[k]) if k.sum() >= 3 else np.nan
    out.update({"pearson": p, "pearson_lo": plo, "pearson_hi": phi, "spearman": s, "spearman_lo": slo,
                "spearman_hi": shi, "boot_valid": min(pn, sn), "loo_min": nan_extreme(loo, min),
                "loo_max": nan_extreme(loo, max), **{f"r_without_{k}": v for k, v in lor.items()}})
    return out


def residual_corr(df: pd.DataFrame, x: str, y: str, control: str) -> float:
    """Pearson r of x with the residual of y after an OLS fit of y on `control`."""
    d = df[[x, y, control]].dropna()
    slope, icpt = ols_line(d[control].to_numpy(float), d[y].to_numpy(float))
    return pearson(d[x].to_numpy(float), d[y].to_numpy(float) - (slope * d[control].to_numpy(float) + icpt))


def correlation_table(table: pd.DataFrame) -> pd.DataFrame:
    sets = {MARK_SET: table[table.is_mark], ALL_SET: table[table.is_mark | table.is_final]}
    rows = [corr_row(d, x, y, name) for name, d in sets.items()
            for x in (*CONTRASTS, "dbpb_fresh") for y in (*OUTCOMES, *(("dbpb_fresh",) if x != "dbpb_fresh" else ()))]
    return pd.DataFrame(rows)


def lookup(corr: pd.DataFrame, x: str, y: str, subset: str, stat: str = "pearson") -> float:
    hit = corr[(corr.x == x) & (corr.y == y) & (corr.subset == subset)]
    return float(hit[stat].iloc[0]) if len(hit) and stat in hit else float("nan")


def strictly_descending(values: Mapping[str, float], order: Sequence[str]) -> bool | None:
    """True if values follow `order` strictly descending; None when one is missing."""
    v = [values.get(k, np.nan) for k in order]
    if any(not np.isfinite(x) for x in v):
        return None
    return all(a > b for a, b in pairwise(v))


def h3_rows(table: pd.DataFrame, base: Mapping[str, float]) -> pd.DataFrame:
    """Per mark: recipes that reached it, achieved Δbpb_fresh spread, forgetting range and the 95% floor of the
    difference between the most- and least-forgetting checkpoints."""
    pm = per_mark(table)
    rows = []
    for m in MARKS:
        d = pm[(pm.mark == m) & pm.forgetting.notna()] if len(pm) else pm
        if len(d) < 2:
            rows.append({"mark": m, "n": len(d)})
            continue
        hi, lo = d.loc[d.forgetting.idxmax()], d.loc[d.forgetting.idxmin()]
        floor = 100 * Z95 * np.sqrt(hi.general_se**2 + lo.general_se**2) / base["general"]
        rows.append({"mark": m, "n": len(d), "recipes": ",".join(d.recipe), "dbpb_min": d.dbpb_fresh.min(),
                     "dbpb_max": d.dbpb_fresh.max(), "forgetting_range": hi.forgetting - lo.forgetting,
                     "most": hi.recipe, "least": lo.recipe, "floor_diff": floor,
                     "exceeds_floor": bool(hi.forgetting - lo.forgetting > floor)})
    return pd.DataFrame(rows)


def verdict_h1(r: float, n: int) -> str:
    v = "pass" if r >= H1_PASS else "soft" if r >= H1_SOFT else "fail"
    return v + (" (underpowered, n < 6)" if n < MIN_N else "") if np.isfinite(r) else "not evaluable"


def rule_table(table: pd.DataFrame, corr: pd.DataFrame, h3: pd.DataFrame, contrast: str) -> pd.DataFrame:
    """H1-H4 and the fail rule for one contrast (contrast_ks primary, contrast_rf secondary), mark checkpoints."""
    n = int(table.is_mark.sum())
    r1 = lookup(corr, contrast, "forgetting", MARK_SET)
    r2 = lookup(corr, contrast, "gain_ifeval", MARK_SET)
    rd = lookup(corr, contrast, "dbpb_fresh", MARK_SET)
    six = per_mark(table)
    six = six[six.mark == "6"].set_index("recipe") if len(six) else six
    c_order = strictly_descending(dict(six[contrast]) if len(six) else {}, H4_ORDER)
    f_order = strictly_descending(dict(six["forgetting"]) if len(six) and "forgetting" in six else {}, H4_ORDER)
    h4 = ("not evaluable" if c_order is None or f_order is None else "pass" if c_order and f_order
          else "soft (contrast order only)" if c_order else "fail")
    h3_ok = int(h3.get("exceeds_floor", pd.Series(dtype=bool)).fillna(False).sum()) if len(h3) else 0
    fail = (np.isfinite(rd) and rd > FAIL_R) or (np.isfinite(r1) and r1 < H1_SOFT and c_order is False)
    rows = [
        {"id": "H1", "contrast": contrast, "statistic": f"Pearson r({contrast}, forgetting), n = {n}", "value": r1,
         "verdict": verdict_h1(r1, n)},
        {"id": "H2", "contrast": contrast, "statistic": f"Pearson r({contrast}, IFEval gain), n = {n}", "value": r2,
         "verdict": ("pass" if abs(r2) < H2_MAX else "fail") if np.isfinite(r2) else "not evaluable"},
        {"id": "H3", "contrast": contrast, "statistic": "marks whose forgetting range exceeds the 95% floor",
         "value": h3_ok, "verdict": "pass" if h3_ok >= 2 else "fail"},
        {"id": "H4", "contrast": contrast, "statistic": "6% mark: contrast and forgetting mathcot > smoltalk > tulu",
         "value": f"contrast {c_order}, forgetting {f_order}", "verdict": h4},
        {"id": "fail rule", "contrast": contrast,
         "statistic": f"r({contrast}, Δbpb_fresh) > 0.9, or H1 < 0.6 with the H4 contrast order violated",
         "value": rd, "verdict": "triggered: lead not supported" if fail else "not triggered"},
    ]
    return pd.DataFrame(rows)


def fmt(v: object, digits: int = 2) -> str:
    if isinstance(v, float | np.floating):
        return "n/a" if not np.isfinite(v) else f"{v:+.{digits}f}" if digits >= 0 else f"{v:.0f}"
    return str(v)


def summary_markdown(table: pd.DataFrame, tests: pd.DataFrame, corr: pd.DataFrame, h3: pd.DataFrame,
                     checks: Mapping[str, object], resid: Mapping[str, float]) -> str:
    out = ["# exp06 summary (written by analyze_matched.py)", "", "## Tests", "",
           "| id | contrast | statistic | value | verdict |", "|---|---|---|---|---|"]
    out += [f"| {r.id} | {r.contrast} | {r.statistic} | {fmt(r.value, 3) if isinstance(r.value, float) else r.value}"
            f" | {r.verdict} |" for r in tests.itertuples()]
    out += ["", "## Checkpoints", "",
            "| recipe | step | marks | tokens (M) | epochs | Δbpb_fresh % | contrast K−S pp | contrast R−F pp | "
            "forgetting % (floor) | IFEval gain | gsm8k gain | ARC loss % | MMLU loss % |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        out.append(f"| {r.recipe} | {r.step} | {r.marks} | {r.token_positions / 1e6:.1f} | {r.epochs:.2f} | "
                   f"{r.dbpb_fresh:+.2f} | {r.contrast_ks:+.2f} | {r.contrast_rf:+.2f} | "
                   f"{fmt(getattr(r, 'forgetting', np.nan))} ({fmt(getattr(r, 'floor_pct', np.nan))}) | "
                   f"{fmt(getattr(r, 'gain_ifeval', np.nan), 1)} | {fmt(getattr(r, 'gain_gsm8k', np.nan), 1)} | "
                   f"{fmt(getattr(r, 'arc_loss', np.nan))} | {fmt(getattr(r, 'mmlu_loss', np.nan))} |")
    out += ["", "## H3 per mark", "", h3.to_markdown(index=False, floatfmt=".3f") if len(h3) else "(none)", "",
            "## Correlations", "", "| subset | x | y | n | Pearson r [95% CI] | Spearman ρ | leave-one-recipe-out r "
            "(tulu / smoltalk / mathcot dropped) |", "|---|---|---|---|---|---|---|"]
    for r in corr.itertuples():
        if not np.isfinite(getattr(r, "pearson", np.nan)):
            continue
        out.append(f"| {r.subset} | {r.x} | {r.y} | {r.n} | {r.pearson:+.3f} [{r.pearson_lo:+.2f}, "
                   f"{r.pearson_hi:+.2f}] | {r.spearman:+.3f} | {fmt(r.r_without_tulu, 2)} / "
                   f"{fmt(r.r_without_smoltalk, 2)} / {fmt(r.r_without_mathcot, 2)} |")
    out += ["", "## Residual reading (forgetting after OLS on Δbpb_fresh, mark checkpoints)", ""]
    out += [f"- r({k}, residual) = {v:+.3f}" for k, v in resid.items()]
    out += ["", "## Sanity checks", "", "```", json.dumps(checks, indent=1, default=float), "```", ""]
    return "\n".join(out)


# ----------------------------------------------------------------------------- figure
def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure(traj: pd.DataFrame, table: pd.DataFrame, corr: pd.DataFrame) -> Figure:
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.6), facecolor=SURFACE)
    (a, b), (c, d) = axs
    for ax in axs.flat:
        style_axis(ax)
    for rec in RECIPES:
        t = traj[traj.recipe == rec]
        if not len(t):
            continue
        col = RECIPE_COLOR[rec]
        a.plot(t.token_positions / 1e6, t.dbpb_fresh, color=col, lw=2, label=rec)
        b.plot(t.dbpb_fresh, t.contrast_ks, color=col, lw=2, label=rec)
        k = table[table.recipe == rec]
        for r in k.itertuples():
            tag = next((m for m in mark_tags(r.marks) if m in MARK_SHAPE), "final")
            for ax, x, y in ((a, r.token_positions / 1e6, r.dbpb_fresh), (b, r.dbpb_fresh, r.contrast_ks)):
                ax.plot(x, y, MARK_SHAPE[tag], color=col, ms=8, mec=SURFACE, mew=2, zorder=3)
    for m in MARKS:
        a.axhline(float(m), color=MUTED, lw=0.8, ls="--")
        b.axvline(float(m), color=MUTED, lw=0.8, ls="--")
    a.set_xlabel("training tokens (M, packed positions)", color=INK2, fontsize=9)
    a.set_ylabel("Δbpb_fresh vs base (%)", color=INK2, fontsize=9)
    a.set_title("(a) drift during training; marks at 3 / 6 / 10%", color=INK, fontsize=10, loc="left")
    b.axhline(0, color=MUTED, lw=0.8)
    b.set_xlabel("Δbpb_fresh vs base (%)", color=INK2, fontsize=9)
    b.set_ylabel("contrast knowledge − style (pp)", color=INK2, fontsize=9)
    b.set_title("(b) direction of the drift at each dose", color=INK, fontsize=10, loc="left")
    pts = table[table.is_mark & table.get("forgetting", pd.Series(np.nan, index=table.index)).notna()]
    for ax, y, ylab, title in ((c, "forgetting", "forgetting (% of base general score)", "(c) H1: contrast vs forgetting"),
                               (d, "gain_ifeval", "IFEval chat gain vs base (points)", "(d) H2: contrast vs IFEval gain")):
        for r in pts.itertuples():
            tag = next(m for m in mark_tags(r.marks) if m in MARKS)
            ax.plot(r.contrast_ks, getattr(r, y), MARK_SHAPE[tag], color=RECIPE_COLOR[r.recipe], ms=9, mec=SURFACE,
                    mew=2)
            ax.annotate(f"{r.recipe} {tag}%", (r.contrast_ks, getattr(r, y)), xytext=(5, 4),
                        textcoords="offset points", fontsize=7, color=INK2)
        rr = lookup(corr, "contrast_ks", y, MARK_SET)
        ax.set_title(f"{title} (r = {fmt(rr, 2)}, n = {len(pts)})", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("contrast knowledge − style (pp)", color=INK2, fontsize=9)
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
        ax.axhline(0, color=MUTED, lw=0.8)
    handles = [Line2D([], [], color=RECIPE_COLOR[r], lw=2, label=r) for r in RECIPES]
    handles += [Line2D([], [], marker=MARK_SHAPE[m], color=MUTED, ls="", ms=7, label=f"{m}% mark") for m in MARKS]
    handles.append(Line2D([], [], marker=MARK_SHAPE["final"], color=MUTED, ls="", ms=7, label="final"))
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# ----------------------------------------------------------------------------- effects
def run_dirs(train: Path) -> dict[str, Path]:
    """Per recipe, the last attempt directory that finished (done.json)."""
    out = {}
    for rec in RECIPES:
        cands = sorted([p for p in train.glob(f"{rec}*") if (p / "done.json").exists()
                        and (p.name == rec or p.name.startswith(f"{rec}-attempt"))])
        if cands:
            out[rec] = cands[-1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(HERE / "results"))
    out = Path(ap.parse_args().results)
    dirs = run_dirs(out / "train")
    traj = pd.concat([pd.read_csv(d / "drift.csv") for d in dirs.values()], ignore_index=True)
    lines = [ln for d in dirs.values() for ln in (d / "checkpoints.txt").read_text().splitlines(keepends=True)]
    evals = {p.parent.name: json.loads(p.read_text()) for p in sorted((out / "lm_eval").glob("*/summary.json"))}
    base_eval = general_scores(pd.DataFrame([eval_scores(evals.pop(BASE_LABEL))])).iloc[0]
    e4 = pd.read_csv(EXP04_SCORES).set_index("model").loc[BASE_MODEL]
    base = {"general": float(e4.general), "general_se": float(e4.general_se), "mmlu": float(e4.mmlu),
            "arc": float((e4.arc_easy + e4.arc_challenge) / 2), "ifeval_plain": float(e4.ifeval_prompt_strict)}
    base_chat = {"ifeval": float(base_eval.ifeval_prompt_strict_chat), "gsm8k": float(base_eval.gsm8k_flex_chat)}
    table = checkpoint_table(lines, traj, evals, base, base_chat)
    corr = correlation_table(table)
    h3 = h3_rows(table, base)
    tests = pd.concat([rule_table(table, corr, h3, c) for c in CONTRASTS], ignore_index=True)
    marks_only = table[table.is_mark]
    resid = {c: residual_corr(marks_only, c, "forgetting", "dbpb_fresh") for c in CONTRASTS}
    checks = {"exp06 base general (rerun, Instruct tokenizer)": float(base_eval.general),
              "exp04 base general": base["general"],
              "base general diff (points)": 100 * (float(base_eval.general) - base["general"]),
              "base chat ifeval": base_chat["ifeval"], "base chat gsm8k": base_chat["gsm8k"],
              "base plain ifeval (exp04)": base["ifeval_plain"],
              "step-0 bpb vs exp03 (max |rel|)": {r: json.loads((d / "base_check.json").read_text())["max_abs_rel"]
                                                 for r, d in dirs.items()},
              "runs": {r: json.loads((d / "done.json").read_text()) for r, d in dirs.items()}}
    table.to_csv(out / "checkpoints.csv", index=False, float_format="%.5f")
    traj.to_csv(out / "trajectories.csv", index=False, float_format="%.5f")
    corr.to_csv(out / "correlations.csv", index=False, float_format="%.4f")
    tests.to_csv(out / "tests.csv", index=False, float_format="%.4f")
    (out / "summary.md").write_text(summary_markdown(table, tests, corr, h3, checks, resid))
    figure(traj, table, corr).savefig(out / "fig_matched.png", dpi=160, facecolor=SURFACE)
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()

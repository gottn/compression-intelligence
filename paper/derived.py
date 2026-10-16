"""Numbers in the report that no RESULTS.md prints, derived from the committed results of exp01–exp06.

Run from the repository root (writes paper/derived_numbers.md, which main.tex quotes):
    .venv/bin/python -m paper.derived

Pure above the line marked "effects"; reads and the one write are in `read_inputs` and `main`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cilab.pairs import FRESH_CORPORA
from cilab.stages import STAGES
from cilab.stats import pearson, resample_indices, spearman
from cilab.tokclass import FREQUENT, KNOWLEDGE, RARE, STYLE, window_class_sums

REPO = Path(__file__).resolve().parents[1]
CORPORA = (*FRESH_CORPORA, "enwik8_ctrl")
MATH_LINE = ("Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
GENERAL = ("arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "mmlu")
N_BOOT, SEED = 5000, 0


@dataclass(frozen=True)
class Inputs:
    exp01_bpb: pd.DataFrame
    exp01_nl: pd.DataFrame  # exp01 blank-line ("\n\n") document-start ablation
    exp03_bpb: pd.DataFrame
    exp03_deltas: pd.DataFrame
    exp04_scores: pd.DataFrame
    exp04_loglik_s: dict[str, float]  # model -> wall s of the log-likelihood group (A10)
    exp05_decomp: pd.DataFrame
    exp06_ckpts: pd.DataFrame
    exp06_traj: pd.DataFrame
    exp06_loglik_s: dict[str, float]  # checkpoint label -> wall s of the log-likelihood group (A10)
    exp06_bits: dict[tuple[str, int, str], np.ndarray]  # (recipe, update, corpus) -> per-token bits
    olmo2_classes: dict[str, dict[str, np.ndarray]]  # corpus -> {pos, freq, lengths}


# ----------------------------------------------------------------------------- cost
def cost(inp: Inputs) -> dict[str, float]:
    """Wall time of the log-likelihood benchmark group (A10) against one drift score (T4 in exp03, A40 in exp06)."""
    t4 = inp.exp03_bpb[inp.exp03_bpb.corpus.isin(FRESH_CORPORA)].groupby("model").wall_s.sum()
    ll = np.array(sorted(inp.exp04_loglik_s.values()))
    a40 = inp.exp06_traj[inp.exp06_traj.step > 0].score_s
    ll6 = np.array(sorted(inp.exp06_loglik_s.values()))
    return {"loglik_median_s": float(np.median(ll)), "loglik_min_s": float(ll.min()), "loglik_max_s": float(ll.max()),
            "drift_t4_median_s": float(t4.median()), "drift_t4_min_s": float(t4.min()), "drift_t4_max_s": float(t4.max()),
            "ratio_exp04": float(np.median(ll) / t4.median()),
            "drift_a40_median_s": float(a40.median()), "loglik6_min_s": float(ll6.min()), "loglik6_max_s": float(ll6.max()),
            "ratio_exp06_lo": float(ll6.min() / a40.median()), "ratio_exp06_hi": float(ll6.max() / a40.median())}


# ----------------------------------------------------------------------------- exp04 readings
def fresh_bpb(bpb: pd.DataFrame, model: str, condition: str = "raw") -> float:
    r = bpb[(bpb.model == model) & (bpb.condition == condition) & bpb.corpus.isin(FRESH_CORPORA)]
    if len(r) != 3:
        raise ValueError(f"{model} {condition}: {len(r)} fresh rows")
    return float(r.bpb.mean())


def absolute_bpb(inp: Inputs) -> dict[str, float]:
    """Absolute fresh-text bpb (mean of the three corpora) against the general score."""
    s = inp.exp04_scores
    src = {m: inp.exp03_bpb for m in inp.exp03_bpb.model.unique()}
    rows = []
    for st in STAGES:
        bpb = src.get(st.model, inp.exp01_bpb)
        g = float(s.loc[s.model == st.model, "general"].iloc[0])
        rows.append((st.model, st.parent is None, fresh_bpb(bpb, st.model), g))
    t = pd.DataFrame(rows, columns=["model", "is_base", "bpb", "general"])
    post, base = t[~t.is_base], t[t.is_base]
    post10 = post[~post.model.isin(MATH_LINE)]
    return {"r_post": pearson(post.bpb, post.general), "rho_post": spearman(post.bpb, post.general),
            "r_post_no_math": pearson(post10.bpb, post10.general), "r_base": pearson(base.bpb, base.general),
            "rho_base": spearman(base.bpb, base.general), "n_post": len(post), "n_base": len(base)}


def gate_families(inp: Inputs, n_boot: int = 10_000, n_perm: int = 20_000) -> dict[str, float]:
    """Gate 1 at family level: a family-cluster bootstrap of r, the family-demeaned r, and one-sided permutation
    p-values of r and rho under exchangeable checkpoints."""
    s = inp.exp04_scores[inp.exp04_scores.stage != "base"].reset_index(drop=True)
    x, y, fam = s.dbpb_fresh.to_numpy(), s.forgetting.to_numpy(), s.family.to_numpy()
    fams = np.array(sorted(set(fam)))
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(fams, size=len(fams), replace=True)
        idx = np.concatenate([np.flatnonzero(fam == f) for f in pick])
        if np.ptp(x[idx]) > 0 and np.ptp(y[idx]) > 0:
            vals.append(pearson(x[idx], y[idx]))
    demean = pd.DataFrame({"x": x, "y": y, "f": fam})
    demean = demean.assign(x=demean.x - demean.groupby("f").x.transform("mean"),
                           y=demean.y - demean.groupby("f").y.transform("mean"))
    r0, rho0 = pearson(x, y), spearman(x, y)
    rng = np.random.default_rng(SEED)
    perm = np.array([rng.permutation(y) for _ in range(n_perm)])
    pr = np.array([pearson(x, p) for p in perm])
    prho = np.array([spearman(x, p) for p in perm])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"r": r0, "cluster_lo": float(lo), "cluster_hi": float(hi), "cluster_valid": len(vals),
            "r_demeaned": pearson(demean.x, demean.y), "n_families": len(fams),
            "perm_p_r": float((1 + (pr >= r0).sum()) / (1 + n_perm)),
            "perm_p_rho": float((1 + (prho >= rho0).sum()) / (1 + n_perm))}


def docstart(inp: Inputs) -> dict[str, float]:
    """Drift with a blank-line document start (exp01 ablation) and Gate 1's r with it substituted where it exists."""
    nl = inp.exp01_nl
    def d(base: str, post: str) -> float:
        per = []
        for c in FRESH_CORPORA:
            b = nl[(nl.model == base) & (nl.corpus == c)].total_bits.iloc[0]
            p = nl[(nl.model == post) & (nl.corpus == c)].total_bits.iloc[0]
            per.append(100 * (p - b) / b)
        return float(np.mean(per))
    q15, math_, r1 = "Qwen/Qwen2.5-1.5B", *MATH_LINE
    sub = {"Qwen/Qwen2.5-1.5B-Instruct": d(q15, "Qwen/Qwen2.5-1.5B-Instruct"), math_: d(q15, math_),
           r1: d(q15, r1), "HuggingFaceTB/SmolLM2-1.7B-Instruct": d("HuggingFaceTB/SmolLM2-1.7B",
                                                                     "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
           "Qwen/Qwen3-1.7B": d("Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B")}
    s = inp.exp04_scores[inp.exp04_scores.stage != "base"]
    x = np.array([sub.get(m, v) for m, v in zip(s.model, s.dbpb_fresh, strict=True)])
    return {"math_vs_base_nl": sub[math_], "r1_vs_base_nl": sub[r1], "r1_vs_math_nl": d(math_, r1),
            "n_substituted": len(sub), "gate_r_nl": pearson(x, s.forgetting), "gate_rho_nl": spearman(x, s.forgetting)}


def enwik8_ratios(inp: Inputs) -> pd.DataFrame:
    """enwik8 Δ divided by fresh-mean Δ for every exp03 comparison."""
    d = inp.exp03_deltas.pivot_table(index=["model", "kind"], columns="corpus", values="delta_pct")
    return (d.enwik8_ctrl / d.fresh_mean).rename("ratio").reset_index()


def scale_free_contrast(inp: Inputs) -> dict[str, float]:
    """exp05's contrast K − S divided by the total drift, against forgetting."""
    d = inp.exp05_decomp[(inp.exp05_decomp.scheme == "pos") & (inp.exp05_decomp.corpus == "fresh_mean")]
    p = d.pivot_table(index="model", columns="class", values="drift_pct")
    sf = ((p.knowledge - p["style"]) / p["all"]).rename("sf")
    s = inp.exp04_scores[inp.exp04_scores.stage != "base"].set_index("model").join(sf)
    s10 = s[~s.index.isin(MATH_LINE)]
    return {"r_sf_12": pearson(s.sf, s.forgetting), "r_sf_10": pearson(s10.sf, s10.forgetting)}


def per_task(scores_post: pd.DataFrame, base: pd.Series) -> pd.DataFrame:
    """Forgetting without ARC and without MMLU, and the ARC and MMLU changes in points, for each row."""
    def mean(df: pd.DataFrame | pd.Series, tasks: tuple[str, ...]) -> pd.Series | float:
        return df[list(tasks)].mean(axis=1) if isinstance(df, pd.DataFrame) else float(df[list(tasks)].mean())
    no_arc, no_mmlu = GENERAL[2:], tuple(t for t in GENERAL if t != "mmlu")
    out = pd.DataFrame(index=scores_post.index)
    out["forget_no_arc"] = 100 * (mean(base, no_arc) - mean(scores_post, no_arc)) / mean(base, no_arc)
    out["forget_no_mmlu"] = 100 * (mean(base, no_mmlu) - mean(scores_post, no_mmlu)) / mean(base, no_mmlu)
    for t in ("arc_easy", "arc_challenge", "mmlu"):
        out[f"d_{t}"] = 100 * (scores_post[t] - float(base[t]))
    return out


def exp04_per_task(inp: Inputs) -> pd.DataFrame:
    s = inp.exp04_scores
    rows = []
    for r in s[s.stage != "base"].itertuples():
        b = s[s.model == r.base].iloc[0]
        rows.append(per_task(s[s.model == r.model], b).assign(model=r.model))
    return pd.concat(rows).set_index("model")


def exp06_per_task(inp: Inputs) -> pd.DataFrame:
    base = inp.exp04_scores[inp.exp04_scores.model == "allenai/OLMo-2-0425-1B"].iloc[0]
    ck = inp.exp06_ckpts.set_index(["recipe", "step"])
    return per_task(ck, base)


# ----------------------------------------------------------------------------- exp06 contrast intervals
def window_sums(bits: np.ndarray, codes: np.ndarray, lengths: np.ndarray, n: int) -> np.ndarray:
    return window_class_sums(bits, codes, [int(v) for v in lengths], n)


def contrast_rows(inp: Inputs, scheme: str, hi_cls: int, lo_cls: int, n_cls: int) -> dict:
    """Per checkpoint: per-corpus (n_boot, ...) resampled class drifts. Returns {(recipe, step): (point, boot)}, the
    contrast drift_hi − drift_lo of the fresh mean, with the same resamples for every checkpoint (paired)."""
    idx = {c: resample_indices(len(inp.olmo2_classes[c]["lengths"]), N_BOOT, SEED + i)
           for i, c in enumerate(FRESH_CORPORA)}
    out = {}
    for r in inp.exp06_ckpts.itertuples():
        pts, boots = [], []
        for c in FRESH_CORPORA:
            cl = inp.olmo2_classes[c]
            wb = window_sums(inp.exp06_bits[(r.recipe, 0, c)], cl[scheme], cl["lengths"], n_cls)
            wp = window_sums(inp.exp06_bits[(r.recipe, int(r.step), c)], cl[scheme], cl["lengths"], n_cls)
            ident = np.arange(len(wb))[None, :]
            def drift(ix: np.ndarray, k: int, wb: np.ndarray = wb, wp: np.ndarray = wp) -> np.ndarray:
                return 100 * (wp[ix, k].sum(1) - wb[ix, k].sum(1)) / wb[ix, k].sum(1)
            pts.append(drift(ident, hi_cls)[0] - drift(ident, lo_cls)[0])
            boots.append(drift(idx[c], hi_cls) - drift(idx[c], lo_cls))
        out[(r.recipe, int(r.step))] = (float(np.mean(pts)), np.mean(boots, axis=0))
    return out


def contrast_table(inp: Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K − S and rare − frequent with 95% window-bootstrap intervals per checkpoint, and paired differences at the
    marks. Points are checked against checkpoints.csv (stage-2 rows use the stage-1 bf16 base as reference)."""
    ks = contrast_rows(inp, "pos", KNOWLEDGE, STYLE, 3)
    rf = contrast_rows(inp, "freq", RARE, FREQUENT, 2)
    rows = []
    for r in inp.exp06_ckpts.itertuples():
        k = (r.recipe, int(r.step))
        for name, d, ref in (("ks", ks, r.contrast_ks), ("rf", rf, r.contrast_rf)):
            pt, bt = d[k]
            if abs(pt - ref) > 0.02:
                raise AssertionError(f"{k} {name}: recomputed {pt:.3f}, checkpoints.csv {ref:.3f}")
            lo, hi = np.percentile(bt, [2.5, 97.5])
            rows.append({"recipe": r.recipe, "step": int(r.step), "marks": str(r.marks), "contrast": name,
                         "point": pt, "csv": ref, "lo": float(lo), "hi": float(hi)})
    t = pd.DataFrame(rows)
    pairs = []
    marks = inp.exp06_ckpts[inp.exp06_ckpts.is_mark]
    for mark in ("3", "6"):
        at = {r.recipe: int(r.step) for r in marks[marks.marks.astype(str) == mark].itertuples()}
        for a, b in (("mathcot", "tulu"), ("mathcot", "smoltalk"), ("tulu", "smoltalk")):
            diff = ks[(a, at[a])][1] - ks[(b, at[b])][1]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            pairs.append({"mark": mark, "pair": f"{a} - {b}", "point": ks[(a, at[a])][0] - ks[(b, at[b])][0],
                          "lo": float(lo), "hi": float(hi)})
    return t, pd.DataFrame(pairs)


def smoothed_marks(inp: Inputs, half: int = 3) -> pd.DataFrame:
    """K − S at each mark, raw and as the median of the 2·half + 1 scored rows around it (±60 updates)."""
    tr = inp.exp06_traj.sort_values(["recipe", "step"]).reset_index(drop=True)
    rows = []
    for r in inp.exp06_ckpts[inp.exp06_ckpts.is_mark].itertuples():
        g = tr[tr.recipe == r.recipe].reset_index(drop=True)
        i = int(np.flatnonzero(g.step.to_numpy() == r.step)[-1])
        win = g.contrast_ks.iloc[max(0, i - half): i + half + 1]
        rows.append({"recipe": r.recipe, "mark": str(r.marks), "raw": float(g.contrast_ks.iloc[i]),
                     "median7": float(win.median())})
    return pd.DataFrame(rows)


def exp06_misc(inp: Inputs) -> dict[str, float]:
    ck, tr = inp.exp06_ckpts, inp.exp06_traj.sort_values(["recipe", "step"])
    marks = ck[ck.is_mark]
    out = {"r_forget_gain_12": pearson(ck.forgetting, ck.gain_ifeval),
           "r_forget_gain_marks": pearson(marks.forgetting, marks.gain_ifeval),
           "r_drift_gain_12": pearson(ck.dbpb_fresh, ck.gain_ifeval),
           "r_drift_gain_marks": pearson(marks.dbpb_fresh, marks.gain_ifeval)}
    s1500 = ck[ck.step == 1500].set_index("recipe")
    for rec in ("tulu", "smoltalk", "mathcot"):
        out[f"loss_tokens_M_{rec}"] = float(s1500.loc[rec, "loss_tokens"]) / 1e6
        out[f"padding_{rec}"] = 1 - float(s1500.loc[rec, "nonpad_tokens"]) / float(s1500.loc[rec, "token_positions"])
    lr = []
    for r in ck.itertuples():
        g = tr[(tr.recipe == r.recipe) & (tr.step == r.step) & (tr.stage == r.stage)]
        lr.append((r.recipe, int(r.step), str(r.marks), float(g.lr.iloc[0])))
    lrs = pd.DataFrame(lr, columns=["recipe", "step", "marks", "lr"])
    out["lr_marks_min"] = float(lrs[lrs.marks != "final"].lr.min())
    out["lr_marks_max"] = float(lrs[lrs.marks != "final"].lr.max())
    out["lr_stage2_final_min"] = float(lrs[(lrs.marks == "final") & (lrs.step > 1500)].lr.min())
    frac = marks.epochs - np.floor(marks.epochs)
    out["marks_within_0.06_passes_after_boundary"] = int((frac < 0.06).sum())
    return out


def boundary_jumps(inp: Inputs) -> pd.DataFrame:
    """Change of Δbpb fresh over the first one and two scored rows (20 and 40 updates) after each pass boundary."""
    rows = []
    for rec, g in inp.exp06_traj.sort_values("step").groupby("recipe"):
        g = g.reset_index(drop=True)
        e = np.floor(g.epochs.to_numpy())
        for i in np.flatnonzero(np.diff(e) > 0):
            j1, j2 = min(i + 1, len(g) - 1), min(i + 2, len(g) - 1)
            rows.append({"recipe": rec, "pass": int(e[i + 1]), "step": int(g.step[i]), "before": float(g.dbpb_fresh[i]),
                         "jump20": float(g.dbpb_fresh[j1] - g.dbpb_fresh[i]),
                         "jump40": float(g.dbpb_fresh[j2] - g.dbpb_fresh[i]),
                         "loss_before": float(g.train_loss[i]), "loss_after": float(g.train_loss[j2])})
    return pd.DataFrame(rows)


def range_of_three_factor() -> float:
    """95th percentile of the range of three iid N(0, 1), divided by the two-sample 95% floor 1.96·√2."""
    x = np.linspace(-8, 8, 16001)
    phi = np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    cdf = np.vectorize(lambda v: 0.5 * (1 + math.erf(v / math.sqrt(2))))
    def p_range(w: float) -> float:
        return float(np.trapezoid(3 * phi * (cdf(x + w) - cdf(x)) ** 2, x))
    lo, hi = 2.0, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_range(mid) < 0.95 else (lo, mid)
    return (lo + hi) / 2 / (1.96 * math.sqrt(2))


# ----------------------------------------------------------------------------- effects
def read_inputs(repo: Path) -> Inputs:
    e4, e6 = repo / "exp04_benchmarks" / "results", repo / "exp06_matched_drift" / "results"
    ll4 = {json.loads(p.read_text())["model"]: json.loads(p.read_text())["groups"]["loglik"]["wall_s"]
           for p in sorted((e4 / "lm_eval").glob("*/summary.json"))}
    ll6 = {p.parent.name: json.loads(p.read_text())["groups"]["loglik"]["wall_s"]
           for p in sorted((e6 / "lm_eval").glob("*/summary.json"))}
    ck = pd.read_csv(e6 / "checkpoints.csv")
    bits = {}
    for rec in ("tulu", "smoltalk", "mathcot"):
        for c in FRESH_CORPORA:
            bits[(rec, 0, c)] = np.load(e6 / "train" / rec / "tokens" / f"step00000__{c}.npy")
    for r in ck.itertuples():
        d = "train" if r.stage == 1 else "train2"
        for c in FRESH_CORPORA:
            bits[(r.recipe, int(r.step), c)] = np.load(e6 / d / r.recipe / "tokens" / f"step{int(r.step):05d}__{c}.npy")
    classes = {}
    for c in FRESH_CORPORA:
        z = np.load(repo / "exp05_decomposition" / "results" / "classes" / f"olmo2__{c}.npz")
        classes[c] = {"pos": z["pos"], "freq": z["freq"], "lengths": z["lengths"]}
    nl = pd.read_csv(repo / "exp01_posttraining_bpc" / "results" / "ablation_docstart.csv")
    return Inputs(
        exp01_bpb=pd.read_csv(repo / "exp01_posttraining_bpc" / "results" / "bpb.csv"), exp01_nl=nl,
        exp03_bpb=pd.read_csv(repo / "exp03_staged_drift" / "results" / "bpb.csv"),
        exp03_deltas=pd.read_csv(repo / "exp03_staged_drift" / "results" / "stage_deltas.csv"),
        exp04_scores=pd.read_csv(e4 / "scores.csv"), exp04_loglik_s=ll4,
        exp05_decomp=pd.read_csv(repo / "exp05_decomposition" / "results" / "decomposition.csv"),
        exp06_ckpts=ck, exp06_traj=pd.read_csv(e6 / "trajectories.csv"), exp06_loglik_s=ll6,
        exp06_bits=bits, olmo2_classes=classes)


def report(inp: Inputs) -> str:
    """All derived numbers as Markdown."""
    def kv(title: str, d: dict) -> list[str]:
        return [f"## {title}", ""] + [f"- {k}: {v:.4g}" if isinstance(v, float) else f"- {k}: {v}"
                                      for k, v in d.items()] + [""]
    ct, pairs = contrast_table(inp)
    out = ["# Derived numbers (written by paper/derived.py; not printed in any RESULTS.md)", ""]
    out += kv("Cost: log-likelihood group (A10) against one drift score", cost(inp))
    out += kv("Absolute fresh-text bpb against the general score (exp03/exp01 bpb, exp04 scores)", absolute_bpb(inp))
    out += kv("Gate 1 at family level (exp04)", gate_families(inp))
    out += kv("Blank-line document start (exp01 ablation_docstart.csv)", docstart(inp))
    out += kv("exp05 contrast divided by total drift", scale_free_contrast(inp))
    out += kv("exp06 correlations, supervision, learning rate", exp06_misc(inp))
    out += kv("Range-of-three floor", {"factor": range_of_three_factor(),
                                      "floor_pp": range_of_three_factor() * float(inp.exp06_ckpts.floor_pct.median())})
    out += ["## enwik8 / fresh ratio (exp03)", "", enwik8_ratios(inp).round(3).to_markdown(index=False), ""]
    out += ["## exp06 contrasts with 95% window-bootstrap intervals", "", ct.round(3).to_markdown(index=False), ""]
    out += ["## exp06 paired K − S differences at the marks", "", pairs.round(3).to_markdown(index=False), ""]
    out += ["## exp06 K − S at the marks, raw and 7-row median", "", smoothed_marks(inp).round(3).to_markdown(index=False), ""]
    out += ["## exp06 pass-boundary jumps", "", boundary_jumps(inp).round(3).to_markdown(index=False), ""]
    out += ["## exp04 per task (points; forgetting in % of base)", "", exp04_per_task(inp).round(2).to_markdown(), ""]
    out += ["## exp06 per task", "", exp06_per_task(inp).round(2).to_markdown(), ""]
    return "\n".join(out)


def main() -> None:
    text = report(read_inputs(REPO))
    (REPO / "paper" / "derived_numbers.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

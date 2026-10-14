"""exp02 analysis: tables, bootstrap CIs, the pre-stated verdict rule and the figure.

Reads results/{runs.csv,windows.csv,grid/*.npy} written by score_temperature.py and writes
results/summary.md and results/fig_temperature.png. Every quantity uses odd-indexed windows only
(T* was fit on the even ones). Pure functions; file I/O only in load_inputs() and main().

Rule (stated before the main run): recovered = (bpb_post(1) − bpb_post(T*)) / (bpb_post(1) − bpb_base(1)).
Headline per pair = mean over the 3 fresh corpora. Both big-penalty pairs (qwen3, qwen2.5-math)
>= 70% -> mostly a confidence shift; both <= 30% -> mostly a change in beliefs; else mixed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cilab`

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from cilab.calib import T_GRID, fit_temperature, recovered_stat
from cilab.pairs import CORPORA, FRESH_CORPORA, PAIRS, model_roles, model_slug
from cilab.stats import bootstrap_ci, paired_pct_change, resample_indices

N_BOOT, SEED = 5000, 0
T1 = T_GRID.index(1.0)
BIG_PENALTY = ("qwen3", "qwen2.5-math")
PAIR_LABEL = {"qwen2.5": "Qwen2.5-1.5B\n→ Instruct", "qwen3": "Qwen3-1.7B-Base\n→ Qwen3-1.7B",
              "smollm2": "SmolLM2-1.7B\n→ Instruct", "qwen2.5-math": "Qwen2.5-Math-1.5B\n→ R1-Distill"}
BASE_C, POST_C = "#2a78d6", "#eb6834"  # dataviz reference palette slots 1, 2 (as in exp01)
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"

Arrays = dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]  # (model, corpus) -> (grid, tstar bits)


# ----------------------------------------------------------------------------- pure
def odd_windows(n: int) -> np.ndarray:
    return np.arange(1, n, 2)


def model_table(runs: pd.DataFrame, windows: pd.DataFrame, arrays: Arrays) -> pd.DataFrame:
    """Per (model, corpus): bpb at T = 1 and at T* on odd windows, T* (even fit) and an odd-fit check."""
    rows = []
    for r in runs.itertuples():
        grid, tstar = arrays[(r.model, r.corpus)]
        odd = odd_windows(len(grid))
        nbytes = windows[(windows.model == r.model) & (windows.corpus == r.corpus)].sort_values("window")
        bytes_odd = float(nbytes.bytes.to_numpy()[odd].sum())
        rows.append({"model": r.model, "pair": r.pair, "role": r.role, "corpus": r.corpus,
                     "n_odd": odd.size, "bytes_odd": int(bytes_odd),
                     "bpb_t1": grid[odd, T1].sum() / bytes_odd, "bpb_tstar": tstar[odd].sum() / bytes_odd,
                     "t_star": float(r.t_star), "at_edge": bool(r.at_edge),
                     "t_star_odd_fit": fit_temperature(T_GRID, grid[odd].sum(0))[0]})
    t = pd.DataFrame(rows)
    t["gain_pct"] = 100 * (t.bpb_t1 - t.bpb_tstar) / t.bpb_t1
    return t


def pair_table(runs: pd.DataFrame, windows: pd.DataFrame, arrays: Arrays) -> pd.DataFrame:
    """Per (pair, corpus): gap at T = 1, T* base/post, recovered fraction and residual gap with CIs."""
    rows = []
    for pair, base, post in PAIRS:
        for corpus in runs.corpus.drop_duplicates():
            if (base, corpus) not in arrays or (post, corpus) not in arrays:
                continue
            (gb, tb), (gp, tp) = arrays[(base, corpus)], arrays[(post, corpus)]
            wb = windows[(windows.model == base) & (windows.corpus == corpus)].sort_values("window")
            wp = windows[(windows.model == post) & (windows.corpus == corpus)].sort_values("window")
            if not np.array_equal(wb[["tokens", "bytes"]].to_numpy(), wp[["tokens", "bytes"]].to_numpy()):
                raise ValueError(f"{pair} {corpus}: base and post windows differ")
            odd = odd_windows(len(gb))
            b1, p1, bt, pt = gb[odd, T1], gp[odd, T1], tb[odd], tp[odd]
            idx = resample_indices(odd.size, N_BOOT, SEED)
            rec = bootstrap_ci(recovered_stat(b1, p1, pt), idx)
            res = bootstrap_ci(paired_pct_change(bt, pt), idx)
            t_star = runs.set_index(["model", "corpus"]).t_star
            rows.append({"pair": pair, "corpus": corpus, "n_odd": odd.size,
                         "gap_t1_pct": 100 * (p1.sum() - b1.sum()) / b1.sum(),
                         "t_star_base": float(t_star[(base, corpus)]), "t_star_post": float(t_star[(post, corpus)]),
                         "recovered": rec[0], "recovered_lo": rec[1], "recovered_hi": rec[2],
                         "residual_pct": res[0], "residual_lo": res[1], "residual_hi": res[2]})
    return pd.DataFrame(rows)


def band(recovered: float) -> str:
    return "confidence shift" if recovered >= 0.70 else "belief change" if recovered <= 0.30 else "mixed"


def verdict(pt: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Headline per pair (mean recovered over the fresh corpora) and the rule's overall verdict."""
    fresh = pt[pt.corpus.isin(FRESH_CORPORA)]
    head = fresh.groupby("pair", sort=False).agg(recovered=("recovered", "mean"),
                                                 residual_pct=("residual_pct", "mean"),
                                                 gap_t1_pct=("gap_t1_pct", "mean")).reset_index()
    head["band"] = head.recovered.map(band)
    bands = set(head[head.pair.isin(BIG_PENALTY)].band)
    overall = bands.pop() if len(bands) == 1 else "mixed (the two big-penalty pairs fall in different bands)"
    return head, overall


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}%"


def summary_markdown(pt: pd.DataFrame, mt: pd.DataFrame, head: pd.DataFrame, overall: str,
                     runs: pd.DataFrame, recheck: pd.DataFrame) -> str:
    out = ["# exp02 summary (auto-generated by analyze_temperature.py)", "",
           "All numbers on odd-indexed windows; T* fit on even-indexed windows. "
           f"95% CIs: bootstrap over odd windows, {N_BOOT} resamples, seed {SEED}.", "",
           "## Headline (mean of the 3 fresh corpora)", "",
           "| pair | Δbpb post vs base at T=1 | gap recovered by T*_post | residual gap at own T* | band |",
           "|---|---|---|---|---|"]
    out += [f"| {r.pair} | {r.gap_t1_pct:+.1f}% | {pct(r.recovered)} | {r.residual_pct:+.1f}% | {r.band} |"
            for r in head.itertuples()]
    out += ["", f"**Rule verdict (Qwen3 and R1 pairs): {overall}.**", "",
            "## Per pair and corpus", "",
            "| pair | corpus | n odd | Δbpb at T=1 | T* base | T* post | gap recovered [95% CI] | "
            "residual gap at own T* [95% CI] |", "|---|---|---|---|---|---|---|---|"]
    out += [f"| {r.pair} | {r.corpus} | {r.n_odd} | {r.gap_t1_pct:+.1f}% | {r.t_star_base:.3f} | "
            f"{r.t_star_post:.3f} | {pct(r.recovered)} [{pct(r.recovered_lo)}, {pct(r.recovered_hi)}] | "
            f"{r.residual_pct:+.1f}% [{r.residual_lo:+.1f}%, {r.residual_hi:+.1f}%] |" for r in pt.itertuples()]
    out += ["", "## Per model and corpus", "",
            "| model | corpus | bpb T=1 | bpb T* | bpb gain | T* (even fit) | T* (odd fit, check) | at grid edge |",
            "|---|---|---|---|---|---|---|---|"]
    out += [f"| {r.model} | {r.corpus} | {r.bpb_t1:.4f} | {r.bpb_tstar:.4f} | {r.gain_pct:.2f}% | {r.t_star:.3f} | "
            f"{r.t_star_odd_fit:.3f} | {r.at_edge} |" for r in mt.itertuples()]
    base = mt[mt.role == "base"]
    out += ["", f"Base-model T* (even fit), all corpora: mean {base.t_star.mean():.3f}, "
            f"range {base.t_star.min():.3f}–{base.t_star.max():.3f}.", "",
            "## Sanity: T = 1.00 vs exp01 per-token RAW bits summed per window", "",
            "| model | corpus | windows | max per-window rel diff | total rel diff | scoring wall_s |",
            "|---|---|---|---|---|---|"]
    out += [f"| {r.model} | {r.corpus} | {r.n_windows} | {float(r.t1_vs_exp01_max_rel):.1e} | "
            f"{float(r.t1_vs_exp01_total_rel):.1e} | {r.wall_s} |" for r in runs.itertuples()]
    out += ["", "## Re-check: T = 1.00 vs RAW re-scored today with exp01's code path (score_temperature.py recheck)", "",
            "For the runs whose exp01 reference rows came from exp01's superseded in-process .half() path.", "",
            "| model | corpus | max per-window rel diff | total rel diff | re-scored total bits |", "|---|---|---|---|---|"]
    out += [f"| {r.model} | {r.corpus} | {float(r.t1_vs_rescore_max_rel):.1e} | {float(r.t1_vs_rescore_total_rel):.1e} | "
            f"{r.rescore_total_bits} |" for r in recheck.itertuples()]
    return "\n".join(out) + "\n"


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def figure(pt: pd.DataFrame, arrays: Arrays, windows: pd.DataFrame, runs: pd.DataFrame) -> Figure:
    """Left: gap recovered per pair and fresh corpus with 95% CIs. Right: bpb vs T for Qwen3 on wiki2026."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURFACE,
                                  gridspec_kw={"width_ratios": [2.6, 1]})
    fresh = pt[pt.corpus.isin(FRESH_CORPORA)]
    w, xs, labels = 0.26, [], []
    for i, (pair, _, _) in enumerate(PAIRS):
        for j, corpus in enumerate(FRESH_CORPORA):
            r = fresh[(fresh.pair == pair) & (fresh.corpus == corpus)].iloc[0]
            x = i + (j - 1) * w
            ax.bar(x, 100 * r.recovered, width=w * 0.88, color=POST_C, edgecolor=SURFACE, linewidth=1)
            ax.errorbar(x, 100 * r.recovered, yerr=[[100 * (r.recovered - r.recovered_lo)],
                                                    [100 * (r.recovered_hi - r.recovered)]],
                        color=INK2, linewidth=1, capsize=2)
            xs.append(x)
            labels.append(corpus.replace("2026", ""))
    for y, txt in ((70, "70%: mostly confidence shift"), (30, "30%: mostly belief change")):
        ax.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(len(PAIRS) - 0.45, y + 1.5, txt, ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7, color=INK2)
    for i, (pair, _, _) in enumerate(PAIRS):
        gap = fresh[fresh.pair == pair].gap_t1_pct.mean()
        ax.text(i, -0.09, f"{PAIR_LABEL[pair]}\n(Δbpb at T=1: {gap:+.1f}%)", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=INK)
    ax.set_ylabel("share of the T = 1 bpb gap removed by T*_post (%)", color=INK2, fontsize=8.5)
    ax.set_title("How much of the post-training bpb penalty one temperature removes (odd windows, 95% CI)",
                 fontsize=9.5, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    style_axis(ax)
    corpus = "wiki2026"
    for mid, color, name in (("Qwen/Qwen3-1.7B-Base", BASE_C, "Qwen3-1.7B-Base"),
                             ("Qwen/Qwen3-1.7B", POST_C, "Qwen3-1.7B (post)")):
        grid, tstar = arrays[(mid, corpus)]
        odd = odd_windows(len(grid))
        wb = windows[(windows.model == mid) & (windows.corpus == corpus)].sort_values("window").bytes.to_numpy()
        nbytes = wb[odd].sum()
        ax2.plot(T_GRID, grid[odd].sum(0) / nbytes, color=color, linewidth=2)
        t_star = float(runs.set_index(["model", "corpus"]).t_star[(mid, corpus)])
        ax2.plot([t_star], [tstar[odd].sum() / nbytes], "o", color=color, markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=2)
        below = color == BASE_C  # base label under its minimum, post label above, so neither hits a curve
        ax2.annotate(f"{name}\nT* = {t_star:.2f}", (t_star, tstar[odd].sum() / nbytes),
                     xytext=(8, -8) if below else (0, 10), textcoords="offset points", ha="left" if below else "center",
                     va="top" if below else "bottom", fontsize=7.5, color=INK)
    ax2.axvline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    lo, hi = ax2.get_ylim()
    ax2.set_ylim(lo - 0.08 * (hi - lo), hi)
    ax2.set_xlabel("temperature T", color=INK2, fontsize=8.5)
    ax2.set_ylabel("bits per byte, wiki2026 odd windows", color=INK2, fontsize=8.5)
    ax2.set_title("Qwen3 on wiki2026: bpb vs T", fontsize=9.5, color=INK, loc="left")
    ax2.grid(color=GRID, linewidth=0.8)
    style_axis(ax2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    return fig


# ----------------------------------------------------------------------------- effects
def load_inputs(results: Path) -> tuple[pd.DataFrame, pd.DataFrame, Arrays]:
    order = {m["model"]: i for i, m in enumerate(model_roles())}
    runs = pd.read_csv(results / "runs.csv")
    runs = runs.assign(_m=runs.model.map(order), _c=runs.corpus.map(CORPORA.index))
    runs = runs.sort_values(["_m", "_c"]).drop(columns=["_m", "_c"]).reset_index(drop=True)
    windows = pd.read_csv(results / "windows.csv")
    arrays = {(r.model, r.corpus): (np.load(results / "grid" / f"{model_slug(r.model)}__{r.corpus}.npy"),
                                    np.load(results / "grid" / f"{model_slug(r.model)}__{r.corpus}.tstar.npy"))
              for r in runs.itertuples()}
    return runs, windows, arrays


def main() -> None:
    res = Path("results")
    runs, windows, arrays = load_inputs(res)
    mt = model_table(runs, windows, arrays)
    pt = pair_table(runs, windows, arrays)
    head, overall = verdict(pt)
    rc = res / "t1_recheck.csv"
    recheck = pd.read_csv(rc) if rc.exists() else pd.DataFrame(columns=["model", "corpus"])
    (res / "summary.md").write_text(summary_markdown(pt, mt, head, overall, runs, recheck))
    figure(pt, arrays, windows, runs).savefig(res / "fig_temperature.png", dpi=160, facecolor=SURFACE)
    print((res / "summary.md").read_text())


if __name__ == "__main__":
    main()

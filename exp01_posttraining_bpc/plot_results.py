"""Turn results/*.csv|json into the figure and markdown tables.

Writes results/fig_bpb_pairs.png, results/summary.md (bpb + delta tables), results/sanity.md.
Pure functions over loaded data; I/O only in main().
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PAIR_ORDER = ["qwen2.5", "qwen3", "smollm2", "qwen2.5-math"]
PAIR_LABEL = {"qwen2.5": "Qwen2.5-1.5B\n→ Instruct", "qwen3": "Qwen3-1.7B-Base\n→ Qwen3-1.7B (post)",
              "smollm2": "SmolLM2-1.7B\n→ Instruct", "qwen2.5-math": "Qwen2.5-Math-1.5B\n→ R1-Distill"}
CORPUS_ORDER = ["wiki2026", "arxiv2026", "gutenberg2026", "enwik8_ctrl"]
BASE_C, POST_C = "#2a78d6", "#eb6834"   # dataviz reference palette slots 1, 2
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def bpb_table(df: pd.DataFrame) -> pd.DataFrame:
    t = df.pivot_table(index=["pair", "corpus"], columns=["role", "condition"], values="bpb")
    t.columns = [f"{r}_{c}" for r, c in t.columns]
    t = t.reset_index()
    for col in ("base_raw", "base_chat", "post_raw", "post_chat"):
        if col not in t:
            t[col] = float("nan")
    t["d_raw"] = t["post_raw"] - t["base_raw"]
    t["d_raw_pct"] = 100 * t["d_raw"] / t["base_raw"]
    t["d_chat_vs_base_raw_pct"] = 100 * (t["post_chat"] - t["base_raw"]) / t["base_raw"]
    t["d_chat_vs_base_chat_pct"] = 100 * (t["post_chat"] - t["base_chat"]) / t["base_chat"]
    t["pair"] = pd.Categorical(t["pair"], PAIR_ORDER)
    t["corpus"] = pd.Categorical(t["corpus"], CORPUS_ORDER)
    return t.sort_values(["pair", "corpus"]).reset_index(drop=True)


def fmt(x, nd=3, pct=False) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.1f}%" if pct else f"{x:.{nd}f}"


def summary_markdown(t: pd.DataFrame, df: pd.DataFrame) -> str:
    lines = ["| pair | corpus | base raw | post raw | Δ raw | Δ raw % | base chat | post chat | post chat vs base raw | post chat vs base chat |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r.pair} | {r.corpus} | {fmt(r.base_raw)} | {fmt(r.post_raw)} | {r.d_raw:+.3f} | "
                     f"{fmt(r.d_raw_pct, pct=True)} | {fmt(r.base_chat)} | {fmt(r.post_chat)} | "
                     f"{fmt(r.d_chat_vs_base_raw_pct, pct=True)} | {fmt(r.d_chat_vs_base_chat_pct, pct=True)} |")
    fresh = t[t.corpus != "enwik8_ctrl"]
    mean = fresh.groupby("pair", observed=True)[["d_raw_pct", "d_chat_vs_base_raw_pct"]].mean()
    lines += ["", "Mean over the 3 fresh corpora (excl. enwik8 control):", "",
              "| pair | Δ raw % | post chat vs base raw % |", "|---|---|---|"]
    for p, r in mean.iterrows():
        lines.append(f"| {p} | {r.d_raw_pct:+.1f}% | {fmt(r.d_chat_vs_base_raw_pct, pct=True)} |")
    lines.append(f"| **all pairs** | {fresh.d_raw_pct.mean():+.1f}% | {fresh.d_chat_vs_base_raw_pct.mean():+.1f}% |")
    units = df[["model", "condition", "corpus", "bytes", "chars", "tokens", "total_bits", "bpb", "bpc",
                "tokens_per_byte", "bits_per_token", "n_windows", "dtype", "wall_s"]]
    lines += ["", "Full rows (unit conversions: bpb = total_bits/bytes, bpc = total_bits/chars, "
              "bits_per_token = total_bits/tokens):", "", units.to_markdown(index=False)]
    return "\n".join(lines) + "\n"


def docstart_markdown(main: pd.DataFrame, abl: pd.DataFrame) -> str:
    """RAW Δ% under the main doc-start (special token) vs a plain '\\n\\n' doc-start."""
    m = main[main.condition == "raw"].merge(abl[["model", "corpus", "bpb"]], on=["model", "corpus"],
                                             suffixes=("_special", "_nl"))
    lines = ["", "## Doc-start robustness (RAW): special-token doc-start (main table) vs plain \"\\n\\n\" prefix", "",
             "| pair | corpus | base special | base \\n\\n | post special | post \\n\\n | Δ% special | Δ% \\n\\n |",
             "|---|---|---|---|---|---|---|---|"]
    for (pair, corpus), g in m.groupby(["pair", "corpus"], sort=False):
        b, p = g[g.role == "base"].iloc[0], g[g.role == "post"].iloc[0]
        lines.append(f"| {pair} | {corpus} | {b.bpb_special:.3f} | {b.bpb_nl:.3f} | {p.bpb_special:.3f} | {p.bpb_nl:.3f} | "
                     f"{100 * (p.bpb_special - b.bpb_special) / b.bpb_special:+.1f}% | "
                     f"{100 * (p.bpb_nl - b.bpb_nl) / b.bpb_nl:+.1f}% |")
    return "\n".join(lines) + "\n"


def plot(t: pd.DataFrame, path: Path) -> None:
    corpora = [c for c in CORPUS_ORDER if c in set(t.corpus.astype(str))]
    fig, axes = plt.subplots(1, len(corpora), figsize=(4.2 * len(corpora), 4.6), sharey=True, facecolor=SURFACE)
    axes = [axes] if len(corpora) == 1 else axes
    w = 0.2
    for ax, corpus in zip(axes, corpora, strict=True):
        sub = t[t.corpus == corpus].set_index("pair").reindex(PAIR_ORDER)
        x = range(len(PAIR_ORDER))
        specs = [("base_raw", -1.5, BASE_C, None), ("base_chat", -0.5, BASE_C, "///"),
                 ("post_raw", 0.5, POST_C, None), ("post_chat", 1.5, POST_C, "///")]
        for col, off, color, hatch in specs:
            vals = sub[col].values
            kw = dict(color=color, edgecolor=SURFACE, linewidth=1.5)
            if hatch:
                kw = dict(facecolor=SURFACE, edgecolor=color, hatch=hatch, linewidth=1.2)
            ax.bar([i + off * w for i in x], vals, width=w * 0.92, **kw)
        for i, p in enumerate(PAIR_ORDER):
            d = sub.loc[p, "d_raw_pct"]
            top = sub.loc[p, ["base_raw", "base_chat", "post_raw", "post_chat"]].max()
            if pd.notna(d):
                ax.text(i, top + 0.02, f"{d:+.1f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_title(corpus + (" (control, old text)" if corpus == "enwik8_ctrl" else ""), fontsize=10.5, color=INK)
        ax.set_xticks(list(x))
        ax.set_xticklabels([PAIR_LABEL[p] for p in PAIR_ORDER], fontsize=7.2, color=INK2)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel("bits per byte (lower = better compression)", color=INK2)
    from matplotlib.patches import Patch
    handles = [Patch(color=BASE_C, label="base, raw"), Patch(facecolor=SURFACE, edgecolor=BASE_C, hatch="///", label="base, chat template"),
               Patch(color=POST_C, label="post-trained, raw"), Patch(facecolor=SURFACE, edgecolor=POST_C, hatch="///", label="post-trained, chat template")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Label above each pair: Δbpb post − base (raw), % of base", y=0.905, fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=160, facecolor=SURFACE)


def sanity_markdown(sanity: dict, checks: list[dict], tok: dict, audit: list[dict]) -> str:
    g = sanity.get("gpt2_raw", {})
    wc = sanity.get("gpt2_window_checks", {})
    out = ["# Sanity checks (auto-generated by plot_results.py from sanity.json, model_checks.jsonl, tokcheck.json)", "",
           f"device={sanity.get('device')} torch={sanity.get('torch')}", "",
           "## (a) GPT-2 (124M) raw bpb, window 1023 tokens + <|endoftext|>", "",
           "| corpus | bytes | tokens | bpb | bpc | bits/token | first-token-of-window share of bpb |", "|---|---|---|---|---|---|---|"]
    for n, r in g.items():
        out.append(f"| {n} | {r['bytes']} | {r['tokens']} | {r['bpb']} | {r['bpc']} | {r['bits_per_token']} | {r['bpb_first_token_of_windows']} |")
    out += ["", "## (b) labels-loss equality, (c) determinism, (d) fp16 vs fp32 — window 0 of the first corpus, every model", "",
            "| model | window tokens | fp32 bits | fp16 bits | fp16 vs fp32 rel | repeat fp16 bits | deterministic | HF loss×n bits | rel diff to HF | dtype used |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for c in [wc, *checks]:
        if "error" in c:
            out.append(f"| {c['model']} | ERROR: {c['error'][:200]} |||||||||")
            continue
        out.append(f"| {c['model']} | {c['window_tokens']} | {c['fp32_bits']} | {c['fp16_bits']} | {c['fp16_vs_fp32_rel']:.2e} | "
                   f"{c['repeat_bits']} | {c['deterministic']} | {c['hf_labels_loss_bits']} | {c['labels_rel_diff']:.2e} | "
                   f"{'float16' if c['fp16_ok'] else 'float32'} |")
    out += ["", "## Document-start token and chat prefix per model", "", "| model | raw doc-start | chat prefix tokens | chat prefix (rendered) |", "|---|---|---|---|"]
    for c in checks:
        if "error" not in c:
            pre = (c.get("chat_prefix") or "—").replace("\n", "\\n").replace("|", "\\|")
            out.append(f"| {c['model']} | {c['doc_start']} | {c.get('chat_prefix_tokens') or '—'} | `{pre}` |")
    ab = sanity.get("r1_prefix_ablation_arxiv2026_bpb")
    if ab:
        out += ["", "## (e) Doc-start ablation, DeepSeek-R1-Distill-Qwen-1.5B raw on arxiv2026", "",
                f"BOS <｜begin▁of▁sentence｜> (151646, used in main table): {ab['bos_151646']} bpb; "
                f"id 151643 (<|endoftext|> of the Qwen base): {ab['eot_151643']} bpb."]
    out += ["", "## (f) Tokenizer identity per pair (add_special_tokens=False, full corpora)", "",
            "| pair | base | post | identical ids on all corpora | round-trip decode == text |", "|---|---|---|---|---|"]
    for p, r in tok.get("pairs", {}).items():
        rt = all(v["roundtrip_base"] and v["roundtrip_post"] for v in r["corpora"].values())
        out.append(f"| {p} | {r['base']} | {r['post']} | {r['all_identical']} | {rt} |")
    out += ["", f"Special-token strings found inside corpora: {tok.get('special_strings_in_corpora')}"]
    if audit:
        out += ["", "## (g) Targeted float32 re-scoring of suspicious windows (score_bpb.py audit)", "",
                "Windows picked because they held extreme per-token surprisals (>35 bits) or outlier window means.", "",
                "| model | condition | corpus | window | device | fp32 bits | saved fp16 bits | rel diff | tokens >35 bits fp32 / fp16 | max per-token diff |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for a in audit:
            out.append(f"| {a['model']} | {a['condition']} | {a['corpus']} | {a['window']} | {a['device']} | {a['bits_audit']} | "
                       f"{a['bits_fp16_saved']} | {a['rel_diff']:+.1e} | {a['tokens_gt35_audit']} / {a['tokens_gt35_fp16']} | "
                       f"{a['max_abs_token_diff']} |")
    return "\n".join(out) + "\n"


def main() -> None:
    res = Path("results")
    df = pd.read_csv(res / "bpb.csv")
    t = bpb_table(df)
    plot(t, res / "fig_bpb_pairs.png")
    abl = res / "ablation_docstart.csv"
    (res / "summary.md").write_text(summary_markdown(t, df) +
                                    (docstart_markdown(df, pd.read_csv(abl)) if abl.exists() else ""))
    checks = [json.loads(ln) for ln in (res / "model_checks.jsonl").read_text().splitlines() if ln.strip()]
    sanity = json.loads((res / "sanity.json").read_text()) if (res / "sanity.json").exists() else {}
    tok = json.loads((res / "tokcheck.json").read_text()) if (res / "tokcheck.json").exists() else {}
    ap = res / "fp32_audit.jsonl"
    audit = [json.loads(ln) for ln in ap.read_text().splitlines() if ln.strip()] if ap.exists() else []
    (res / "sanity.md").write_text(sanity_markdown(sanity, checks, tok, audit))
    print((res / "summary.md").read_text()[:4000])


if __name__ == "__main__":
    main()

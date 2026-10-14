"""Audit: is the base-vs-post gap a whole-text effect or a window-start / prefix (attention-sink) effect?

Rebuilds each run's per-token window positions (tokenizer only, no model), joins them with the saved
per-token bits (results/tokens/*.npy) and reports bits/token by position bucket. Within a pair the
tokenizers are identical (results/tokcheck.json), so base and post are scored on exactly the same
tokens and bits/token deltas are directly comparable.

Writes results/positions.csv and results/positions.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cilab`

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from cilab.io import load_corpora, load_token_bits
from cilab.stats import bootstrap_ci, bucket_stats, paired_pct_change, window_sums
from cilab.text import window_positions


def main() -> None:
    res = Path("results")
    df = pd.read_csv(res / "bpb.csv")
    corpora = load_corpora(Path("data"), sorted(df.corpus.unique()), None)
    rows, toks = [], {}
    for r in df.itertuples():
        bits = load_token_bits(res, r.model, r.condition, r.corpus)
        tok = toks.setdefault(r.model, AutoTokenizer.from_pretrained(r.model))
        pos = window_positions(tok, corpora[r.corpus], int(r.window), r.condition)
        if pos.size != bits.size:
            raise ValueError(f"{r.model} {r.condition} {r.corpus}: {pos.size} positions vs {bits.size} bits")
        rows.append({"model": r.model, "pair": r.pair, "role": r.role, "condition": r.condition,
                     "corpus": r.corpus, "bpb": r.bpb, **bucket_stats(bits, pos)})
    p = pd.DataFrame(rows)
    p.to_csv(res / "positions.csv", index=False)
    late = p.pivot_table(index=["pair", "corpus"], columns=["role", "condition"], values="bpt_pos128+")
    early = p.pivot_table(index=["pair", "corpus"], columns=["role", "condition"], values="bpt_pos0-0")
    lines = ["# Position audit: bits/token by position within window", "",
             "bpt = mean bits per token. 'pos128+' = tokens with >= 128 tokens of in-window context; "
             "'pos0' = first text token of each window (conditioned only on the doc-start token or chat prefix).", "",
             "| pair | corpus | base raw pos128+ | post raw pos128+ | Δ% raw pos128+ | post chat pos128+ | Δ% post chat vs base raw pos128+ | base raw pos0 | post raw pos0 | post chat pos0 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (pair, corpus), r in late.iterrows():
        br, pr = r.get(("base", "raw"), np.nan), r.get(("post", "raw"), np.nan)
        pc = r.get(("post", "chat"), np.nan)
        e = early.loc[(pair, corpus)]
        lines.append(f"| {pair} | {corpus} | {br:.3f} | {pr:.3f} | {100 * (pr - br) / br:+.1f}% | {pc:.3f} | "
                     f"{100 * (pc - br) / br:+.1f}% | {e.get(('base', 'raw'), np.nan):.2f} | "
                     f"{e.get(('post', 'raw'), np.nan):.2f} | {e.get(('post', 'chat'), np.nan):.2f} |")
    lines += concentration_lines(df, res)
    lines += bootstrap_lines(df, res, corpora, toks)
    (res / "positions.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def concentration_lines(df: pd.DataFrame, res: Path) -> list[str]:
    """Where does the post-minus-base gap come from? Same tokens within a pair (identical tokenizers)."""
    out = ["", "# Concentration of the gap (RAW, token-aligned within each pair)", "",
           "Δ = post − base bits on the same token. 'top1% share' = fraction of the total Δ contributed by the 1% "
           "of tokens with the largest Δ; 'Δ>10 share' = fraction from tokens where post pays >10 extra bits.", "",
           "| pair | corpus | Δ total bits | Δ % | tokens where post worse | median Δ (bits) | top1% share | Δ>10 share |",
           "|---|---|---|---|---|---|---|---|"]
    raw = df[df.condition == "raw"]
    for (pair, corpus), g in raw.groupby(["pair", "corpus"], sort=False):
        try:
            b = g[g.role == "base"].iloc[0]
            p = g[g.role == "post"].iloc[0]
        except IndexError:
            continue
        base_bits = load_token_bits(res, b.model, "raw", b.corpus)
        d = load_token_bits(res, p.model, "raw", p.corpus) - base_bits
        top = np.sort(d)[::-1][: max(1, d.size // 100)]
        out.append(f"| {pair} | {corpus} | {d.sum():+.0f} | {100 * d.sum() / base_bits.sum():+.1f}% | "
                   f"{100 * (d > 0).mean():.0f}% | {np.median(d):+.3f} | {top.sum() / d.sum():.2f} | "
                   f"{d[d > 10].sum() / d.sum():.2f} |")
    return out


def bootstrap_lines(df: pd.DataFrame, res: Path, corpora: dict[str, str],
                    toks: dict[str, PreTrainedTokenizerBase], n_boot: int = 5000) -> list[str]:
    """Paired bootstrap over windows (same windows for base and post) -> 95% CI of Δbpb %.

    One generator (seed 0) is consumed in table order, so every row's resample differs; kept as in the
    original exp01 run so results/positions.md stays byte-identical."""
    rng = np.random.default_rng(0)
    out = ["", "# Uncertainty: paired bootstrap over windows (5000 resamples, seed 0)", "",
           "Resamples whole windows (the same windows for base and post), so the CI reflects text-sample "
           "variation within each corpus, not variation across corpora or seeds.", "",
           "| pair | corpus | condition (post vs base) | n windows | Δ% | 95% CI |", "|---|---|---|---|---|---|"]
    for (pair, corpus), g in df.groupby(["pair", "corpus"], sort=False):
        for pc, bc in (("raw", "raw"), ("chat", "raw"), ("chat", "chat")):
            b = g[(g.role == "base") & (g.condition == bc)]
            p = g[(g.role == "post") & (g.condition == pc)]
            if b.empty or p.empty:
                continue
            b, p = b.iloc[0], p.iloc[0]
            wb = window_sums(load_token_bits(res, b.model, b.condition, b.corpus),
                             window_positions(toks[b.model], corpora[corpus], int(b.window), b.condition))
            wp = window_sums(load_token_bits(res, p.model, p.condition, p.corpus),
                             window_positions(toks[p.model], corpora[corpus], int(p.window), p.condition))
            idx = rng.integers(0, wb.size, size=(n_boot, wb.size))
            point, lo, hi = bootstrap_ci(paired_pct_change(wb, wp), idx)
            out.append(f"| {pair} | {corpus} | {pc} vs {bc} | {wb.size} | {point:+.1f}% | "
                       f"[{lo:+.1f}%, {hi:+.1f}%] |")
    return out


if __name__ == "__main__":
    main()

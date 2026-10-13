"""Audit: is the base-vs-post gap a whole-text effect or a window-start / prefix (attention-sink) effect?

Rebuilds each run's per-token window positions (tokenizer only, no model), joins them with the saved
per-token bits (results/tokens/*.npy) and reports bits/token by position bucket. Within a pair the
tokenizers are identical (results/tokcheck.json), so base and post are scored on exactly the same
tokens and bits/token deltas are directly comparable.

Writes results/positions.csv and results/positions.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from score_bpb import encode, load_corpora, make_windows

BUCKETS = ((0, 1), (1, 16), (16, 128), (128, 10**9))


def positions(tok, text: str, window: int, condition: str) -> np.ndarray:
    wins = make_windows(tok, encode(tok, text), text, window)
    if condition == "raw":
        return np.concatenate([np.arange(e - b) for b, e, _ in wins])
    return np.concatenate([np.arange(len(encode(tok, s))) for _, _, s in wins])


def bucket_stats(bits: np.ndarray, pos: np.ndarray) -> dict:
    out = {}
    for lo, hi in BUCKETS:
        m = (pos >= lo) & (pos < hi)
        name = f"pos{lo}" + (f"-{hi - 1}" if hi < 10**9 else "+")
        out[f"bpt_{name}"] = float(bits[m].mean()) if m.any() else float("nan")
        out[f"share_bits_{name}"] = float(bits[m].sum() / bits.sum())
    return out


def main() -> None:
    res = Path("results")
    df = pd.read_csv(res / "bpb.csv")
    corpora = load_corpora(Path("data"), sorted(df.corpus.unique()), None)
    rows, toks = [], {}
    for r in df.itertuples():
        f = res / "tokens" / f"{r.model.replace('/', '__')}__{r.condition}__{r.corpus}.npy"
        bits = np.load(f).astype(np.float64)
        tok = toks.setdefault(r.model, AutoTokenizer.from_pretrained(r.model))
        pos = positions(tok, corpora[r.corpus], int(r.window), r.condition)
        if pos.size != bits.size:
            raise ValueError(f"{f.name}: {pos.size} positions vs {bits.size} bits")
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
        load = lambda r: np.load(res / "tokens" / f"{r.model.replace('/', '__')}__raw__{r.corpus}.npy").astype(np.float64)  # noqa: E731
        d = load(p) - load(b)
        top = np.sort(d)[::-1][: max(1, d.size // 100)]
        out.append(f"| {pair} | {corpus} | {d.sum():+.0f} | {100 * d.sum() / load(b).sum():+.1f}% | "
                   f"{100 * (d > 0).mean():.0f}% | {np.median(d):+.3f} | {top.sum() / d.sum():.2f} | "
                   f"{d[d > 10].sum() / d.sum():.2f} |")
    return out


def window_sums(bits: np.ndarray, pos: np.ndarray) -> np.ndarray:
    starts = np.flatnonzero(pos == 0)
    return np.add.reduceat(bits, starts)


def bootstrap_lines(df: pd.DataFrame, res: Path, corpora: dict, toks: dict, n_boot: int = 5000) -> list[str]:
    """Paired bootstrap over windows (same windows for base and post) -> 95% CI of Δbpb %."""
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
            load = lambda r: np.load(res / "tokens" / f"{r.model.replace('/', '__')}__{r.condition}__{r.corpus}.npy").astype(np.float64)  # noqa: E731
            wb = window_sums(load(b), positions(toks[b.model], corpora[corpus], int(b.window), b.condition))
            wp = window_sums(load(p), positions(toks[p.model], corpora[corpus], int(p.window), p.condition))
            idx = rng.integers(0, wb.size, size=(n_boot, wb.size))
            d = 100 * (wp[idx].sum(1) - wb[idx].sum(1)) / wb[idx].sum(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            out.append(f"| {pair} | {corpus} | {pc} vs {bc} | {wb.size} | {100 * (wp.sum() - wb.sum()) / wb.sum():+.1f}% | "
                       f"[{lo:+.1f}%, {hi:+.1f}%] |")
    return out


if __name__ == "__main__":
    main()

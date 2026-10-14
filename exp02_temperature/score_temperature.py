"""exp02: does one scalar temperature recover the post-training compression gap on fresh text?

  python score_temperature.py run [--models ...] [--corpora ...] [--max-bytes N]

RAW condition exactly as exp01: same corpora (../exp01_posttraining_bpc/data), the same 1024-token
windows and the same document-start token. For each model (one process per model) and corpus:
  * one fp16 forward pass per window; from its float32 logits, the window's total bits for every T
    in T_GRID (0.50, 0.55, …, 1.60): bits_T = −log2 softmax(logits / T)[target]. Logits are not stored.
  * even-indexed windows are scored first and T* is fit on them (grid argmin, then a 3-point
    parabola in log T); odd-indexed windows are then scored on the grid and at T* exactly, from the
    same single forward pass.
  * sanity, before the row is written: grid bits at T = 1.00 vs exp01's saved per-token RAW bits
    summed per window (max relative difference over windows).

Writes results/grid/<model>__<corpus>.npy        (n_windows × 23) bits per window and T
       results/grid/<model>__<corpus>.tstar.npy  (n_windows,) bits at T*; NaN on even windows
       results/windows.csv                        tokens and UTF-8 bytes per window
       results/runs.csv                           one row per (model, corpus): T*, sanity, wall_s
Resumable: (model, corpus) pairs already in runs.csv are skipped.

  python score_temperature.py recheck --models ... --corpora ...
      Re-scores RAW today with exp01's code path (cilab.score.score_raw) and compares it per window with
      the grid at T = 1 -> results/t1_recheck.csv. Used for the 4 runs whose exp01 reference rows were
      scored by exp01's superseded in-process .half() path (fp16 rotary buffers).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cilab`

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedModel

from cilab.calib import T_GRID, bits_at_temperatures, fit_temperature
from cilab.io import append_rows, existing_keys, load_corpora, load_token_bits, read_tokenizer_config
from cilab.pairs import CORPORA, model_roles, model_slug
from cilab.score import load_model, pick_device, score_raw, target_logits
from cilab.stats import window_totals
from cilab.text import Window, doc_start_token, encode, make_windows

WINDOW = 1024
RUN_COLS = ("model", "pair", "role", "corpus", "n_windows", "tokens", "bytes", "doc_start", "t_star",
            "at_edge", "t1_vs_exp01_max_rel", "t1_vs_exp01_total_rel", "wall_s", "max_bytes")
WINDOW_COLS = ("model", "corpus", "window", "tokens", "bytes")
RECHECK_COLS = ("model", "corpus", "t1_vs_rescore_max_rel", "t1_vs_rescore_total_rel", "rescore_total_bits")


# ----------------------------------------------------------------------------- pure helpers
def split_even_odd(n: int) -> tuple[list[int], list[int]]:
    return list(range(0, n, 2)), list(range(1, n, 2))


def window_lengths(wins: list[Window]) -> list[int]:
    return [e - b for b, e, _ in wins]


def rel_diffs(a: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """(max over windows, total) relative difference of a vs ref."""
    return float(np.max(np.abs(a - ref) / ref)), float(abs(a.sum() - ref.sum()) / ref.sum())


def window_rows(model_id: str, corpus: str, wins: list[Window]) -> list[dict[str, object]]:
    return [{"model": model_id, "corpus": corpus, "window": k, "tokens": e - b, "bytes": len(s.encode("utf-8"))}
            for k, (b, e, s) in enumerate(wins)]


def grid_path(results: Path, model_id: str, corpus: str, suffix: str = "") -> Path:
    return results / "grid" / f"{model_slug(model_id)}__{corpus}{suffix}.npy"


# ----------------------------------------------------------------------------- effects
def window_bits(model: PreTrainedModel, ids: list[int], win: Window, start_id: int,
                temps: tuple[float, ...], device: str) -> np.ndarray:
    """One forward pass over [start_id, *window]; total window bits at each temperature."""
    b, e, _ = win
    logits, targets = target_logits(model, [start_id, *ids[b:e]], e - b, device)
    return bits_at_temperatures(logits, targets, temps)


def score_corpus(model: PreTrainedModel, ids: list[int], wins: list[Window], start_id: int,
                 device: str) -> dict[str, object]:
    """Grid bits for every window, T* fit on even windows, exact bits at T* on odd windows."""
    evens, odds = split_even_odd(len(wins))
    even_rows = {k: window_bits(model, ids, wins[k], start_id, T_GRID, device) for k in evens}
    t_star, at_edge = fit_temperature(T_GRID, np.sum(list(even_rows.values()), axis=0))
    odd_rows = {k: window_bits(model, ids, wins[k], start_id, (*T_GRID, t_star), device) for k in odds}
    rows = {**even_rows, **{k: r[:-1] for k, r in odd_rows.items()}}
    grid = np.stack([rows[k] for k in range(len(wins))])
    tstar_bits = np.array([odd_rows[k][-1] if k in odd_rows else np.nan for k in range(len(wins))])
    return {"grid": grid, "tstar_bits": tstar_bits, "t_star": t_star, "at_edge": at_edge}


def run_model(spec: dict[str, str], corpora: dict[str, str], args: argparse.Namespace) -> None:
    out = Path(args.results)
    (out / "grid").mkdir(parents=True, exist_ok=True)
    mid = spec["model"]
    done = existing_keys(out / "runs.csv", ("model", "corpus"))
    todo = {n: t for n, t in corpora.items() if (mid, n) not in done}
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(mid)
    start_id, start_str = doc_start_token(tok, read_tokenizer_config(mid))
    device = pick_device()
    model = load_model(mid, torch.float16, device)
    for n, text in todo.items():
        ids = encode(tok, text)
        wins = make_windows(tok, ids, text, WINDOW)
        t0 = time.time()
        res = score_corpus(model, ids, wins, start_id, device)
        wall_s = round(time.time() - t0, 1)
        t1 = res["grid"][:, T_GRID.index(1.0)]
        if args.max_bytes:  # exp01 scored the full corpora; no like-for-like reference
            max_rel, total_rel = float("nan"), float("nan")
        else:
            ref = window_totals(load_token_bits(Path(args.exp01_results), mid, "raw", n), window_lengths(wins))
            max_rel, total_rel = rel_diffs(t1, ref)
        np.save(grid_path(out, mid, n), res["grid"])
        np.save(grid_path(out, mid, n, ".tstar"), res["tstar_bits"])
        append_rows(out / "windows.csv", window_rows(mid, n, wins), WINDOW_COLS)
        row = {**spec, "corpus": n, "n_windows": len(wins), "tokens": len(ids),
               "bytes": len(text.encode("utf-8")), "doc_start": f"{start_str} ({start_id})",
               "t_star": round(res["t_star"], 5), "at_edge": res["at_edge"],
               "t1_vs_exp01_max_rel": f"{max_rel:.2e}", "t1_vs_exp01_total_rel": f"{total_rel:.2e}",
               "wall_s": wall_s, "max_bytes": args.max_bytes or ""}
        append_rows(out / "runs.csv", [row], RUN_COLS)
        print(json.dumps({k: row[k] for k in ("model", "corpus", "t_star", "at_edge", "t1_vs_exp01_max_rel",
                                              "wall_s")}), flush=True)
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def cmd_run(args: argparse.Namespace) -> None:
    specs = [s for s in model_roles() if not args.models or s["model"] in args.models]
    if len(specs) > 1:  # one process per model: weights are guaranteed freed between models
        for spec in specs:
            cmd = [sys.executable, os.path.abspath(__file__), "run", "--models", spec["model"],
                   "--corpora", *args.corpora, "--data", args.data, "--results", args.results,
                   "--exp01-results", args.exp01_results]
            cmd += ["--max-bytes", str(args.max_bytes)] if args.max_bytes else []
            rc = subprocess.run(cmd, check=False).returncode
            print(json.dumps({"model": spec["model"], "returncode": rc}), flush=True)
        return
    run_model(specs[0], load_corpora(Path(args.data), args.corpora, args.max_bytes), args)


def cmd_recheck(args: argparse.Namespace) -> None:
    """Re-score RAW with exp01's own code path (cilab.score.score_raw) today and compare per window with
    the grid at T = 1. For runs whose exp01 reference predates the direct-fp16 load."""
    out = Path(args.results)
    corpora = load_corpora(Path(args.data), args.corpora, args.max_bytes)
    device = pick_device()
    for mid in args.models:
        tok = AutoTokenizer.from_pretrained(mid)
        start_id, _ = doc_start_token(tok, read_tokenizer_config(mid))
        model = load_model(mid, torch.float16, device)
        rows = []
        for n, text in corpora.items():
            ids = encode(tok, text)
            wins = make_windows(tok, ids, text, WINDOW)
            ref = window_totals(score_raw(model, wins, ids, start_id, device), window_lengths(wins))
            max_rel, total_rel = rel_diffs(np.load(grid_path(out, mid, n))[:, T_GRID.index(1.0)], ref)
            rows.append({"model": mid, "corpus": n, "t1_vs_rescore_max_rel": f"{max_rel:.2e}",
                         "t1_vs_rescore_total_rel": f"{total_rel:.2e}", "rescore_total_bits": round(float(ref.sum()), 3)})
            print(json.dumps(rows[-1]), flush=True)
        append_rows(out / "t1_recheck.csv", rows, RECHECK_COLS)
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "recheck"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--corpora", nargs="*", default=list(CORPORA))
    ap.add_argument("--data", default="../exp01_posttraining_bpc/data")
    ap.add_argument("--exp01-results", default="../exp01_posttraining_bpc/results")
    ap.add_argument("--results", default="results")
    ap.add_argument("--max-bytes", type=int, default=None, help="cut every corpus to this many bytes")
    args = ap.parse_args()
    torch.manual_seed(0)
    {"run": cmd_run, "recheck": cmd_recheck}[args.cmd](args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

"""exp03: fresh-text bits per byte of every post-training stage (cilab.stages.STAGES), RAW condition.

  python exp03_staged_drift/score_stages.py tokcheck [--data D] [--results R]
  python exp03_staged_drift/score_stages.py run [--models ...] [--corpora ...] [--device cuda] [--data D] [--results R]

Mechanics are exp01's RAW condition, through the same cilab helpers:
  * corpus -> ids with add_special_tokens=False; consecutive 1024-token windows that never split a character;
  * input = [doc_start, *window]; every text token is scored; bpb = Σ bits / UTF-8 bytes of the corpus;
  * doc_start = BOS if the tokenizer config adds one (R1-Distill), else <|endoftext|> (Qwen2.5, OLMo-2), else EOS;
  * -log2 p from float32 logits; weights loaded directly in float16 (float32 on CPU).

`run`, per model (one child process each, so the weights are freed between models):
  1. float32 bits of window 0 of the first pending corpus (load, score, free);
  2. float16 load; the same window in float16, twice (determinism). If fp16 is not finite or differs from
     fp32 by >= FP16_TOL relative, the model is scored in float32 instead; a corpus whose fp16 bits are
     not finite is re-scored in float32;
  3. each corpus not yet in bpb.csv: per-token bits -> tokens/<model>__raw__<corpus>.npy (float32),
     per-window tokens and bytes -> windows.csv, one row -> bpb.csv (exp01 columns + stage columns),
     one printed line with bpb and wall time; per-model checks -> model_checks.jsonl.
Resumable and idempotent: (model, condition, corpus) rows already in bpb.csv are skipped.

`tokcheck`: every tokenizer group (cilab.stages.tokenizer_groups) must encode every corpus to identical
ids; also round-trip, vocabulary size and the doc-start token per model -> tokcheck.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cilab`

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

from cilab.io import append_rows, existing_keys, load_corpora, read_tokenizer_config, token_bits_path
from cilab.pairs import CORPORA
from cilab.score import load_model, pick_device, release_device_memory, score_raw, score_tokens, summarize
from cilab.stages import STAGES, Stage, stage_of, tokenizer_groups
from cilab.text import Window, adds_bos, decode, doc_start_token, encode, make_windows

WINDOW = 1024
CONDITION = "raw"
FP16_TOL = 5e-3  # max relative fp16-vs-fp32 difference of window-0 bits (as exp01)
EXP01_COLS = ("model", "pair", "role", "condition", "corpus", "bytes", "chars", "tokens", "total_bits",
              "bpb", "bpc", "wall_s", "tokens_per_byte", "bits_per_token", "n_windows", "window", "dtype",
              "doc_start")
CSV_COLS = (*EXP01_COLS, "family", "stage", "parent", "doc_start_id", "ids_sha1", "gpu", "torch")
KEY_COLS = ("model", "condition", "corpus")
WINDOW_COLS = ("model", "corpus", "window", "tokens", "bytes")


# ----------------------------------------------------------------------------- pure helpers
def ids_digest(ids: Sequence[int]) -> str:
    """Short SHA-1 of a token-id sequence (int64 bytes), for cross-model identity checks."""
    return hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:16]


def choose_dtype(fp16_bits: float, fp32_bits: float | None, tol: float = FP16_TOL) -> str:
    """'float16' when fp16 bits are finite and agree with fp32 within tol (or fp32 could not be run),
    else 'float32'."""
    if not np.isfinite(fp16_bits):
        return "float32"
    if fp32_bits is None:
        return "float16"
    return "float16" if abs(fp16_bits - fp32_bits) / fp32_bits < tol else "float32"


def pending_corpora(model: str, corpora: Sequence[str], done: set[tuple[str, ...]]) -> list[str]:
    """Corpora of `model` not yet in the results CSV, in the given order."""
    return [c for c in corpora if (model, CONDITION, c) not in done]


def window_rows(model: str, corpus: str, wins: Sequence[Window]) -> list[dict[str, object]]:
    return [{"model": model, "corpus": corpus, "window": k, "tokens": e - b, "bytes": len(s.encode("utf-8"))}
            for k, (b, e, s) in enumerate(wins)]


def stage_meta(stage: Stage) -> dict[str, object]:
    """CSV fields from the stage table; exp01's `pair`/`role` carry the family and the stage label."""
    return {"model": stage.model, "pair": stage.family, "role": stage.stage, "family": stage.family,
            "stage": stage.stage, "parent": stage.parent or ""}


def compare_ids(reference: Mapping[str, Sequence[int]], other: Mapping[str, Sequence[int]]) -> dict[str, bool]:
    """{corpus: other's ids == reference's ids}, over the reference's corpora."""
    return {c: list(other[c]) == list(reference[c]) for c in reference}


# ----------------------------------------------------------------------------- effects
def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device == "cuda" else device


def dtype_of(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[name]


def fp32_reference(model_id: str, seq: list[int], n_score: int, device: str) -> tuple[float | None, str]:
    """(fp32 bits of the last n_score tokens, error). The fp32 weights exist only inside this call."""
    bits, err = None, ""
    try:
        bits = float(score_tokens(load_model(model_id, torch.float32, device), seq, n_score, device).sum())
    except (RuntimeError, MemoryError) as e:  # torch.OutOfMemoryError is a RuntimeError
        err = f"{type(e).__name__}: {e}"[:300]
    release_device_memory()
    return bits, err


def run_model(stage: Stage, corpora: dict[str, str], out: Path, device: str) -> dict[str, object]:
    """Score every pending corpus of one model; returns the model-check record."""
    mid = stage.model
    todo = pending_corpora(mid, list(corpora), existing_keys(out / "bpb.csv", KEY_COLS))
    if not todo:
        return {"model": mid, "skipped": "all corpora already in bpb.csv"}
    t_model = time.time()
    tok = AutoTokenizer.from_pretrained(mid)
    cfg = read_tokenizer_config(mid)
    start_id, start_str = doc_start_token(tok, cfg)
    ids = {n: encode(tok, corpora[n]) for n in todo}
    wins = {n: make_windows(tok, ids[n], corpora[n], WINDOW) for n in todo}
    b, e, _ = wins[todo[0]][0]
    seq = [start_id, *ids[todo[0]][b:e]]
    t0 = time.time()
    fp32_bits, fp32_err = fp32_reference(mid, seq, e - b, device)
    fp32_s = round(time.time() - t0, 1)
    t0 = time.time()
    model = load_model(mid, torch.float16 if device != "cpu" else torch.float32, device)
    load_s = round(time.time() - t0, 1)
    fp16_a = float(score_tokens(model, seq, e - b, device).sum())
    fp16_b = float(score_tokens(model, seq, e - b, device).sum())
    dtype = choose_dtype(fp16_a, fp32_bits) if device != "cpu" else "float32"
    if dtype == "float32" and device != "cpu":
        del model
        release_device_memory()
        model = load_model(mid, torch.float32, device)
    check = {"model": mid, "family": stage.family, "stage": stage.stage, "gpu": device_name(device),
             "torch": torch.__version__, "transformers": transformers.__version__,
             "doc_start": start_str, "doc_start_id": start_id, "adds_bos": adds_bos(tok, cfg),
             "vocab": len(tok), "window0_tokens": e - b, "fp32_bits": fp32_bits, "fp32_error": fp32_err,
             "fp16_bits": round(fp16_a, 4), "deterministic": fp16_a == fp16_b,
             "fp16_vs_fp32_rel": None if fp32_bits is None else abs(fp16_a - fp32_bits) / fp32_bits,
             "dtype": dtype, "fp32_load_and_window_s": fp32_s, "load_s": load_s, "fp32_fallback_corpora": []}
    print(json.dumps({k: check[k] for k in ("model", "gpu", "doc_start", "dtype", "fp16_vs_fp32_rel", "load_s")},
                     ensure_ascii=False), flush=True)
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    for n in todo:
        t1 = time.time()
        bits = score_raw(model, wins[n], ids[n], start_id, device)
        if not np.isfinite(bits).all() and dtype == "float16":  # this and later corpora in fp32
            del model
            release_device_memory()
            model, dtype = load_model(mid, torch.float32, device), "float32"
            check["fp32_fallback_corpora"].append(n)
            bits = score_raw(model, wins[n], ids[n], start_id, device)
        wall_s = round(time.time() - t1, 1)
        if not np.isfinite(bits).all():
            raise FloatingPointError(f"non-finite bits for {mid} {n} in {dtype}")
        np.save(token_bits_path(out, mid, CONDITION, n), bits.astype(np.float32))
        append_rows(out / "windows.csv", window_rows(mid, n, wins[n]), WINDOW_COLS)
        meta = {**stage_meta(stage), "condition": CONDITION, "corpus": n, "n_windows": len(wins[n]),
                "window": WINDOW, "dtype": dtype, "doc_start": start_str, "doc_start_id": start_id,
                "ids_sha1": ids_digest(ids[n]), "gpu": device_name(device), "torch": torch.__version__}
        row = {**summarize(bits, corpora[n], meta), "wall_s": wall_s}
        append_rows(out / "bpb.csv", [row], CSV_COLS)
        print(json.dumps({k: row[k] for k in ("model", "corpus", "tokens", "bpb", "dtype", "wall_s")}), flush=True)
    del model
    release_device_memory()
    return {**check, "total_model_s": round(time.time() - t_model, 1)}


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    models = args.models or [s.model for s in STAGES]
    if len(models) > 1:  # one child process per model: weights are guaranteed freed between models
        failed = []
        for mid in models:
            cmd = [sys.executable, os.path.abspath(__file__), "run", "--models", mid, "--corpora", *args.corpora,
                   "--data", args.data, "--results", args.results, "--device", args.device]
            rc = subprocess.run(cmd, check=False).returncode
            failed += [mid] if rc else []
        print(json.dumps({"models": len(models), "failed": failed}), flush=True)
        return 1 if failed else 0
    stage = stage_of(models[0])
    device = pick_device() if args.device == "auto" else args.device
    corpora = load_corpora(Path(args.data), args.corpora, None)
    rc = 0
    try:
        check = run_model(stage, corpora, out, device)
    except Exception as e:  # record, and let the parent continue with the next model
        check, rc = {"model": stage.model, "error": f"{type(e).__name__}: {e}"[:1000]}, 1
    with (out / "model_checks.jsonl").open("a") as f:
        f.write(json.dumps(check, ensure_ascii=False) + "\n")
    print(json.dumps(check, ensure_ascii=False), flush=True)
    return rc


def cmd_tokcheck(args: argparse.Namespace) -> int:
    corpora = load_corpora(Path(args.data), args.corpora, None)
    report: dict[str, object] = {"corpora": {n: {"bytes": len(t.encode("utf-8")), "chars": len(t)}
                                             for n, t in corpora.items()}, "groups": {}}
    for group, models in tokenizer_groups().items():
        ref: dict[str, list[int]] = {}
        per_model = {}
        for mid in models:
            tok = AutoTokenizer.from_pretrained(mid)
            cfg = read_tokenizer_config(mid)
            ids = {n: encode(tok, t) for n, t in corpora.items()}
            ref = ref or ids
            start_id, start_str = doc_start_token(tok, cfg)
            per_model[mid] = {"vocab": len(tok), "doc_start": start_str, "doc_start_id": start_id,
                              "adds_bos": adds_bos(tok, cfg), "tokens": {n: len(v) for n, v in ids.items()},
                              "ids_sha1": {n: ids_digest(v) for n, v in ids.items()},
                              "roundtrip": {n: decode(tok, ids[n]) == t for n, t in corpora.items()},
                              "identical_to_first": compare_ids(ref, ids)}
        report["groups"][group] = {"models": per_model, "all_identical": all(
            all(m["identical_to_first"].values()) and all(m["roundtrip"].values()) for m in per_model.values())}
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokcheck.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for group, g in report["groups"].items():
        starts = sorted({f"{m['doc_start']} ({m['doc_start_id']})" for m in g["models"].values()})
        print(json.dumps({"group": group, "models": len(g["models"]), "all_identical": g["all_identical"],
                          "doc_starts": starts}, ensure_ascii=False), flush=True)
    return 0 if all(g["all_identical"] for g in report["groups"].values()) else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "tokcheck"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--corpora", nargs="*", default=list(CORPORA))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default="exp01_posttraining_bpc/data")
    ap.add_argument("--results", default="exp03_staged_drift/results")
    args = ap.parse_args()
    torch.manual_seed(0)
    transformers.utils.logging.set_verbosity_error()  # one log line per (model, corpus), no bars
    transformers.utils.logging.disable_progress_bar()
    raise SystemExit({"run": cmd_run, "tokcheck": cmd_tokcheck}[args.cmd](args))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

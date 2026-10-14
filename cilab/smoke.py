"""Smoke scorer: GPT-2 bits per byte on one corpus, to prove tokenizer, windows, scoring and device on a new machine.

    python -m cilab.smoke --data exp01_posttraining_bpc/data --corpus enwik8_ctrl --device cuda --out /run/dir/smoke.json

Expected on enwik8_ctrl: 1.243 bpb (exp01 sanity on the Mac, fp16; fp32 agrees to 2e-4). Exit code 2 on mismatch.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cilab.io import load_corpora, read_tokenizer_config
from cilab.score import free_model, load_model, pick_device, score_raw, summarize
from cilab.text import doc_start_token, encode, make_windows

MODEL_ID = "openai-community/gpt2"
WINDOW = 1023  # GPT-2 has 1024 positions; one is the doc-start token
REFERENCE_BPB = {"enwik8_ctrl": 1.243}
TOLERANCE_BPB = 0.01


def verdict(bpb: float, corpus: str, reference: dict[str, float] = REFERENCE_BPB,
            tol: float = TOLERANCE_BPB) -> str:
    """'ok' when within tol of the Mac reference, 'mismatch' otherwise, 'unreferenced' for other corpora."""
    if corpus not in reference:
        return "unreferenced"
    return "ok" if abs(bpb - reference[corpus]) <= tol else "mismatch"


def dtype_for(device: str) -> torch.dtype:
    return torch.float32 if device == "cpu" else torch.float16


def score_corpus(model: object, tok: object, text: str, start_id: int, device: str) -> dict[str, object]:
    ids = encode(tok, text)
    wins = make_windows(tok, ids, text, WINDOW)
    t0 = time.time()
    bits = score_raw(model, wins, ids, start_id, device)
    return {**summarize(bits, text, {"model": MODEL_ID, "window": WINDOW}),
            "wall_s": round(time.time() - t0, 1), "n_windows": len(wins)}


def cmd_smoke(args: argparse.Namespace) -> int:
    device = pick_device() if args.device == "auto" else args.device
    text = load_corpora(Path(args.data), [args.corpus], args.max_bytes)[args.corpus]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    start_id, start_str = doc_start_token(tok, read_tokenizer_config(MODEL_ID))
    model = load_model(MODEL_ID, dtype_for(device), device)
    res = score_corpus(model, tok, text, start_id, device)
    free_model(model)
    out = {"device": device, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
           "doc_start": start_str, "corpus": args.corpus, **res,
           "reference_bpb": REFERENCE_BPB.get(args.corpus), "verdict": verdict(float(res["bpb"]), args.corpus)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0 if out["verdict"] != "mismatch" else 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--corpus", default="enwik8_ctrl")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-bytes", dest="max_bytes", type=int, default=None)
    ap.add_argument("--out", required=True)
    raise SystemExit(cmd_smoke(ap.parse_args()))


if __name__ == "__main__":
    main()

"""Score bits-per-byte (bpb) of causal LMs on text corpora, RAW and CHAT conditions.

Subcommands
  python score_bpb.py tokcheck            tokenizer identity per pair + round-trip -> results/tokcheck.json
  python score_bpb.py sanity              GPT-2 on enwik8 control + checks (b,c,d) -> results/sanity.json
  python score_bpb.py run [--models ...]  main runs, appends to results/bpb.csv (resumable)

Mechanics (identical for every model, see RESULTS.md):
  * corpus -> token ids with add_special_tokens=False (round-trip decode must equal the corpus).
  * ids split into consecutive non-overlapping windows of <= W text tokens (W=1024; 1023 for GPT-2,
    whose context is 1024 positions). A boundary is moved left by a token or two if it would split a
    multi-byte UTF-8 character, so that the concatenated window texts reproduce the corpus exactly.
  * RAW: input = [doc_start] + window_ids. doc_start = the tokenizer's BOS if its config adds BOS
    (only DeepSeek-R1-Distill), else the pre-training document separator <|endoftext|> (GPT-2,
    Qwen, SmolLM2). EVERY text token is scored (the first one conditioned on doc_start only), so
    sum(bits) is a complete code length for the corpus: bpb = sum(bits) / utf8_bytes(corpus).
  * CHAT: input = render(chat_template([user:"Continue the text.", assistant:@@P@@])) cut at @@P@@
    (Qwen3: enable_thinking=False) + tokens(window_text). Only the window_text tokens are scored.
    bytes are identical to RAW because the window texts concatenate to the corpus.
  * -log2 p computed from float32 log-softmax of the logits.
No globals besides immutable constants; model/tokenizer/device are passed explicitly.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PAIRS = (
    ("qwen2.5", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen3", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B"),
    ("smollm2", "HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("qwen2.5-math", "Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
)
CORPORA = ("wiki2026", "arxiv2026", "gutenberg2026", "enwik8_ctrl")
CSV_COLS = ("model", "pair", "role", "condition", "corpus", "bytes", "chars", "tokens", "total_bits",
            "bpb", "bpc", "wall_s", "tokens_per_byte", "bits_per_token", "n_windows", "window", "dtype",
            "doc_start")
CHAT_USER = "Continue the text."
PLACEHOLDER = "@@PLACEHOLDER@@"
LN2 = math.log(2.0)


# ----------------------------------------------------------------------------- pure helpers
def model_roles(pairs=PAIRS) -> list[dict]:
    out = []
    for pair, base, post in pairs:
        out += [{"model": base, "pair": pair, "role": "base"}, {"model": post, "pair": pair, "role": "post"}]
    return out


def load_corpora(data_dir: Path, names, max_bytes: int | None) -> dict[str, str]:
    out = {}
    for n in names:
        t = (data_dir / f"{n}.txt").read_text(encoding="utf-8")
        if max_bytes:
            t = t.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        out[n] = t
    return out


def adds_bos(model_dir: str, tok) -> bool:
    """True if the tokenizer config asks for a BOS in front of every sequence."""
    cfg_path = Path(model_dir) / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return bool(cfg.get("add_bos_token")) and tok.bos_token_id is not None


def doc_start_token(tok, model_dir: str) -> tuple[int, str]:
    if adds_bos(model_dir, tok):
        return tok.bos_token_id, tok.bos_token
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    # compare by string, not against unk_token_id: SmolLM2-Instruct's unk token IS <|endoftext|> (id 0)
    if eot is not None and tok.convert_ids_to_tokens(eot) == "<|endoftext|>":
        return eot, "<|endoftext|>"
    if tok.eos_token_id is not None:
        return tok.eos_token_id, tok.eos_token
    raise ValueError("no document-start token")


def encode(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def decode(tok, ids) -> str:
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def make_windows(tok, ids: list[int], text: str, size: int) -> list[tuple[int, int, str]]:
    """Consecutive (start, end, text) windows of <= size tokens whose texts concatenate to `text`."""
    wins, b, pos = [], 0, 0
    while b < len(ids):
        e = min(b + size, len(ids))
        s = decode(tok, ids[b:e])
        while e < len(ids) and not (text.startswith(s, pos) and not s.endswith("�")):
            e -= 1
            s = decode(tok, ids[b:e])
        if e <= b or not text.startswith(s, pos):
            raise ValueError(f"window {len(wins)} does not decode to a prefix of the remaining corpus")
        wins.append((b, e, s))
        pos += len(s)
        b = e
    if pos != len(text):
        raise ValueError(f"windows cover {pos} of {len(text)} chars")
    return wins


def chat_prefix(tok, model_id: str) -> tuple[list[int], str] | None:
    if tok.chat_template is None:
        return None
    kw = {"enable_thinking": False} if "Qwen3" in model_id else {}
    msgs = [{"role": "user", "content": CHAT_USER}, {"role": "assistant", "content": PLACEHOLDER}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, **kw)
    if rendered.count(PLACEHOLDER) != 1:
        raise ValueError("placeholder not rendered exactly once")
    pre = rendered.split(PLACEHOLDER)[0]
    return encode(tok, pre), pre  # special-token strings in `pre` are parsed as special tokens


def summarize(bits: np.ndarray, text: str, meta: dict) -> dict:
    nbytes, nchars, ntok = len(text.encode("utf-8")), len(text), int(bits.size)
    total = float(bits.sum(dtype=np.float64))
    return {**meta, "bytes": nbytes, "chars": nchars, "tokens": ntok, "total_bits": round(total, 3),
            "bpb": round(total / nbytes, 5), "bpc": round(total / nchars, 5),
            "tokens_per_byte": round(ntok / nbytes, 5), "bits_per_token": round(total / ntok, 5)}


# ----------------------------------------------------------------------------- model I/O
def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(model_id: str, dtype: torch.dtype, device: str):
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    return m.to(device).eval()


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def model_dir(model_id: str) -> str:
    from huggingface_hub import hf_hub_download
    return str(Path(hf_hub_download(model_id, "tokenizer_config.json", local_files_only=True)).parent)


@torch.no_grad()
def score_tokens(model, input_ids: list[int], n_score: int, device: str) -> np.ndarray:
    """-log2 p of the last n_score tokens of input_ids, each conditioned on everything before it."""
    x = torch.tensor([input_ids], device=device)
    logits = model(input_ids=x, use_cache=False).logits[0]
    lo = len(input_ids) - n_score
    pred = logits[lo - 1:-1].float()
    nats = F.cross_entropy(pred, x[0, lo:], reduction="none")
    return (nats / LN2).cpu().numpy().astype(np.float64)


def score_raw(model, wins, ids, start_id: int, device: str) -> np.ndarray:
    return np.concatenate([score_tokens(model, [start_id] + ids[b:e], e - b, device) for b, e, _ in wins])


def score_chat(model, tok, wins, prefix_ids: list[int], device: str) -> np.ndarray:
    out = []
    for _, _, s in wins:
        t = encode(tok, s)
        out.append(score_tokens(model, prefix_ids + t, len(t), device))
    return np.concatenate(out)


@torch.no_grad()
def hf_loss_bits(model, input_ids: list[int], device: str) -> float:
    """Sum of bits implied by HF's own labels= loss (mean nats over len-1 shifted targets)."""
    x = torch.tensor([input_ids], device=device)
    loss = model(input_ids=x, labels=x, use_cache=False).loss.float().item()
    return loss * (len(input_ids) - 1) / LN2


# ----------------------------------------------------------------------------- checks
def fp32_reference_bits(model_id: str, seq: list[int], n_score: int, device: str, data_args: list[str]) -> float:
    """fp32 bits for one window, computed in a SEPARATE process so the fp32 weights (up to 8 GB for
    Qwen3-1.7B) are guaranteed to be released before the fp16 copy is loaded (an in-process
    model.half() kept the fp32 copy alive on MPS -> 12 GB RSS and heavy swapping)."""
    cmd = [sys.executable, os.path.abspath(__file__), "fp32ref", "--models", model_id, *data_args]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    rec = json.loads(out)
    if rec["n_score"] != n_score or rec["seq_len"] != len(seq):
        raise ValueError(f"fp32 reference window mismatch: {rec} vs n_score={n_score} len={len(seq)}")
    return rec["fp32_bits"]


def window_checks(model_id: str, tok, ids, wins, start_id: int, device: str, data_args: list[str]) -> dict:
    """(b) labels-loss equality, (c) determinism, (d) fp16 vs fp32 on window 0. Returns the fp16 model."""
    b, e, _ = wins[0]
    seq = [start_id] + ids[b:e]
    fp32 = fp32_reference_bits(model_id, seq, e - b, device, data_args)
    model = load_model(model_id, torch.float16, device)
    fp16_a = score_tokens(model, seq, e - b, device).sum()
    fp16_b = score_tokens(model, seq, e - b, device).sum()
    hf = hf_loss_bits(model, seq, device)
    rel = abs(fp16_a - fp32) / fp32
    rec = {"model": model_id, "window_tokens": e - b, "fp32_bits": round(float(fp32), 4),
           "fp16_bits": round(float(fp16_a), 4), "fp16_vs_fp32_rel": float(rel),
           "fp16_ok": bool(np.isfinite(fp16_a) and rel < 0.005),
           "repeat_bits": round(float(fp16_b), 4), "deterministic": bool(fp16_a == fp16_b),
           "hf_labels_loss_bits": round(float(hf), 4), "labels_rel_diff": float(abs(hf - fp16_a) / fp16_a)}
    if not rec["fp16_ok"]:  # fall back to fp32 for this model
        free_model(model)
        model = load_model(model_id, torch.float32, device)
    return {"check": rec, "model": model}


def cmd_docstart(args) -> None:
    """Ablation: RAW scoring with a plain-text doc-start ("\\n\\n" tokens) instead of a special token.
    One process per model; rows -> results/ablation_docstart.csv, bits -> results/ablation_tokens/."""
    specs = [s for s in model_roles() if not args.models or s["model"] in args.models]
    if len(specs) > 1:
        for spec in specs:
            subprocess.run([sys.executable, os.path.abspath(__file__), "docstart", "--models", spec["model"],
                            "--results", args.results, *data_args_of(args)], check=False)
        return
    spec = specs[0]
    mid = spec["model"]
    out = Path(args.results)
    (out / "ablation_tokens").mkdir(exist_ok=True)
    corpora = load_corpora(Path(args.data), args.corpora, args.max_bytes)
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(mid)
    prefix = encode(tok, "\n\n")
    model = load_model(mid, torch.float16, device)
    rows = []
    for n, text in corpora.items():
        ids = encode(tok, text)
        wins = make_windows(tok, ids, text, args.window)
        t0 = time.time()
        bits = np.concatenate([score_tokens(model, prefix + ids[b:e], e - b, device) for b, e, _ in wins])
        np.save(out / "ablation_tokens" / f"{mid.replace('/', '__')}__raw_nl__{n}.npy", bits.astype(np.float32))
        meta = {"model": mid, "pair": spec["pair"], "role": spec["role"], "condition": "raw_nl", "corpus": n,
                "n_windows": len(wins), "window": args.window, "dtype": "float16", "doc_start": f"\\n\\n {prefix}"}
        rows.append({**summarize(bits, text, meta), "wall_s": round(time.time() - t0, 1)})
        print(json.dumps({k: rows[-1][k] for k in ("model", "corpus", "bpb")}), flush=True)
    free_model(model)
    append_rows(out / "ablation_docstart.csv", rows)


def cmd_audit(args) -> None:
    """Recompute selected windows in float32 (or --dtype) and compare with the saved fp16 per-token bits.
    Appends one JSON line per window to results/fp32_audit.jsonl."""
    mid, corpus, cond = args.models[0], args.corpora[0], args.condition
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    text = load_corpora(Path(args.data), [corpus], args.max_bytes)[corpus]
    tok = AutoTokenizer.from_pretrained(mid)
    start_id, _ = doc_start_token(tok, model_dir(mid))
    ids = encode(tok, text)
    wins = make_windows(tok, ids, text, args.window)
    cp = chat_prefix(tok, mid) if cond == "chat" else None
    lens = [len(encode(tok, w[2])) if cond == "chat" else w[1] - w[0] for w in wins]
    edges = np.cumsum([0] + lens)
    saved = np.load(Path(args.results) / "tokens" / f"{mid.replace('/', '__')}__{cond}__{corpus}.npy").astype(np.float64)
    device = pick_device() if args.device == "auto" else args.device
    model = load_model(mid, dtype, device)
    targets = args.windows if args.windows else list(range(len(wins)))
    with (Path(args.results) / "fp32_audit.jsonl").open("a") as f:
        for k in targets:
            b, e, w = wins[k]
            bits = (score_tokens(model, [start_id] + ids[b:e], e - b, device) if cond == "raw"
                    else score_tokens(model, cp[0] + encode(tok, w), lens[k], device))
            old = saved[edges[k]:edges[k + 1]]
            rec = {"model": mid, "condition": cond, "corpus": corpus, "window": k, "dtype": args.dtype,
                   "device": device, "bits_audit": round(float(bits.sum()), 3), "bits_fp16_saved": round(float(old.sum()), 3),
                   "rel_diff": float((bits.sum() - old.sum()) / bits.sum()),
                   "tokens_gt35_audit": int((bits > 35).sum()), "tokens_gt35_fp16": int((old > 35).sum()),
                   "max_abs_token_diff": round(float(np.abs(bits - old).max()), 3)}
            f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)


def cmd_fp32ref(args) -> None:
    """Child process: fp32 bits of window 0 of the first corpus for one model; prints one JSON line."""
    mid = args.models[0]
    text = load_corpora(Path(args.data), args.corpora[:1], args.max_bytes)[args.corpora[0]]
    tok = AutoTokenizer.from_pretrained(mid)
    start_id, _ = doc_start_token(tok, model_dir(mid))
    ids = encode(tok, text)
    b, e, _ = make_windows(tok, ids, text, args.window)[0]
    seq = [start_id] + ids[b:e]
    device = pick_device()
    model = load_model(mid, torch.float32, device)
    bits = float(score_tokens(model, seq, e - b, device).sum())
    print(json.dumps({"model": mid, "fp32_bits": bits, "n_score": e - b, "seq_len": len(seq)}))


# ----------------------------------------------------------------------------- runners
def existing_keys(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        return {(r["model"], r["condition"], r["corpus"]) for r in csv.DictReader(f)}


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})


def run_one_model(spec: dict, corpora: dict[str, str], window: int, out_dir: Path, device: str,
                  done: set, data_args: list[str]) -> tuple[list[dict], dict]:
    mid = spec["model"]
    mdir = model_dir(mid)
    tok = AutoTokenizer.from_pretrained(mid)
    start_id, start_str = doc_start_token(tok, mdir)
    toks = {n: encode(tok, t) for n, t in corpora.items()}
    wins = {n: make_windows(tok, toks[n], corpora[n], window) for n in corpora}
    first = next(iter(corpora))
    t0 = time.time()
    loaded = window_checks(mid, tok, toks[first], wins[first], start_id, device, data_args)
    model, check = loaded["model"], loaded["check"]
    check["load_and_check_s"] = round(time.time() - t0, 1)
    dtype = "float16" if check["fp16_ok"] else "float32"
    cp = chat_prefix(tok, mid)
    check["chat_prefix"] = None if cp is None else cp[1]
    check["chat_prefix_tokens"] = None if cp is None else len(cp[0])
    check["doc_start"] = f"{start_str} ({start_id})"
    rows = []
    conds = ["raw"] + (["chat"] if cp is not None else [])
    for cond in conds:
        for n, text in corpora.items():
            if (mid, cond, n) in done:
                continue
            t1 = time.time()
            bits = (score_raw(model, wins[n], toks[n], start_id, device) if cond == "raw"
                    else score_chat(model, tok, wins[n], cp[0], device))
            meta = {"model": mid, "pair": spec["pair"], "role": spec["role"], "condition": cond, "corpus": n,
                    "n_windows": len(wins[n]), "window": window, "dtype": dtype,
                    "doc_start": start_str if cond == "raw" else "chat_template"}
            row = {**summarize(bits, text, meta), "wall_s": round(time.time() - t1, 1)}
            if not np.isfinite(row["total_bits"]):
                raise FloatingPointError(f"non-finite bits for {mid} {cond} {n}")
            tdir = out_dir / "tokens"
            tdir.mkdir(exist_ok=True)
            np.save(tdir / f"{mid.replace('/', '__')}__{cond}__{n}.npy", bits.astype(np.float32))
            append_rows(out_dir / "bpb.csv", [row])
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("model", "condition", "corpus", "tokens", "bpb", "wall_s")}),
                  flush=True)
    free_model(model)
    return rows, check


def data_args_of(args) -> list[str]:
    a = ["--corpora", *args.corpora, "--data", args.data, "--window", str(args.window)]
    return a + (["--max-bytes", str(args.max_bytes)] if args.max_bytes else [])


def cmd_run(args) -> None:
    out = Path(args.results)
    out.mkdir(exist_ok=True)
    specs = [s for s in model_roles() if not args.models or s["model"] in args.models]
    if len(specs) > 1:  # one process per model: weights are guaranteed freed between models
        for spec in specs:
            cmd = [sys.executable, os.path.abspath(__file__), "run", "--models", spec["model"],
                   "--results", args.results, *data_args_of(args)]
            subprocess.run(cmd, check=False)
        return
    corpora = load_corpora(Path(args.data), args.corpora, args.max_bytes)
    device = pick_device()
    for spec in specs:
        done = existing_keys(out / "bpb.csv")
        conds = ["raw"] + (["chat"] if AutoTokenizer.from_pretrained(spec["model"]).chat_template else [])
        if all((spec["model"], c, n) in done for c in conds for n in corpora):
            continue
        t0 = time.time()
        try:
            _, check = run_one_model(spec, corpora, args.window, out, device, done, data_args_of(args))
            check["total_model_s"] = round(time.time() - t0, 1)
        except Exception as e:  # record and continue with the next model
            check = {"model": spec["model"], "error": f"{type(e).__name__}: {e}"[:1000]}
        with (out / "model_checks.jsonl").open("a") as f:
            f.write(json.dumps(check, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in check.items() if k != "chat_prefix"}), flush=True)


def cmd_tokcheck(args) -> None:
    corpora = load_corpora(Path(args.data), args.corpora, args.max_bytes)
    res = {"corpora": {}, "pairs": {}, "special_strings_in_corpora": {}}
    for n, t in corpora.items():
        res["corpora"][n] = {"bytes": len(t.encode("utf-8")), "chars": len(t)}
        res["special_strings_in_corpora"][n] = [s for s in ("<|endoftext|>", "<|im_start|>", "<|im_end|>",
                                                            "<think>", "<｜") if s in t]
    for pair, base, post in PAIRS:
        tb, tp = AutoTokenizer.from_pretrained(base), AutoTokenizer.from_pretrained(post)
        per = {}
        for n, t in corpora.items():
            ib, ip = encode(tb, t), encode(tp, t)
            per[n] = {"identical_ids": ib == ip, "tokens_base": len(ib), "tokens_post": len(ip),
                      "roundtrip_base": decode(tb, ib) == t, "roundtrip_post": decode(tp, ip) == t}
        res["pairs"][pair] = {"base": base, "post": post, "vocab_base": len(tb), "vocab_post": len(tp),
                              "corpora": per,
                              "all_identical": all(v["identical_ids"] for v in per.values())}
    Path(args.results).mkdir(exist_ok=True)
    (Path(args.results) / "tokcheck.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


def cmd_sanity(args) -> None:
    """(a) GPT-2 bpb on enwik8 control; (b,c,d) on GPT-2 window 0; prefix ablation for R1-distill."""
    device = pick_device()
    corpora = load_corpora(Path(args.data), ["enwik8_ctrl", "arxiv2026"], args.max_bytes)
    mid = "openai-community/gpt2"
    tok = AutoTokenizer.from_pretrained(mid)
    start_id, start_str = doc_start_token(tok, model_dir(mid))
    window = 1023  # GPT-2 has 1024 positions; one is the doc-start token
    res = {"device": device, "torch": torch.__version__}
    ids = encode(tok, corpora["enwik8_ctrl"])
    wins = make_windows(tok, ids, corpora["enwik8_ctrl"], window)
    loaded = window_checks(mid, tok, ids, wins, start_id, device,
                           ["--corpora", "enwik8_ctrl", "--data", args.data, "--window", str(window)])
    model = loaded["model"]
    res["gpt2_window_checks"] = loaded["check"]
    out = {}
    for n, text in corpora.items():
        ids_n = encode(tok, text)
        w_n = make_windows(tok, ids_n, text, window)
        t0 = time.time()
        bits = score_raw(model, w_n, ids_n, start_id, device)
        out[n] = {**summarize(bits, text, {"model": mid, "window": window, "doc_start": start_str}),
                  "wall_s": round(time.time() - t0, 1),
                  "bpb_first_token_of_windows": round(float(bits[[b for b, _, _ in w_n]].sum() / len(text.encode())), 5)}
    res["gpt2_raw"] = out
    free_model(model)
    # prefix ablation: R1-distill's natural BOS vs the <|endoftext|>-equivalent id used for its base
    mid = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tok = AutoTokenizer.from_pretrained(mid)
    text = corpora["arxiv2026"]
    ids = encode(tok, text)
    wins = make_windows(tok, ids, text, 1024)
    model = load_model(mid, torch.float16, device)
    abl = {}
    for name, sid in (("bos_151646", tok.bos_token_id), ("eot_151643", 151643)):
        bits = score_raw(model, wins, ids, sid, device)
        abl[name] = summarize(bits, text, {"model": mid, "doc_start_id": sid})["bpb"]
    res["r1_prefix_ablation_arxiv2026_bpb"] = abl
    free_model(model)
    Path(args.results).mkdir(exist_ok=True)
    (Path(args.results) / "sanity.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "sanity", "tokcheck", "fp32ref", "audit", "docstart"])
    ap.add_argument("--condition", default="raw", choices=["raw", "chat"])
    ap.add_argument("--windows", nargs="*", type=int, default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--corpora", nargs="*", default=list(CORPORA))
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--max-bytes", type=int, default=None, help="cut every corpus to this many bytes")
    args = ap.parse_args()
    torch.manual_seed(0)
    {"run": cmd_run, "sanity": cmd_sanity, "tokcheck": cmd_tokcheck, "fp32ref": cmd_fp32ref, "audit": cmd_audit, "docstart": cmd_docstart}[args.cmd](args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

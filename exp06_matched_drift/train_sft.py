"""exp06 training, run INSIDE the AAU GPU jobs (never on the front end): full SFT of OLMo-2-0425-1B on one recipe's
packed, token-matched subset, with the drift hook and checkpoints at drift marks.

    python exp06_matched_drift/train_sft.py train --recipe R --data PREP_DIR --out RUN_DIR [--max-steps 1500]
    python exp06_matched_drift/train_sft.py selftest [--device cuda]

`train` (definitions in RESULTS.md; pure helpers in cilab/sft.py and cilab/drift.py):
  * data: PREP_DIR/<recipe>.npz from prep_data.py; rows of 2,048 tokens packed greedily in a per-epoch seeded order
    (cilab.sft.row_stream), 32 rows per update, micro-batches of --micro-batch rows;
  * model: fp32 master weights, bf16 autocast, SDPA; position ids restart per example, so transformers builds a
    block-diagonal causal mask (checked by `selftest`); loss = mean cross-entropy over the update's loss targets,
    computed on the hidden states of target positions only;
  * AdamW (0.9, 0.999, 1e-8, wd 0), clip 1.0, lr 2e-5 with 45 warmup updates and cosine decay to 0 at update 1,500
    (the schedule is fixed; --max-steps only stops early, so a dry run is the first steps of the real run);
  * drift hook, at update 0 and every --score-every updates and at the stop: the fp32 weights are copied into a bf16
    model (the weights a checkpoint saves), which scores exp01's 4 corpora RAW with exp03's mechanics; per-token bits
    -> RUN_DIR/<recipe>/tokens/step<NNNNN>__<corpus>.npy, one row -> drift.csv (Δbpb per corpus, Δbpb_fresh, class
    drifts and contrasts against update 0, the base in bf16 on the same GPU);
  * checkpoints: bf16 safetensors + the Instruct tokenizer at the first scored update at or above each mark
    (3/6/10% Δbpb_fresh) and at the stop -> RUN_DIR/<recipe>/ckpt/step<NNNNN>/, one line each appended (with a file
    lock) to RUN_DIR/checkpoints.txt and RUN_DIR/<recipe>/checkpoints.txt;
  * stop at --max-steps updates or --max-wall-s seconds of training wall time.
A Slurm requeue restarts from scratch into RUN_DIR/<recipe>-attempt<k>/ (nothing is overwritten).

`selftest`: a tiny random OLMo-2 checks that a packed row gives the same logits as its examples run alone, and that
the target-only loss equals transformers' labels= loss.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cilab`

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cilab.drift import drift_row
from cilab.io import append_rows, load_corpora, read_tokenizer_config
from cilab.pairs import CORPORA, FRESH_CORPORA
from cilab.score import release_device_memory
from cilab.sft import (
    checkpoint_label,
    checkpoint_line,
    final_checkpoint,
    lr_at,
    marks_due,
    parse_marks,
    remaining_marks,
    row_arrays,
    row_stream,
    saved_marks,
    stop_reason,
    warmup_steps,
)
from cilab.text import doc_start_token, encode, make_windows

ROOT = Path(__file__).resolve().parents[1]
BASE = "allenai/OLMo-2-0425-1B"
TEMPLATE_TOKENIZER = "allenai/OLMo-2-0425-1B-Instruct"
RECIPES: tuple[str, ...] = ("tulu", "smoltalk", "mathcot")  # Slurm array index -> recipe
ROW_LEN, ROWS_PER_STEP = 2048, 32  # tokens per packed row, rows per update
PEAK_LR, WARMUP_FRAC, TOTAL_STEPS = 2e-5, 0.03, 1500
BETAS, EPS, WEIGHT_DECAY, CLIP = (0.9, 0.999), 1e-8, 0.0, 1.0
MAX_WALL_S = 5 * 3600  # training wall time, s
SCORE_EVERY = 20  # updates
MARKS: tuple[float, ...] = (3.0, 6.0, 10.0)  # Δbpb_fresh, %
WINDOW = 1024  # scoring window, tokens (exp03)
SEED = 0
LOG_EVERY = 10  # updates
LABEL_PREFIX = "exp06"
DATA_DIR = ROOT / "exp01_posttraining_bpc" / "data"
CLASSES_DIR = ROOT / "exp05_decomposition" / "results" / "classes"
EXP03_BPB = ROOT / "exp03_staged_drift" / "results" / "bpb.csv"
BASE_BPB_TOL = 5e-3  # relative, bf16 here vs exp03's fp16 T4
DRIFT_COLS = ("recipe", "step", "token_positions", "nonpad_tokens", "loss_tokens", "epochs", "lr", "train_loss",
              *(f"bpb_{c}" for c in CORPORA), *(f"dbpb_{c}" for c in sorted(CORPORA)), "dbpb_fresh",
              "drift_knowledge", "drift_middle", "drift_style", "drift_rare", "drift_frequent", "contrast_ks",
              "contrast_rf", "score_s", "wall_s", "checkpoint")
SELFTEST_TOL = 5e-3  # logits (absolute) and losses (relative); TF32 GEMMs in the NGC image reach ~1e-3; a leak is ~0.5
TRAIN_COLS = ("recipe", "step", "lr", "loss", "grad_norm", "epochs", "token_positions", "loss_tokens", "wall_s",
              "tok_per_s", "peak_mem_gb")


# ----------------------------------------------------------------------------- pure
def first_free(candidates: Sequence[Path], taken: Sequence[bool]) -> Path:
    """The first candidate directory not already used by an earlier attempt."""
    for c, t in zip(candidates, taken, strict=True):
        if not t:
            return c
    raise RuntimeError("no free run directory")


def example_slices(offsets: np.ndarray) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in pairwise(offsets)]


def bpb_of(bits: np.ndarray, text: str) -> float:
    return float(bits.sum()) / len(text.encode("utf-8"))


def base_check(bpb_now: Mapping[str, float], bpb_ref: Mapping[str, float], tol: float) -> dict[str, object]:
    """Relative difference of the step-0 bpb from exp03's base bpb per corpus, and whether all are within tol."""
    rel = {c: (bpb_now[c] - bpb_ref[c]) / bpb_ref[c] for c in sorted(bpb_ref)}
    return {"rel_diff": rel, "max_abs_rel": max(abs(v) for v in rel.values()), "ok": all(abs(v) < tol for v in rel.values())}


# ----------------------------------------------------------------------------- effects: data and scoring
def load_examples(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """(ids, loss mask) per example from prep_data.py's npz."""
    with np.load(path) as z:
        ids, mask, offsets = z["ids"].astype(np.int64), z["mask"].astype(bool), z["offsets"]
    return [(ids[a:b], mask[a:b]) for a, b in example_slices(offsets)]


def batches(examples: Sequence[tuple[np.ndarray, np.ndarray]], pad_id: int,
            skip_updates: int = 0) -> Iterator[dict[str, object]]:
    """Endless updates: 32 packed rows each, as numpy arrays, with the epoch of the last row. The first
    `skip_updates` updates of the stream (an earlier stage's data) are consumed without building rows; the first item
    yielded is then their totals (keys skipped_*), so a continued run resumes the same data order."""
    stream = row_stream([len(i) for i, _ in examples], ROW_LEN, SEED)
    n_ex = len(examples)
    seen = 0
    if skip_updates:
        nonpad = loss = 0
        for _ in range(skip_updates * ROWS_PER_STEP):
            _, r = next(stream)
            seen += len(r)
            nonpad += sum(len(examples[i][0]) for i in r)
            loss += sum(int(examples[i][1].sum()) for i in r)
        yield {"skipped_updates": skip_updates, "skipped_nonpad": nonpad, "skipped_loss": loss, "epochs": seen / n_ex}
    while True:
        rows, epoch, nonpad = [], 0, 0
        for _ in range(ROWS_PER_STEP):
            epoch, r = next(stream)
            rows.append(row_arrays([examples[i] for i in r], ROW_LEN, pad_id))
            seen += len(r)
            nonpad += sum(len(examples[i][0]) for i in r)
        ids, pos, tgt = (np.stack(a) for a in zip(*rows, strict=True))
        yield {"ids": ids, "pos": pos, "tgt": tgt, "epochs": seen / n_ex, "epoch": epoch, "nonpad": nonpad}


def corpus_setup(tok: object, base_id: str) -> dict[str, object]:
    """Corpora, ids, windows, doc-start id and exp05's classes (verified against these ids and windows)."""
    import hashlib
    texts = load_corpora(DATA_DIR, CORPORA, None)
    start_id, start_str = doc_start_token(tok, read_tokenizer_config(base_id))
    ids = {c: encode(tok, t) for c, t in texts.items()}
    wins = {c: make_windows(tok, ids[c], texts[c], WINDOW) for c in CORPORA}
    classes = {}
    for c in CORPORA:
        with np.load(CLASSES_DIR / f"olmo2__{c}.npz") as z:
            sha = hashlib.sha1(np.asarray(ids[c], dtype=np.int64).tobytes()).hexdigest()[:16]
            lengths = tuple(e - b for b, e, _ in wins[c])
            if str(z["ids_sha1"]) != sha or tuple(int(x) for x in z["lengths"]) != lengths:
                raise ValueError(f"{c}: exp05 classes do not match these ids/windows ({z['ids_sha1']} vs {sha})")
            classes[c] = {"pos": z["pos"], "freq": z["freq"]}
    return {"texts": texts, "ids": ids, "wins": wins, "start_id": start_id, "start_str": start_str,
            "classes": classes}


def resolve_init(path: Path, recipe: str) -> tuple[str, int, list[str]]:
    """(checkpoint dir, its update, the earlier stage's checkpoint lines) from a checkpoint dir or a run dir."""
    if (path / "config.json").exists():
        meta = json.loads((path / "exp06_checkpoint.json").read_text())
        lines_path = path.parents[1] / "checkpoints.txt"  # <run>/<recipe>/ckpt/stepNNNNN -> <run>/<recipe>/
        lines = lines_path.read_text().splitlines(keepends=True) if lines_path.exists() else []
        return str(path), int(meta["step"]), lines
    lines = (path / "checkpoints.txt").read_text().splitlines(keepends=True)
    ckpt, step = final_checkpoint(lines, recipe)
    return ckpt, step, lines


def score_corpora(model: object, corp: Mapping[str, object], device: str) -> dict[str, np.ndarray]:
    from cilab.score import score_raw
    return {c: score_raw(model, corp["wins"][c], corp["ids"][c], corp["start_id"], device) for c in CORPORA}


@torch.no_grad()
def sync_shadow(shadow: torch.nn.Module, model: torch.nn.Module) -> None:
    """Copy the fp32 training weights into the bf16 scoring/saving model (round to nearest)."""
    for (n1, p1), (n2, p2) in zip(shadow.named_parameters(), model.named_parameters(), strict=True):
        if n1 != n2:
            raise ValueError(f"parameter order differs: {n1} vs {n2}")
        p1.copy_(p2)


def target_loss_sum(model: torch.nn.Module, ids: torch.Tensor, pos: torch.Tensor, tgt: torch.Tensor,
                    autocast: bool) -> torch.Tensor:
    """Σ cross-entropy (nats) over target positions: logits only where the next token is a target."""
    dev = ids.device.type
    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=autocast):
        hid = model.model(input_ids=ids, position_ids=pos, use_cache=False).last_hidden_state
        sel = tgt[:, 1:]
        logits = model.lm_head(hid[:, :-1][sel])
    return F.cross_entropy(logits.float(), ids[:, 1:][sel], reduction="sum")


def append_locked(path: Path, line: str) -> None:
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def save_checkpoint(shadow: object, tpl_tok: object, out: Path, recipe: str, step: int, marks: Sequence[str],
                    meta: Mapping[str, object], lists: Sequence[Path]) -> Path:
    d = out / "ckpt" / f"step{step:05d}"
    t0 = time.time()
    shadow.save_pretrained(d)
    tpl_tok.save_pretrained(d)
    (d / "exp06_checkpoint.json").write_text(json.dumps({"recipe": recipe, "step": step, "marks": list(marks),
                                                         **meta}, indent=1))
    line = checkpoint_line(checkpoint_label(LABEL_PREFIX, recipe, step), str(d), recipe, step, marks)
    for p in lists:
        append_locked(p, line)
    print(json.dumps({"checkpoint": str(d), "marks": list(marks), "save_s": round(time.time() - t0, 1)}), flush=True)
    return d


# ----------------------------------------------------------------------------- effects: commands
def cmd_train(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cilab.score import load_model

    torch.manual_seed(SEED)
    device = args.device
    recipe = args.recipe
    root = Path(args.out)
    cands = [root / recipe, *(root / f"{recipe}-attempt{k}" for k in range(2, 10))]
    out = first_free(cands, [(c / "drift.csv").exists() for c in cands])
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    t_job = time.time()
    tok = AutoTokenizer.from_pretrained(BASE)
    tpl_tok = AutoTokenizer.from_pretrained(TEMPLATE_TOKENIZER)
    if tpl_tok.get_vocab() != tok.get_vocab():
        raise ValueError("template tokenizer vocabulary differs from the base tokenizer")
    corp = corpus_setup(tok, BASE)
    examples = load_examples(Path(args.data) / f"{recipe}.npz")
    n_tokens = sum(len(i) for i, _ in examples)
    weights, offset, marks, init_meta = args.base_weights or BASE, args.step_offset or 0, parse_marks(args.marks), {}
    base_ref: dict[str, np.ndarray] = {}
    if args.init_from:
        weights, init_step, prior = resolve_init(Path(args.init_from), recipe)
        offset = init_step if args.step_offset is None else args.step_offset
        if offset != init_step:
            raise ValueError(f"--step-offset {offset} but the init checkpoint is at update {init_step}")
        marks = remaining_marks(marks, saved_marks(prior, recipe))
        init_meta = json.loads((Path(weights) / "exp06_checkpoint.json").read_text())
    if args.drift_ref == "base" and weights != BASE:  # score the original base once, on this GPU, then free it
        t0 = time.time()
        ref_model = load_model(BASE, torch.bfloat16, device)
        base_ref = score_corpora(ref_model, corp, device)
        del ref_model
        release_device_memory()
        print(json.dumps({"drift_ref": "base", "base_scored_s": round(time.time() - t0, 1)}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.float32, attn_implementation="sdpa").to(device)
    model.train()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    shadow = load_model(weights, torch.bfloat16, device)
    sync_shadow(shadow, model)  # identical rounding path for step 0 and every later score
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY,
                            fused=device == "cuda")
    warm = warmup_steps(args.schedule_steps, WARMUP_FRAC)
    accum = ROWS_PER_STEP // args.micro_batch
    if accum * args.micro_batch != ROWS_PER_STEP:
        raise ValueError(f"micro-batch {args.micro_batch} does not divide {ROWS_PER_STEP}")
    env = {"recipe": recipe, "out": str(out), "device": device, "examples": len(examples), "tokens": n_tokens,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else device, "torch": torch.__version__,
           "weights": weights, "micro_batch": args.micro_batch, "accum": accum, "warmup": warm,
           "max_steps": args.max_steps, "max_wall_s": args.max_wall_s, "score_every": args.score_every,
           "doc_start": corp["start_str"], "setup_s": round(time.time() - t_job, 1),
           "schedule_steps": args.schedule_steps, "step_offset": offset, "marks": list(marks),
           "drift_ref": args.drift_ref, "init_from": args.init_from, "init_checkpoint": init_meta or None,
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "restart": os.environ.get("SLURM_RESTART_COUNT")}
    (out / "started.json").write_text(json.dumps(env, indent=1))
    print(json.dumps(env), flush=True)

    lists = [root / "checkpoints.txt", out / "checkpoints.txt"]
    log: list[tuple[int, float]] = []
    base_bits: dict[str, np.ndarray] = dict(base_ref)  # empty: the first score (update `offset`) is the reference
    stats = {"token_positions": offset * ROWS_PER_STEP * ROW_LEN, "nonpad": 0, "loss_tokens": 0, "epochs": 0.0,
             "lr": 0.0, "loss": float("nan")}

    def score(step: int, wall_s: float, final: bool) -> None:
        t0 = time.time()
        sync_shadow(shadow, model)
        bits = score_corpora(shadow, corp, device)
        score_s = time.time() - t0
        for c, b in bits.items():
            np.save(out / "tokens" / f"step{step:05d}__{c}.npy", b.astype(np.float32))
        if step == offset:
            if not base_bits:
                base_bits.update(bits)
            ref = pd.read_csv(EXP03_BPB)
            ref = ref[(ref.model == BASE) & (ref.condition == "raw")].drop_duplicates("corpus", keep="last")
            chk = base_check({c: bpb_of(base_bits[c], corp["texts"][c]) for c in CORPORA},
                             dict(zip(ref.corpus, ref.bpb, strict=True)), BASE_BPB_TOL)
            (out / "base_check.json").write_text(json.dumps(chk, indent=1))
            print(json.dumps({"base_check": chk}), flush=True)
        d = drift_row(base_bits, bits, corp["classes"], FRESH_CORPORA)
        log.append((step, d["dbpb_fresh"]))
        due = marks_due(log, marks)
        tags = [f"{m:g}" for m in due] + (["final"] if final else [])
        row = {"recipe": recipe, "step": step, "token_positions": stats["token_positions"],
               "nonpad_tokens": stats["nonpad"], "loss_tokens": stats["loss_tokens"], "epochs": stats["epochs"],
               "lr": stats["lr"], "train_loss": stats["loss"],
               **{f"bpb_{c}": bpb_of(bits[c], corp["texts"][c]) for c in CORPORA}, **d,
               "score_s": round(score_s, 1), "wall_s": round(wall_s, 1), "checkpoint": ",".join(tags)}
        append_rows(out / "drift.csv", [row], DRIFT_COLS)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                          if k in ("step", "dbpb_fresh", "contrast_ks", "contrast_rf", "train_loss", "epochs",
                                   "score_s", "checkpoint")}), flush=True)
        if step > offset and tags:
            meta = {"dbpb_fresh": d["dbpb_fresh"], "contrast_ks": d["contrast_ks"], "contrast_rf": d["contrast_rf"],
                    "token_positions": stats["token_positions"], "loss_tokens": stats["loss_tokens"],
                    "epochs": stats["epochs"], "base": weights, "template_tokenizer": TEMPLATE_TOKENIZER,
                    **({"init_from": weights, "drift_ref": args.drift_ref} if args.init_from else {})}
            save_checkpoint(shadow, tpl_tok, out, recipe, step, tags, meta, lists)

    feed = batches(examples, int(tok.pad_token_id), offset if args.init_from else 0)
    if args.init_from:  # resume the data order where the earlier stage stopped
        sk = next(feed)
        stats.update(nonpad=sk["skipped_nonpad"], loss_tokens=sk["skipped_loss"], epochs=sk["epochs"])
        print(json.dumps(sk), flush=True)
    t_train = time.time()
    score(offset, 0.0, False)
    local, step, recent_loss, recent_tgt, since_loss, since_tgt = 0, offset, 0.0, 0, 0.0, 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    while True:
        lr = lr_at(local, PEAK_LR, warm, args.schedule_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        b = next(feed)
        ids = torch.from_numpy(b["ids"]).to(device, non_blocking=True)
        pos = torch.from_numpy(b["pos"]).to(device, non_blocking=True)
        tgt = torch.from_numpy(b["tgt"]).to(device, non_blocking=True)
        n_tgt = int(b["tgt"][:, 1:].sum())
        loss_sum = 0.0
        for k in range(accum):
            sl = slice(k * args.micro_batch, (k + 1) * args.micro_batch)
            ls = target_loss_sum(model, ids[sl], pos[sl], tgt[sl], autocast=True)
            (ls / n_tgt).backward()
            loss_sum += float(ls.detach())
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
        opt.step()
        opt.zero_grad(set_to_none=True)
        local += 1
        step = offset + local
        stats["token_positions"] += ROWS_PER_STEP * ROW_LEN
        stats["nonpad"] += b["nonpad"]
        stats["loss_tokens"] += n_tgt
        stats["epochs"], stats["lr"] = b["epochs"], lr
        recent_loss += loss_sum
        recent_tgt += n_tgt
        since_loss += loss_sum
        since_tgt += n_tgt
        wall = time.time() - t_train
        if step % LOG_EVERY == 0:
            tr = {"recipe": recipe, "step": step, "lr": lr, "loss": recent_loss / recent_tgt, "grad_norm": gnorm,
                  "epochs": b["epochs"], "token_positions": stats["token_positions"],
                  "loss_tokens": stats["loss_tokens"], "wall_s": round(wall, 1),
                  "tok_per_s": round(stats["token_positions"] / max(wall, 1e-9), 1),
                  "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else 0.0}
            append_rows(out / "train.csv", [tr], TRAIN_COLS)
            print(json.dumps(tr), flush=True)
            recent_loss, recent_tgt = 0.0, 0
        reason = stop_reason(local, args.max_steps, wall, args.max_wall_s)
        if step % args.score_every == 0 or reason:
            stats["loss"] = since_loss / since_tgt  # mean target loss (nats) since the previous score
            score(step, wall, final=reason is not None)
            since_loss, since_tgt = 0.0, 0
        if reason:
            break
    done = {**env, "stop": reason, "steps": step, "train_wall_s": round(time.time() - t_train, 1),
            "job_wall_s": round(time.time() - t_job, 1), "token_positions": stats["token_positions"],
            "loss_tokens": stats["loss_tokens"], "epochs": stats["epochs"],
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else 0.0}
    (out / "done.json").write_text(json.dumps(done, indent=1))
    print(json.dumps({"done": done}), flush=True)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Packed row == examples alone (max |Δlogit|), and target-only loss == transformers' labels= loss."""
    from transformers import Olmo2Config, Olmo2ForCausalLM

    torch.manual_seed(SEED)
    dev = args.device
    cfg = Olmo2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64, pad_token_id=0)
    cfg._attn_implementation = "sdpa"
    model = Olmo2ForCausalLM(cfg).to(dev).float().eval()
    rng = np.random.default_rng(0)
    exs = [(rng.integers(1, 128, n), rng.random(n) < 0.6) for n in (7, 12, 5)]
    for _, m in exs:
        m[0] = False
    ids, pos, tgt = (torch.from_numpy(a)[None].to(dev) for a in row_arrays(exs, 32, 0))
    with torch.no_grad():
        packed = model(input_ids=ids, position_ids=pos, use_cache=False).logits[0]
        alone = [model(input_ids=torch.from_numpy(i)[None].to(dev), use_cache=False).logits[0] for i, _ in exs]
        offs = np.cumsum([0, *(len(i) for i, _ in exs)])
        diff = max(float((packed[a:b] - x).abs().max()) for a, b, x in zip(offs[:-1], offs[1:], alone, strict=True))
        no_pos = model(input_ids=ids, use_cache=False).logits[0]  # without restarts: example 2 sees example 1
        leak = float((no_pos[offs[1]:offs[2]] - alone[1]).abs().max())
        i0, m0 = exs[1]
        x = torch.from_numpy(i0)[None].to(dev)
        labels = torch.where(torch.from_numpy(m0)[None].to(dev), x, torch.full_like(x, -100))
        hf = float(model(input_ids=x, labels=labels, use_cache=False).loss)
        mine = float(target_loss_sum(model, x, torch.arange(len(i0), device=dev)[None],
                                     torch.from_numpy(m0)[None].to(dev), autocast=False)) / int(m0[1:].sum())
        row_sum = float(target_loss_sum(model, ids, pos, tgt, autocast=False))
        parts = sum(float(target_loss_sum(model, torch.from_numpy(i)[None].to(dev),
                                          torch.arange(len(i), device=dev)[None], torch.from_numpy(m)[None].to(dev),
                                          autocast=False)) for i, m in exs)
    res = {"device": dev, "packed_vs_alone_max_abs_logit": diff, "leak_without_restart": leak,
           "hf_loss": hf, "target_loss": mine, "packed_loss_sum": row_sum, "alone_loss_sum": parts,
           "ok": diff < SELFTEST_TOL < leak and abs(hf - mine) < SELFTEST_TOL * abs(hf)
           and abs(row_sum - parts) < SELFTEST_TOL * abs(parts)}
    print(json.dumps(res), flush=True)
    return 0 if res["ok"] else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--recipe", choices=RECIPES, required=True)
    t.add_argument("--data", required=True, help="prep_data.py output directory")
    t.add_argument("--out", required=True, help="run directory ($CI_OUT)")
    t.add_argument("--max-steps", type=int, default=TOTAL_STEPS, help="updates in this run (after the offset)")
    t.add_argument("--schedule-steps", type=int, default=TOTAL_STEPS, help="cosine length; warmup = 3%% of it")
    t.add_argument("--marks", default=",".join(f"{m:g}" for m in MARKS), help="Δbpb_fresh marks, %%")
    t.add_argument("--init-from", default=None, help="stage 2: a checkpoint dir, or a run dir whose checkpoints.txt "
                   "names this recipe's final checkpoint; continues its data order and step numbering")
    t.add_argument("--drift-ref", choices=("init", "base"), default="init",
                   help="drift reference: the first scored weights (init) or the original OLMo-2 base")
    t.add_argument("--step-offset", type=int, default=None, help="first update number (default: the init's step)")
    t.add_argument("--max-wall-s", type=float, default=MAX_WALL_S)
    t.add_argument("--score-every", type=int, default=SCORE_EVERY)
    t.add_argument("--micro-batch", type=int, default=2)
    t.add_argument("--grad-ckpt", action="store_true")
    t.add_argument("--device", default="cuda")
    t.add_argument("--base-weights", default=None, help="weights to start from (default: the base); smoke tests only")
    s = sub.add_parser("selftest")
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    import transformers
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    raise SystemExit({"train": cmd_train, "selftest": cmd_selftest}[args.cmd](args))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

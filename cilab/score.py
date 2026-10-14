"""Bits from a causal LM. `summarize` is pure; everything below the marker touches a model or device.

Effect functions: pick_device, load_model, free_model, release_device_memory, score_tokens, score_raw,
score_chat, hf_loss_bits, target_logits.
-log2 p is always computed from float32 logits, whatever dtype the weights are in.
"""
from __future__ import annotations

import gc
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cilab.text import Window, encode

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

LN2 = math.log(2.0)


def summarize(bits: np.ndarray, text: str, meta: Mapping[str, object]) -> dict[str, object]:
    """Row with unit conversions: bpb = total_bits / UTF-8 bytes, bpc = total_bits / chars."""
    nbytes, nchars, ntok = len(text.encode("utf-8")), len(text), int(bits.size)
    total = float(bits.sum(dtype=np.float64))
    return {**meta, "bytes": nbytes, "chars": nchars, "tokens": ntok, "total_bits": round(total, 3),
            "bpb": round(total / nbytes, 5), "bpc": round(total / nchars, 5),
            "tokens_per_byte": round(ntok / nbytes, 5), "bits_per_token": round(total / ntok, 5)}


# ----------------------------------------------------------------------------- effects: model and device
def pick_device() -> str:
    """cuda (cluster GPUs) if available, else mps (Apple silicon), else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(model_id: str, dtype: torch.dtype, device: str) -> PreTrainedModel:
    """Load weights directly in `dtype` (never convert in-process: MPS keeps the old copy alive)."""
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    return m.to(device).eval()


def free_model(model: PreTrainedModel) -> None:
    """Drop this reference and release cached device memory; callers must drop theirs too."""
    del model
    release_device_memory()


def release_device_memory() -> None:
    """Collect garbage and return cached allocator blocks on CUDA and MPS."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.no_grad()
def score_tokens(model: PreTrainedModel, input_ids: list[int], n_score: int, device: str) -> np.ndarray:
    """-log2 p (bits) of the last n_score tokens of input_ids, each conditioned on everything before it."""
    x = torch.tensor([input_ids], device=device)
    logits = model(input_ids=x, use_cache=False).logits[0]
    lo = len(input_ids) - n_score
    pred = logits[lo - 1:-1].float()
    nats = F.cross_entropy(pred, x[0, lo:], reduction="none")
    return (nats / LN2).cpu().numpy().astype(np.float64)


def score_raw(model: PreTrainedModel, wins: Sequence[Window], ids: list[int], start_id: int,
              device: str) -> np.ndarray:
    """RAW condition: per-token bits of every window scored as [start_id, *window]."""
    return np.concatenate([score_tokens(model, [start_id, *ids[b:e]], e - b, device) for b, e, _ in wins])


def score_chat(model: PreTrainedModel, tok: PreTrainedTokenizerBase, wins: Sequence[Window],
               prefix_ids: list[int], device: str) -> np.ndarray:
    """CHAT condition: per-token bits of each window's text tokens after the chat prefix."""
    out = []
    for _, _, s in wins:
        t = encode(tok, s)
        out.append(score_tokens(model, prefix_ids + t, len(t), device))
    return np.concatenate(out)


@torch.no_grad()
def hf_loss_bits(model: PreTrainedModel, input_ids: list[int], device: str) -> float:
    """Sum of bits implied by HF's own labels= loss (mean nats over len-1 shifted targets)."""
    x = torch.tensor([input_ids], device=device)
    loss = model(input_ids=x, labels=x, use_cache=False).loss.float().item()
    return loss * (len(input_ids) - 1) / LN2


@torch.no_grad()
def target_logits(model: PreTrainedModel, input_ids: list[int], n_score: int,
                  device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(float32 logits (n_score, V), target ids (n_score,)) for the last n_score tokens, on device.

    Same slice as score_tokens: row i predicts target i from everything before it."""
    x = torch.tensor([input_ids], device=device)
    logits = model(input_ids=x, use_cache=False).logits[0]
    lo = len(input_ids) - n_score
    return logits[lo - 1:-1].float(), x[0, lo:]

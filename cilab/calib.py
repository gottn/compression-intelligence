"""Temperature calibration: bits under softmax(logits / T), the T* fit, and gap-recovery statistics.

All pure. `bits_at_temperatures` runs on whatever device its tensors live on (MPS has no float64, so
per-token bits are computed in float32 there and summed in float64 on the CPU).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch

LN2 = math.log(2.0)
T_GRID: tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(23))  # 0.50 … 1.60


def bits_at_temperatures(logits: torch.Tensor, targets: torch.Tensor, temps: Sequence[float]) -> np.ndarray:
    """Total bits of `targets` under softmax(logits / T), one value per T.

    logits: (n, V) float32 scores predicting each target; targets: (n,) token ids.
    bits_T = Σ_i [logsumexp(z_i / T) − z_i[t_i] / T] / ln 2.
    """
    z_t = logits.gather(1, targets[:, None])[:, 0]
    per_token = torch.stack([(torch.logsumexp(logits / t, dim=-1) - z_t / t) / LN2 for t in temps])
    return per_token.cpu().numpy().astype(np.float64).sum(axis=1)


def parabola_vertex(x: Sequence[float], y: Sequence[float]) -> float:
    """x of the extremum of the parabola through three points (x0, y0), (x1, y1), (x2, y2)."""
    (x0, x1, x2), (y0, y1, y2) = x, y
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
    return -b / (2 * a)


def fit_temperature(temps: Sequence[float], total_bits: np.ndarray) -> tuple[float, bool]:
    """(T*, at_edge): grid argmin of total bits, refined by a 3-point parabola in log T.

    At a grid edge there is no bracketing neighbour, so T* is the edge value and at_edge is True.
    The refined vertex is clamped to the bracketing interval.
    """
    k = int(np.argmin(total_bits))
    if k in (0, len(temps) - 1):
        return float(temps[k]), True
    x = np.log(np.asarray(temps[k - 1:k + 2], dtype=np.float64))
    v = parabola_vertex(x, np.asarray(total_bits[k - 1:k + 2], dtype=np.float64))
    return float(np.exp(np.clip(v, x[0], x[2]))), False


def gap_recovered(base_t1: float, post_t1: float, post_tstar: float) -> float:
    """Fraction of the post−base gap at T = 1 that the post model's own T* removes."""
    return (post_t1 - post_tstar) / (post_t1 - base_t1)


def recovered_stat(base_t1: np.ndarray, post_t1: np.ndarray,
                   post_tstar: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """bootstrap_ci statistic: gap_recovered on the per-window bit sums of each resampled row.

    The windows are the same text for base and post (identical tokenizers), so bytes cancel and
    the bpb ratio equals the bits ratio."""
    return lambda ix: ((post_t1[ix].sum(1) - post_tstar[ix].sum(1))
                       / (post_t1[ix].sum(1) - base_t1[ix].sum(1)))

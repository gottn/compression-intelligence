"""Pure statistics over per-token and per-window bits: position buckets and window bootstrap."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

BUCKETS: tuple[tuple[int, int], ...] = ((0, 1), (1, 16), (16, 128), (128, 10**9))


def bucket_stats(bits: np.ndarray, pos: np.ndarray,
                 buckets: tuple[tuple[int, int], ...] = BUCKETS) -> dict[str, float]:
    """Mean bits/token (bpt_*) and share of total bits (share_bits_*) per in-window position bucket."""
    out = {}
    for lo, hi in buckets:
        m = (pos >= lo) & (pos < hi)
        name = f"pos{lo}" + (f"-{hi - 1}" if hi < 10**9 else "+")
        out[f"bpt_{name}"] = float(bits[m].mean()) if m.any() else float("nan")
        out[f"share_bits_{name}"] = float(bits[m].sum() / bits.sum())
    return out


def window_sums(bits: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Per-window bit totals from per-token bits and in-window positions (a window starts at pos 0)."""
    starts = np.flatnonzero(pos == 0)
    return np.add.reduceat(bits, starts)


def resample_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) window indices drawn with replacement; a pure function of the seed."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(stat: Callable[[np.ndarray], np.ndarray], idx: np.ndarray,
                 level: float = 0.95) -> tuple[float, float, float]:
    """(point, lo, hi) of a window-level statistic.

    `stat` maps an (k, n) index matrix to k values (one per row of resampled windows); the point
    estimate is `stat` on the identity resample, the CI the percentile interval over the rows of idx.
    """
    point = float(stat(np.arange(idx.shape[1])[None, :])[0])
    a = 100.0 * level
    lo, hi = np.percentile(stat(idx), [(100.0 - a) / 2, (100.0 + a) / 2])
    return point, float(lo), float(hi)


def paired_pct_change(base: np.ndarray, post: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """bootstrap_ci statistic: 100 * (Σ post − Σ base) / Σ base over each row of resampled windows."""
    return lambda ix: 100 * (post[ix].sum(1) - base[ix].sum(1)) / base[ix].sum(1)

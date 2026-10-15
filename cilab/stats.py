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


def window_totals(token_bits: np.ndarray, lengths: list[int]) -> np.ndarray:
    """Per-window totals of consecutive per-token bits, given each window's token count."""
    starts = np.cumsum([0, *lengths])[:-1]
    return np.add.reduceat(token_bits, starts)


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


def paired_share(base_part: np.ndarray, post_part: np.ndarray, base_all: np.ndarray,
                 post_all: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """bootstrap_ci statistic: 100 * (Σ post_part − Σ base_part) / (Σ post_all − Σ base_all) over each row of
    resampled windows, the percentage of the total change in bits that falls on one part (a token class)."""
    return lambda ix: (100 * (post_part[ix].sum(1) - base_part[ix].sum(1))
                       / (post_all[ix].sum(1) - base_all[ix].sum(1)))


def stratified_mean_ci(stats: list[Callable[[np.ndarray], np.ndarray]], idxs: list[np.ndarray],
                       level: float = 0.95) -> tuple[float, float, float]:
    """(point, lo, hi) of the mean over strata (corpora) of a per-stratum window statistic.

    Windows are resampled independently within each stratum: stats[c] maps an (k, n_c) index matrix to
    k values and idxs[c] is that stratum's (n_boot, n_c) resample; every stratum needs the same n_boot.
    The point estimate is the mean of each statistic on its identity resample.
    """
    point = float(np.mean([s(np.arange(ix.shape[1])[None, :])[0] for s, ix in zip(stats, idxs, strict=True)]))
    rows = np.mean([s(ix) for s, ix in zip(stats, idxs, strict=True)], axis=0)
    a = 100.0 * level
    lo, hi = np.percentile(rows, [(100.0 - a) / 2, (100.0 + a) / 2])
    return point, float(lo), float(hi)


# ----------------------------------------------------------------------------- across checkpoints (exp04)
def rank_avg(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties given their average rank (as scipy.stats.rankdata 'average')."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        tie = x == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r; nan when either column is constant."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dx, dy = x - x.mean(), y - y.mean()
    den = np.sqrt((dx * dx).sum() * (dy * dy).sum())
    return float((dx * dy).sum() / den) if den > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho = Pearson r of average ranks."""
    return pearson(rank_avg(x), rank_avg(y))


def bootstrap_corr(x: np.ndarray, y: np.ndarray, corr: Callable[[np.ndarray, np.ndarray], float],
                   n_boot: int, seed: int, level: float = 0.95) -> tuple[float, float, float, int]:
    """(point, lo, hi, n_valid): percentile CI of a correlation over items resampled with replacement.
    Resamples where the statistic is undefined (a constant column) are dropped and counted out of n_valid."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    idx = resample_indices(len(x), n_boot, seed)
    vals = np.array([corr(x[i], y[i]) for i in idx])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return corr(x, y), float("nan"), float("nan"), 0
    a = 100.0 * level
    lo, hi = np.percentile(vals, [(100.0 - a) / 2, (100.0 + a) / 2])
    return corr(x, y), float(lo), float(hi), len(vals)


def leave_one_out(x: np.ndarray, y: np.ndarray, corr: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """The statistic with each item left out in turn (entry i = without item i)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = ~np.eye(len(x), dtype=bool)
    return np.array([corr(x[k], y[k]) for k in keep])


def ols_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(slope, intercept) of the least-squares line y = slope * x + intercept."""
    slope, intercept = np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), 1)
    return float(slope), float(intercept)


def bootstrap_line_band(x: np.ndarray, y: np.ndarray, grid: np.ndarray, n_boot: int, seed: int,
                        level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise percentile band of the OLS line on `grid`, refitting on items resampled with replacement
    (resamples with a constant x are skipped)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    fits = []
    for i in resample_indices(len(x), n_boot, seed):
        if np.ptp(x[i]) > 0:
            s, b = ols_line(x[i], y[i])
            fits.append(s * grid + b)
    a = 100.0 * level
    lo, hi = np.percentile(np.array(fits), [(100.0 - a) / 2, (100.0 + a) / 2], axis=0)
    return lo, hi

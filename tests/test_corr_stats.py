from __future__ import annotations

import numpy as np

from cilab.stats import (
    bootstrap_corr,
    bootstrap_line_band,
    leave_one_out,
    ols_line,
    pearson,
    rank_avg,
    spearman,
)


def test_rank_avg_handles_ties() -> None:
    assert rank_avg(np.array([10.0, 30.0, 20.0, 20.0])).tolist() == [1.0, 4.0, 2.5, 2.5]


def test_pearson_and_spearman_reference_values() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(pearson(x, 2 * x + 1), 1.0) and np.isclose(pearson(x, -x), -1.0)
    y = np.array([2.0, 1.0, 4.0, 3.0, 5.0])
    assert np.isclose(pearson(x, y), np.corrcoef(x, y)[0, 1])
    assert np.isclose(spearman(x, x**3), 1.0)  # monotone, not linear
    assert np.isclose(spearman(x, y), 0.8)  # 1 - 6*4/(5*24)
    assert np.isnan(pearson(x, np.ones(5)))


def test_bootstrap_corr_brackets_point_and_is_seeded() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=30)
    y = x + rng.normal(scale=0.5, size=30)
    p, lo, hi, n = bootstrap_corr(x, y, pearson, 2000, seed=0)
    assert lo < p < hi and n == 2000
    assert bootstrap_corr(x, y, pearson, 2000, seed=0) == (p, lo, hi, n)


def test_bootstrap_corr_drops_degenerate_resamples() -> None:
    x = np.array([0.0, 0.0, 0.0, 1.0])
    _, _, _, n = bootstrap_corr(x, np.array([1.0, 2.0, 3.0, 4.0]), pearson, 500, seed=3)
    assert 0 < n < 500  # some resamples draw only zeros
    p, lo, hi, n = bootstrap_corr(np.arange(4.0), np.ones(4), pearson, 50, seed=0)
    assert np.isnan(p) and np.isnan(lo) and np.isnan(hi) and n == 0


def test_leave_one_out_and_ols() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    loo = leave_one_out(x, y, pearson)
    assert len(loo) == 5 and np.isclose(loo[4], 1.0)  # dropping the outlier restores a perfect line
    assert np.allclose(ols_line(np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0])), (2.0, 1.0))


def test_bootstrap_line_band_contains_exact_line() -> None:
    x = np.arange(10.0)
    y = 0.5 * x + 1
    lo, hi = bootstrap_line_band(x, y, np.array([0.0, 5.0, 9.0]), 200, seed=0)
    assert np.allclose(lo, [1.0, 3.5, 5.5]) and np.allclose(hi, [1.0, 3.5, 5.5])

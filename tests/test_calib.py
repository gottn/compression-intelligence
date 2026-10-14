from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cilab.calib import (
    LN2,
    T_GRID,
    bits_at_temperatures,
    fit_temperature,
    gap_recovered,
    parabola_vertex,
    recovered_stat,
)
from cilab.stats import bootstrap_ci, resample_indices


def test_grid_is_23_values_from_050_to_160() -> None:
    assert len(T_GRID) == 23 and T_GRID[0] == 0.5 and T_GRID[-1] == 1.6 and 1.0 in T_GRID


def test_bits_at_t1_equal_cross_entropy() -> None:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(40, 97, generator=g) * 3
    targets = torch.randint(0, 97, (40,), generator=g)
    ce_bits = float(F.cross_entropy(logits, targets, reduction="sum")) / LN2
    out = bits_at_temperatures(logits, targets, (0.7, 1.0, 1.3))
    assert out.shape == (3,)
    assert out[1] == pytest.approx(ce_bits, rel=1e-6)
    scaled = float(F.cross_entropy(logits / 0.7, targets, reduction="sum")) / LN2
    assert out[0] == pytest.approx(scaled, rel=1e-6)


def test_overconfident_model_prefers_t_above_one() -> None:
    g = torch.Generator().manual_seed(1)
    true_logits = torch.randn(4000, 20, generator=g)
    targets = torch.distributions.Categorical(logits=true_logits).sample()
    bits = bits_at_temperatures(true_logits * 1.5, targets, T_GRID)  # sharpened by 1.5
    t_star, at_edge = fit_temperature(T_GRID, bits)
    assert not at_edge
    assert t_star == pytest.approx(1.5, abs=0.1)


def test_parabola_vertex_recovers_known_minimum() -> None:
    x = np.array([-0.2, 0.0, 0.15])
    assert parabola_vertex(x, 3.0 * (x - 0.05) ** 2 + 7.0) == pytest.approx(0.05)


def test_fit_temperature_on_grid_curve_and_edges() -> None:
    temps = np.array(T_GRID)
    curve = 100 + 50 * (np.log(temps) - np.log(1.23)) ** 2
    t_star, at_edge = fit_temperature(T_GRID, curve)
    assert t_star == pytest.approx(1.23, rel=1e-9) and not at_edge
    assert fit_temperature(T_GRID, -temps) == (1.6, True)
    assert fit_temperature(T_GRID, temps) == (0.5, True)


def test_gap_recovered_and_bootstrap_contains_point() -> None:
    assert gap_recovered(base_t1=10.0, post_t1=12.0, post_tstar=11.0) == 0.5
    rng = np.random.default_rng(2)
    base = rng.gamma(5.0, 300.0, size=12)
    post1 = base * 1.15 + rng.normal(0, 20, size=12)
    postt = post1 - 0.6 * (post1 - base) + rng.normal(0, 10, size=12)
    point, lo, hi = bootstrap_ci(recovered_stat(base, post1, postt), resample_indices(12, 2000, seed=0))
    assert point == pytest.approx(gap_recovered(base.sum(), post1.sum(), postt.sum()))
    assert lo < point < hi

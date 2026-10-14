from __future__ import annotations

import numpy as np

from cilab.score import summarize
from cilab.stats import bootstrap_ci, bucket_stats, paired_pct_change, resample_indices, window_sums


def test_summarize_bpb_is_bits_over_utf8_bytes() -> None:
    text = "héllo wörld 🙂"  # 13 chars, 18 bytes
    bits = np.array([3.5, 1.25, 7.0, 0.5])
    row = summarize(bits, text, {"model": "m"})
    assert row["model"] == "m"
    assert (row["bytes"], row["chars"], row["tokens"]) == (len(text.encode("utf-8")), len(text), 4)
    assert row["bytes"] > row["chars"]
    assert row["total_bits"] == 12.25
    assert row["bpb"] == round(12.25 / row["bytes"], 5)
    assert row["bpc"] == round(12.25 / row["chars"], 5)
    assert row["bits_per_token"] == round(12.25 / 4, 5)


def test_window_sums_and_bucket_stats() -> None:
    bits = np.array([4.0, 2.0, 1.0, 5.0, 3.0])
    pos = np.array([0, 1, 2, 0, 1])
    assert window_sums(bits, pos).tolist() == [7.0, 8.0]
    st = bucket_stats(bits, pos)
    assert st["bpt_pos0-0"] == 4.5
    assert st["bpt_pos1-15"] == 2.0
    assert np.isnan(st["bpt_pos128+"])
    assert st["share_bits_pos0-0"] == 9.0 / 15.0


def test_bootstrap_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(1)
    base = rng.gamma(5.0, 200.0, size=30)
    post = base * 1.1 + rng.normal(0, 30, size=30)
    point, lo, hi = bootstrap_ci(paired_pct_change(base, post), resample_indices(30, 2000, seed=0))
    assert lo < point < hi
    assert np.isclose(point, 100 * (post.sum() - base.sum()) / base.sum())


def test_resample_indices_is_pure_in_seed() -> None:
    a, b = resample_indices(7, 50, seed=3), resample_indices(7, 50, seed=3)
    assert a.shape == (50, 7) and (a == b).all() and a.min() >= 0 and a.max() < 7

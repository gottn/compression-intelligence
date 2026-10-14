from __future__ import annotations

import numpy as np
import pytest

from cilab.stages import (
    FAMILIES,
    STAGES,
    Stage,
    comparisons,
    family_base,
    stage_of,
    tokenizer_groups,
)
from cilab.stats import bootstrap_ci, paired_pct_change, resample_indices, stratified_mean_ci
from exp03_staged_drift.score_stages import (
    choose_dtype,
    compare_ids,
    ids_digest,
    pending_corpora,
    stage_meta,
    window_rows,
)

TOY = (
    Stage("o/base", "f", "base", "pretrain", None, "t"),
    Stage("o/sft", "f", "sft", "sft", "o/base", "t"),
    Stage("o/dpo", "f", "dpo", "dpo", "o/sft", "t"),
    Stage("o/rl0", "f", "rl-zero", "grpo", "o/base", "t"),
)


def test_stage_table_is_well_formed() -> None:
    models = [s.model for s in STAGES]
    assert len(models) == len(set(models)) == 14
    assert {s.family for s in STAGES} == set(FAMILIES)
    for s in STAGES:
        assert s.parent is None or stage_of(s.parent).family == s.family
    assert family_base("olmo2-1b") == "allenai/OLMo-2-0425-1B"
    assert stage_of("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").parent == "Qwen/Qwen2.5-Math-1.5B"


def test_comparisons_add_parent_only_when_it_is_not_the_base() -> None:
    assert comparisons(TOY) == [("o/sft", "o/base", "vs_base"), ("o/dpo", "o/base", "vs_base"),
                                ("o/dpo", "o/sft", "vs_parent"), ("o/rl0", "o/base", "vs_base")]
    r1 = [c for c in comparisons() if c[0] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]
    assert r1 == [("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-1.5B", "vs_base"),
                  ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-Math-1.5B", "vs_parent")]


def test_family_base_and_stage_of_reject_ambiguity() -> None:
    with pytest.raises(ValueError):
        family_base("f", (*TOY, Stage("o/b2", "f", "base", "x", None, "t")))
    with pytest.raises(KeyError):
        stage_of("o/missing", TOY)


def test_tokenizer_groups() -> None:
    g = tokenizer_groups()
    assert list(g) == ["olmo2", "qwen2.5"]
    assert len(g["olmo2"]) == 4 and len(g["qwen2.5"]) == 10


def test_stratified_mean_ci_point_is_mean_of_strata() -> None:
    rng = np.random.default_rng(2)
    stats, idxs, points = [], [], []
    for c, n in enumerate((20, 25, 30)):
        base = rng.gamma(5.0, 200.0, size=n)
        post = base * (1.02 + 0.02 * c) + rng.normal(0, 20, size=n)
        idx = resample_indices(n, 2000, seed=c)
        stats.append(paired_pct_change(base, post))
        idxs.append(idx)
        points.append(bootstrap_ci(stats[-1], idx)[0])
    point, lo, hi = stratified_mean_ci(stats, idxs)
    assert np.isclose(point, np.mean(points))
    assert lo < point < hi


def test_choose_dtype() -> None:
    assert choose_dtype(1000.0, 1001.0) == "float16"
    assert choose_dtype(1000.0, 1100.0) == "float32"
    assert choose_dtype(float("inf"), 1000.0) == "float32"
    assert choose_dtype(float("nan"), None) == "float32"
    assert choose_dtype(1000.0, None) == "float16"


def test_ids_digest_and_compare_ids() -> None:
    assert ids_digest([1, 2, 3]) == ids_digest((1, 2, 3)) != ids_digest([1, 2, 4])
    assert len(ids_digest([])) == 16
    assert compare_ids({"a": [1, 2], "b": [3]}, {"a": [1, 2], "b": [4], "c": [9]}) == {"a": True, "b": False}


def test_pending_corpora_and_rows() -> None:
    done = {("m", "raw", "wiki2026"), ("m", "chat", "arxiv2026"), ("x", "raw", "arxiv2026")}
    assert pending_corpora("m", ["wiki2026", "arxiv2026"], done) == ["arxiv2026"]
    rows = window_rows("m", "c", [(0, 3, "ab"), (3, 5, "é")])
    assert [(r["window"], r["tokens"], r["bytes"]) for r in rows] == [(0, 3, 2), (1, 2, 2)]
    meta = stage_meta(TOY[2])
    assert (meta["pair"], meta["role"], meta["parent"]) == ("f", "dpo", "o/sft")
    assert stage_meta(TOY[0])["parent"] == ""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cilab.pairs import CORPORA, FRESH_CORPORA
from exp03_staged_drift.analyze_stages import (
    FRESH,
    OLMO,
    delta_table,
    expectation_table,
    pct_label,
    reproduction_table,
    rl_headline,
    step_rows,
)

CHAIN = (f"{OLMO}-SFT", f"{OLMO}-DPO", f"{OLMO}-Instruct")
SCALE = {OLMO: 1.0, CHAIN[0]: 1.05, CHAIN[1]: 1.06, CHAIN[2]: 1.0594}  # bits relative to the base


def toy(scale: dict[str, float]) -> tuple[dict, dict]:
    rng = np.random.default_rng(0)
    base = {c: rng.gamma(5.0, 100.0, size=12) for c in CORPORA}
    wbits = {(m, c): base[c] * s for m, s in scale.items() for c in CORPORA}
    lengths = {(m, c): (1024,) * 12 for m in scale for c in CORPORA}
    return wbits, lengths


def olmo_comps() -> list[tuple[str, str, str]]:
    return [(CHAIN[0], OLMO, "vs_base"), (CHAIN[1], OLMO, "vs_base"), (CHAIN[1], CHAIN[0], "vs_parent"),
            (CHAIN[2], OLMO, "vs_base"), (CHAIN[2], CHAIN[1], "vs_parent")]


def test_delta_table_points_are_exact_ratios_with_a_fresh_mean_row() -> None:
    d = delta_table(*toy(SCALE), olmo_comps())
    assert len(d) == len(olmo_comps()) * (len(CORPORA) + 1)
    sft = d[(d.model == CHAIN[0]) & (d.reference == OLMO)]
    assert np.allclose(sft.delta_pct, 5.0)
    step = d[(d.model == CHAIN[2]) & (d.reference == CHAIN[1]) & (d.corpus == FRESH)].iloc[0]
    assert np.isclose(step.delta_pct, 100 * (1.0594 / 1.06 - 1))
    assert step.n_windows == 12 * len(FRESH_CORPORA)


def test_delta_table_rejects_different_windows() -> None:
    wbits, lengths = toy(SCALE)
    lengths[(CHAIN[0], "wiki2026")] = (1023,) * 12
    with pytest.raises(ValueError):
        delta_table(wbits, lengths, olmo_comps())


def test_steps_expectations_and_rl_headline() -> None:
    d = delta_table(*toy(SCALE), olmo_comps())
    steps = step_rows(d)
    assert set(zip(steps.model, steps.reference, strict=True)) == {(CHAIN[0], OLMO), (CHAIN[1], CHAIN[0]),
                                                                   (CHAIN[2], CHAIN[1])}
    e = expectation_table(d).set_index("id")
    assert e.loc["E1", "met"] == "met" and "yes" in e.loc["E1", "note"]  # +5% and the largest step
    assert e.loc["E2", "met"] == "met"  # +0.95%
    assert e.loc["E3", "met"] == "met"  # -0.06%
    assert e.loc["E4", "met"] == "n/a"  # Qwen models absent from the toy table
    rl = rl_headline(d)
    assert set(rl.model) == {CHAIN[2]}
    assert rl.set_index("corpus").loc[FRESH, "below_predecessor"]  # identical windows -> zero-width CI below 0


def test_reproduction_table_takes_the_last_exp01_row() -> None:
    bpb = pd.DataFrame([{"model": "m", "corpus": "c", "bpb": 0.80005, "total_bits": 800.05}])
    exp01 = pd.DataFrame([{"model": "m", "corpus": "c", "condition": "raw", "bpb": 0.7, "total_bits": 700.0},
                          {"model": "m", "corpus": "c", "condition": "raw", "bpb": 0.8, "total_bits": 800.0}])
    tok = {("m", "c"): (np.array([1.0, 2.0]), np.array([1.0, 2.5]))}
    r = reproduction_table(bpb, exp01, tok).iloc[0]
    assert np.isclose(r.diff_bpb, 5e-5) and r.within_tol and r.max_token_abs_bits == 0.5


def test_pct_label() -> None:
    assert pct_label(0.123) == "+0.12%"
    assert pct_label(-14.06) == "-14.1%"

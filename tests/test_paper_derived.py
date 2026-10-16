"""Guards the derived numbers the report quotes (paper/derived.py) against the committed results."""
from __future__ import annotations

from pathlib import Path

import pytest

from paper.derived import (
    Inputs,
    absolute_bpb,
    contrast_table,
    cost,
    docstart,
    exp04_per_task,
    exp06_misc,
    exp06_per_task,
    gate_families,
    range_of_three_factor,
    read_inputs,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inp() -> Inputs:
    return read_inputs(REPO)


def test_cost(inp: Inputs) -> None:
    c = cost(inp)
    assert c["loglik_median_s"] == pytest.approx(167.0, abs=0.5)
    assert c["drift_t4_median_s"] == pytest.approx(14.95, abs=0.01)
    assert 11 < c["ratio_exp04"] < 11.5


def test_absolute_bpb_and_gate(inp: Inputs) -> None:
    a = absolute_bpb(inp)
    assert a["r_post"] == pytest.approx(-0.81, abs=0.005) and a["r_base"] == pytest.approx(-0.97, abs=0.005)
    g = gate_families(inp, n_boot=2000, n_perm=2000)
    assert g["r"] == pytest.approx(0.887, abs=5e-4) and g["cluster_lo"] < 0 < g["cluster_hi"]
    d = docstart(inp)
    assert d["math_vs_base_nl"] == pytest.approx(16.03, abs=0.01) and d["gate_r_nl"] == pytest.approx(0.835, abs=0.001)


def test_exp06_contrasts_reproduce_and_overlap(inp: Inputs) -> None:
    _, pairs = contrast_table(inp)  # raises if a point differs from checkpoints.csv by more than 0.02 pp
    six = pairs[(pairs["mark"] == "6") & (pairs["pair"] == "mathcot - tulu")].iloc[0]
    assert six.lo < 0 < six.hi
    m = exp06_misc(inp)
    assert m["r_forget_gain_12"] == pytest.approx(-0.935, abs=0.001)
    assert m["r_drift_gain_12"] == pytest.approx(0.133, abs=0.001)


def test_forgetting_is_arc_led(inp: Inputs) -> None:
    e6 = exp06_per_task(inp)
    assert e6.loc["mathcot", "forget_no_arc"].max() < 1.6
    e4 = exp04_per_task(inp)
    assert abs(e4.loc["HuggingFaceTB/SmolLM2-1.7B-Instruct", "forget_no_arc"]) < 0.1
    assert range_of_three_factor() == pytest.approx(1.196, abs=0.002)

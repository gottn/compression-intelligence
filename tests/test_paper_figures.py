"""Guards the report's figures: the data functions reproduce the headline numbers of exp01–exp06 from the
committed results (each data function also checks every plotted number against RESULTS.md and raises)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from paper.figures.make_figures import (
    RC,
    Inputs,
    f1_counts,
    f1_data,
    f1_figure,
    f2_data,
    f2_figure,
    f2_means,
    f3_data,
    f3_figure,
    f4_data,
    f4_figure,
    read_inputs,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inp() -> Inputs:
    return read_inputs(REPO)


def test_f1_staged_drift(inp: Inputs) -> None:
    t = f1_data(inp).set_index("model")
    assert t.loc["allenai/OLMo-2-0425-1B-SFT", "delta"] == pytest.approx(6.99, abs=0.005)
    assert t.loc["allenai/OLMo-2-0425-1B-DPO", "parent_delta"] == pytest.approx(6.65, abs=0.005)
    assert t.loc["allenai/OLMo-2-0425-1B-Instruct", "parent_delta"] == pytest.approx(0.49, abs=0.005)
    assert t.loc["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "delta"] == pytest.approx(42.95, abs=0.005)
    assert t.loc["Qwen/Qwen3-1.7B", "delta"] == pytest.approx(16.18, abs=0.005)
    assert f1_counts(t.reset_index()) == {"comparisons": 15, "all_corpora_above_zero": 15}


def test_f2_temperature(inp: Inputs) -> None:
    m = f2_means(f2_data(inp))
    assert m.loc["qwen3", "recovered"] == pytest.approx(35, abs=0.5)
    assert m.loc["qwen2.5-math", "recovered"] == pytest.approx(19, abs=0.5)
    assert m.loc["qwen3", "residual"] == pytest.approx(10.4, abs=0.05)
    assert m.recovered.max() < 36


def test_f3_benchmarks(inp: Inputs) -> None:
    t, s = f3_data(inp)
    assert s["r_forgetting"] == pytest.approx(0.887, abs=5e-4)
    assert s["r_ifeval_chat"] == pytest.approx(0.892, abs=5e-4)
    assert (s["n_within"], s["n_chat"]) == (8, 8)
    above = t[t.dbpb_fresh > 15]
    assert len(above) == 3 and not above.within_floor.any()


def test_f4_matched(inp: Inputs) -> None:
    _, ck, s = f4_data(inp)
    assert (s["math_forget_min"], s["math_forget_max"]) == (pytest.approx(4.07, abs=0.005),
                                                            pytest.approx(4.41, abs=0.005))
    assert s["chat_forget_max"] < 0 and s["chat_drift_max"] == pytest.approx(8.97, abs=0.005)
    assert s["r_marks"] == pytest.approx(0.431, abs=5e-4)
    assert s["r_all"] == pytest.approx(0.288, abs=5e-4)
    assert len(ck) == 12 and int(ck.is_mark.sum()) == 6


def test_figures_build(inp: Inputs) -> None:
    with plt.rc_context(RC):
        f3, s3 = f3_data(inp)
        traj, ck, s4 = f4_data(inp)
        figs = [f1_figure(f1_data(inp)), f2_figure(f2_data(inp)), f3_figure(f3, s3), f4_figure(traj, ck, s4)]
    for fig in figs:
        assert fig.get_size_inches()[0] == pytest.approx(16 / 2.54)
        plt.close(fig)

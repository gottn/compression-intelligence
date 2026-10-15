from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cilab.drift import class_drift, drift_row, pct_change
from cilab.io import load_token_bits
from cilab.pairs import CORPORA, FRESH_CORPORA

ROOT = Path(__file__).resolve().parents[1]
EXP03, EXP05 = ROOT / "exp03_staged_drift" / "results", ROOT / "exp05_decomposition" / "results"


def test_class_drift_and_pct_change() -> None:
    base = np.array([1.0, 2.0, 3.0, 4.0])
    post = np.array([2.0, 2.0, 3.0, 2.0])
    codes = np.array([0, 1, 1, 0])
    assert pct_change(10.0, 11.0) == pytest.approx(10.0)
    assert class_drift(base, post, codes, 2).tolist() == pytest.approx([-20.0, 0.0])
    with pytest.raises(ValueError):
        class_drift(base, post[:3], codes, 2)


def test_drift_row_contrasts() -> None:
    base = {c: np.ones(4) for c in ("a", "b", "z")}
    post = {"a": np.array([2.0, 1.0, 1.0, 1.0]), "b": np.array([1.0, 1.0, 1.0, 1.5]), "z": np.ones(4)}
    cls = {c: {"pos": np.array([0, 1, 2, 2]), "freq": np.array([0, 0, 1, 1])} for c in base}
    r = drift_row(base, post, cls, ("a", "b"))
    assert r["dbpb_a"] == pytest.approx(25.0) and r["dbpb_b"] == pytest.approx(12.5) and r["dbpb_z"] == 0.0
    assert r["dbpb_fresh"] == pytest.approx(18.75)
    assert r["drift_knowledge"] == pytest.approx(50.0) and r["drift_style"] == pytest.approx(12.5)
    assert r["contrast_ks"] == pytest.approx(37.5)
    assert r["drift_rare"] == pytest.approx(25.0) and r["drift_frequent"] == pytest.approx(12.5)
    assert r["contrast_rf"] == pytest.approx(12.5)


def test_drift_row_reproduces_exp05_for_olmo2_sft() -> None:
    """exp05's fresh-mean class drifts of OLMo-2 SFT vs base, from the same committed bits and classes."""
    base_id, post_id = "allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-SFT"
    base = {c: load_token_bits(EXP03, base_id, "raw", c) for c in CORPORA}
    post = {c: load_token_bits(EXP03, post_id, "raw", c) for c in CORPORA}
    cls = {}
    for c in CORPORA:
        with np.load(EXP05 / "classes" / f"olmo2__{c}.npz") as z:
            cls[c] = {"pos": z["pos"], "freq": z["freq"]}
    r = drift_row(base, post, cls, FRESH_CORPORA)
    d = pd.read_csv(EXP05 / "decomposition.csv")
    fr = d[(d.model == post_id) & (d.corpus == "fresh_mean")].set_index(["scheme", "class"]).drift_pct
    assert r["dbpb_fresh"] == pytest.approx(fr[("pos", "all")], abs=1e-4)
    for scheme, names in (("pos", ("knowledge", "middle", "style")), ("freq", ("rare", "frequent"))):
        for n in names:
            assert r[f"drift_{n}"] == pytest.approx(fr[(scheme, n)], abs=1e-4)
    assert r["contrast_ks"] == pytest.approx(4.01, abs=0.01)

from __future__ import annotations

import numpy as np
import pandas as pd

from cilab.stages import Stage
from exp04_benchmarks.analyze_forgetting import (
    GENERAL,
    checkpoint_table,
    exp01_deltas,
    gate_verdict,
    general_scores,
    point_label,
    predictor_table,
    score_row,
)

TOY = (
    Stage("o/base", "f", "base", "pretrain", None, "t"),
    Stage("o/sft", "f", "sft", "sft", "o/base", "t"),
    Stage("o/dpo", "f", "dpo", "dpo", "o/sft", "t"),
    Stage("x/base", "g", "base", "pretrain", None, "u"),
    Stage("x/inst", "g", "instruct", "sft", "x/base", "u"),
)


def summary(model: str, acc: float, gsm: float, ok: bool = True) -> dict:
    tasks = {t: {"metrics": {"acc,none": acc - 0.01, key: acc}} for t, key, _ in GENERAL}  # primary wins
    tasks_g = {"gsm8k": {"metrics": {"exact_match,flexible-extract": gsm, "exact_match,strict-match": gsm - 0.1}},
               "ifeval": {"metrics": {"prompt_level_strict_acc,none": 0.2, "inst_level_strict_acc,none": 0.3}}}
    return {"model": model, "gpu": "T4", "groups": {
        "loglik": {"status": "ok", "wall_s": 100.0, "tasks": tasks},
        "gsm8k": {"status": "ok" if ok else "failed", "wall_s": 50.0, "tasks": tasks_g if ok else {}}}}


def test_score_row_reads_primary_metric_and_flags_failures() -> None:
    r = score_row(summary("o/sft", 0.6, 0.3, ok=False))
    assert r["arc_easy"] == 0.6 and r["winogrande"] == 0.6  # winogrande primary is acc
    assert np.isnan(r["gsm8k_flex"]) and r["failed_groups"] == "gsm8k" and r["eval_wall_s"] == 150.0


def test_general_scores_mean_and_chance_adjustment() -> None:
    df = general_scores(pd.DataFrame([score_row(summary("m", 0.5, 0.1))]))
    assert np.isclose(df.general.iloc[0], 0.5)
    # (0.5-0.25)/0.75 for four tasks, (0.5-0.5)/0.5 = 0 for piqa and winogrande
    assert np.isclose(df.general_chance.iloc[0], (4 * (1 / 3) + 0) / 6)


def test_exp01_deltas_use_total_bits_and_fresh_mean() -> None:
    rows = []
    for m, scale in (("b", 1.0), ("p", 1.1)):
        for c, bits in (("wiki2026", 100.0), ("arxiv2026", 200.0), ("gutenberg2026", 300.0), ("enwik8_ctrl", 50.0)):
            extra = 1.3 if (m == "p" and c == "arxiv2026") else 1.0
            rows.append({"model": m, "condition": "raw", "corpus": c, "bytes": 10, "tokens": 5,
                         "total_bits": bits * scale * extra})
    rows.append({"model": "p", "condition": "chat", "corpus": "wiki2026", "bytes": 10, "tokens": 5, "total_bits": 1e9})
    d = exp01_deltas(pd.DataFrame(rows), [("p", "b")]).iloc[0]
    assert np.isclose(d.dbpb_fresh, np.mean([10.0, 43.0, 10.0])) and np.isclose(d.dbpb_enwik8, 10.0)


def test_predictor_table_prefers_exp03_and_fills_from_exp01() -> None:
    sd = pd.DataFrame([
        {"model": "o/sft", "reference": "o/base", "kind": "vs_base", "corpus": "fresh_mean", "delta_pct": 7.0},
        {"model": "o/sft", "reference": "o/base", "kind": "vs_base", "corpus": "enwik8_ctrl", "delta_pct": 9.0},
        {"model": "o/dpo", "reference": "o/base", "kind": "vs_base", "corpus": "fresh_mean", "delta_pct": 14.0},
        {"model": "o/dpo", "reference": "o/base", "kind": "vs_base", "corpus": "enwik8_ctrl", "delta_pct": 19.0},
        {"model": "o/dpo", "reference": "o/sft", "kind": "vs_parent", "corpus": "fresh_mean", "delta_pct": 6.6},
    ])
    e1 = pd.DataFrame([{"model": m, "condition": "raw", "corpus": c, "bytes": 1, "tokens": 1, "total_bits": b}
                       for m, b in (("x/base", 100.0), ("x/inst", 104.0))
                       for c in ("wiki2026", "arxiv2026", "gutenberg2026", "enwik8_ctrl")])
    p = predictor_table(sd, e1, TOY)
    assert p.model.tolist() == ["o/sft", "o/dpo", "x/inst"]
    assert p.dbpb_fresh.tolist() == [7.0, 14.0, 4.0] and p.dbpb_enwik8.tolist()[:2] == [9.0, 19.0]
    assert p.bpb_source.tolist()[2].startswith("exp01")


def test_checkpoint_table_forgetting_and_gain_vs_family_base() -> None:
    scores = pd.DataFrame([score_row(summary(m, a, g)) for m, a, g in
                           (("o/base", 0.5, 0.2), ("o/sft", 0.45, 0.3), ("o/dpo", 0.4, 0.25),
                            ("x/base", 0.6, 0.4), ("x/inst", 0.6, 0.5))])
    preds = pd.DataFrame({"model": ["o/sft", "o/dpo", "x/inst"], "dbpb_fresh": [7.0, 14.0, 4.0],
                          "dbpb_enwik8": [9.0, 19.0, 5.0], "bpb_source": ["a", "a", "b"]})
    t = checkpoint_table(scores, preds, TOY).set_index("model")
    assert np.isclose(t.loc["o/sft", "forgetting"], 10.0) and np.isclose(t.loc["o/dpo", "forgetting"], 20.0)
    assert np.isclose(t.loc["o/dpo", "gain_gsm8k"], 5.0)  # vs the family base, not the SFT parent
    assert np.isclose(t.loc["x/inst", "forgetting"], 0.0) and np.isclose(t.loc["x/inst", "gain_gsm8k"], 10.0)
    assert np.isnan(t.loc["o/base", "forgetting"])


def test_gate_verdict_thresholds_and_labels() -> None:
    assert [gate_verdict(r) for r in (0.85, 0.8, 0.7, 0.59, float("nan"))] == \
        ["pass", "pass", "soft pass", "fail", "undefined"]
    st = {s.model: s for s in TOY}
    assert point_label(st["o/sft"]) == "SFT"
    assert point_label(Stage("Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b", "instruct", "", "q", "q")) == "Instruct 0.5B"
    assert point_label(Stage("Qwen/Qwen3-1.7B", "qwen3", "instruct", "", "q", "q")) == "Qwen3 Instruct"

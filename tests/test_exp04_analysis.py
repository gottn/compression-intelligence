from __future__ import annotations

import numpy as np
import pandas as pd

from cilab.stages import Stage
from exp04_benchmarks.analyze_forgetting import (
    ALL12,
    GENERAL,
    NO_MATH,
    R1_PARENT,
    bpb_cost,
    checkpoint_table,
    exp01_deltas,
    gate_verdict,
    general_scores,
    nudge_labels,
    outcome_fields,
    parent_reference_row,
    point_label,
    predictor_table,
    score_row,
    subsets,
    threshold_reading,
)

TOY = (
    Stage("o/base", "f", "base", "pretrain", None, "t"),
    Stage("o/sft", "f", "sft", "sft", "o/base", "t"),
    Stage("o/dpo", "f", "dpo", "dpo", "o/sft", "t"),
    Stage("x/base", "g", "base", "pretrain", None, "u"),
    Stage("x/inst", "g", "instruct", "sft", "x/base", "u"),
)


def summary(model: str, acc: float, gsm: float, ok: bool = True) -> dict:
    tasks = {t: {"metrics": {"acc,none": acc - 0.01, key: acc, key.replace(",", "_stderr,"): 0.01}}
             for t, key, _ in GENERAL}  # primary wins
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


def test_bpb_cost_sums_fresh_raw_rows_only() -> None:
    rows = [{"model": "m", "condition": cond, "corpus": c, "wall_s": w}
            for cond, c, w in (("raw", "wiki2026", 2.0), ("raw", "arxiv2026", 3.0), ("raw", "gutenberg2026", 4.0),
                               ("raw", "enwik8_ctrl", 50.0), ("chat", "wiki2026", 70.0))]
    c = bpb_cost(pd.DataFrame(rows), "T4").iloc[0]
    assert (c.model, c.bpb_fresh_wall_s, c.bpb_hardware) == ("m", 9.0, "T4")


def test_outcome_fields_noise_floor() -> None:
    ref = {"general": 0.5, "general_se": 0.004, "general_chance": 0.3, "gsm8k_flex": 0.3, "gsm8k_strict": 0.2,
           "ifeval_prompt_strict": 0.1}
    post = {**ref, "general": 0.49, "general_se": 0.003, "gsm8k_flex": 0.35}
    f = outcome_fields(post, ref)
    assert np.isclose(f["forgetting"], 2.0) and np.isclose(f["gain_gsm8k"], 5.0)
    assert np.isclose(f["floor_pct"], 100 * 1.96 * 0.005 / 0.5) and f["within_floor"] is False  # 2.0% > 1.96%
    assert outcome_fields({**post, "general": 0.495}, ref)["within_floor"] is True


def test_threshold_reading_contiguous_largest_inside_and_exceptions() -> None:
    post = pd.DataFrame({"model": ["a/1", "a/2", "a/3", "a/4", "a/5"], "dbpb_fresh": [1.0, 3.6, 15.0, 24.0, 43.0],
                         "forgetting": [0.5, 3.8, -1.5, 14.0, 22.0], "floor_pct": [2.0, 2.1, 2.0, 2.2, 2.4],
                         "within_floor": [True, False, True, False, False]})
    r = threshold_reading(post.iloc[::-1])
    assert (r["contiguous"], r["n_contiguous"], r["largest_inside"], r["n_inside"]) == (1.0, 1, 15.0, 2)
    assert [a[0] for a in r["above"]] == ["4", "5"] and [a[0] for a in r["exceptions"]] == ["2"]
    assert r["max_abs_forgetting_inside"] == 1.5 and r["n"] == 5


def test_subsets_and_parent_reference_row() -> None:
    stages = (*TOY, Stage("x/math", "g", "math-cpt", "cpt", "x/base", "u"),
              Stage("x/r1", "g", "r1-distill", "sft", "x/math", "u"))
    scores = pd.DataFrame([score_row(summary(m, a, g)) for m, a, g in
                           (("o/base", 0.5, 0.2), ("o/sft", 0.45, 0.3), ("o/dpo", 0.4, 0.25), ("x/base", 0.6, 0.4),
                            ("x/inst", 0.6, 0.5), ("x/math", 0.5, 0.6), ("x/r1", 0.45, 0.5))])
    preds = pd.DataFrame({"model": ["o/sft", "o/dpo", "x/inst", "x/math", "x/r1"],
                          "dbpb_fresh": [7.0, 14.0, 4.0, 24.0, 43.0], "dbpb_enwik8": [9.0, 19.0, 5.0, 58.0, 74.0],
                          "bpb_source": ["a"] * 5})
    table = checkpoint_table(scores, preds, stages)
    sd = pd.DataFrame([{"model": "x/r1", "reference": "x/math", "kind": "vs_parent", "corpus": c, "delta_pct": v}
                       for c, v in (("fresh_mean", 15.8), ("enwik8_ctrl", 9.8))])
    r1p = parent_reference_row(table, sd, "x/r1", stages)
    assert (r1p["reference"], r1p["dbpb_fresh"]) == ("x/math", 15.8)
    assert np.isclose(r1p["forgetting"], 10.0) and np.isclose(r1p["gain_gsm8k"], -10.0)  # vs Math, not vs base
    sets = subsets(table, r1p)
    assert len(sets[ALL12]) == 5 and set(sets[NO_MATH].model) == {"o/sft", "o/dpo", "x/inst"}
    swapped = sets[R1_PARENT].set_index("model")
    assert len(swapped) == 5 and swapped.loc["x/r1", "dbpb_fresh"] == 15.8


def test_nudge_labels_separates_overlapping_boxes_only() -> None:
    offs = nudge_labels([(0.0, 0.0), (2.0, 1.0), (300.0, 0.0)], [50.0, 50.0, 50.0], height=9.0)
    assert offs[0] == (5.0, 4.0) and offs[2] == (5.0, 4.0)  # far-apart labels stay put
    assert offs[1][1] - offs[0][1] >= 9.0 - 1.0  # the close one moves up by at least a line
    # a marker sitting where the label would go pushes it up
    lone = nudge_labels([(0.0, 0.0), (20.0, 8.0)], [50.0, 50.0], height=9.0, marker=4.5)
    assert lone[0][1] > 4.0


def test_chat_condition_scores_and_gain_use_the_base_prompted_reference() -> None:
    chat = {"model": "o/sft", "groups": {"gsm8k_chat": {"status": "ok", "wall_s": 30.0, "tasks": {
        "gsm8k": {"metrics": {"exact_match,flexible-extract": 0.5, "exact_match,strict-match": 0.45}}}},
        "ifeval_chat": {"status": "failed", "wall_s": 5.0}}}
    r = score_row(summary("o/sft", 0.45, 0.3), chat)
    assert (r["gsm8k_flex_chat"], r["gsm8k_strict_chat"], r["chat_wall_s"]) == (0.5, 0.45, 35.0)
    assert np.isnan(r["ifeval_prompt_strict_chat"]) and r["failed_groups"] == "ifeval_chat"
    assert np.isnan(score_row(summary("o/b", 0.5, 0.2))["gsm8k_flex_chat"])
    scores = pd.DataFrame([score_row(summary("o/base", 0.5, 0.2)), r])
    preds = pd.DataFrame({"model": ["o/sft"], "dbpb_fresh": [7.0], "dbpb_enwik8": [9.0], "bpb_source": ["a"]})
    t = checkpoint_table(scores, preds, TOY[:2]).set_index("model")
    assert np.isclose(t.loc["o/sft", "gain_gsm8k_chat"], 30.0)  # 0.5 chat vs the base's 0.2 base-prompted
    assert np.isclose(t.loc["o/sft", "gain_gsm8k"], 10.0)

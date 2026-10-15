from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cilab.sft import checkpoint_line
from exp06_matched_drift.analyze_matched import (
    MARK_SET,
    checkpoint_table,
    correlation_table,
    eval_scores,
    h3_rows,
    per_mark,
    residual_corr,
    rule_table,
    strictly_descending,
)

TASKS = (("arc_easy", "acc_norm"), ("arc_challenge", "acc_norm"), ("hellaswag", "acc_norm"), ("piqa", "acc_norm"),
         ("winogrande", "acc"), ("mmlu", "acc"))
BASE = {"general": 0.6, "general_se": 0.005, "mmlu": 0.4, "arc": 0.6, "ifeval_plain": 0.1}
BASE_CHAT = {"ifeval": 0.12, "gsm8k": 0.2}


def summary(label: str, acc: float, ifeval: float, gsm8k: float = 0.3) -> dict[str, object]:
    ll = {t: {"metrics": {f"{m},none": acc, f"{m}_stderr,none": 0.01}} for t, m in TASKS}
    return {"model": f"/r/{label}", "gpu": "A10", "groups": {
        "loglik": {"status": "ok", "wall_s": 100.0, "tasks": ll},
        "gsm8k_chat": {"status": "ok", "wall_s": 50.0,
                       "tasks": {"gsm8k": {"metrics": {"exact_match,flexible-extract": gsm8k}}}},
        "ifeval_chat": {"status": "ok", "wall_s": 40.0,
                        "tasks": {"ifeval": {"metrics": {"prompt_level_strict_acc,none": ifeval}}}}}}


def fixture() -> tuple[list[str], pd.DataFrame, dict[str, dict[str, object]]]:
    """3 recipes × marks 3/6/10 (+ final); forgetting rises with the contrast, gain does not."""
    contrast = {"tulu": 0.0, "smoltalk": 3.0, "mathcot": 8.0}
    drift_rows, lines, evals = [], [], {}
    for rec, c0 in contrast.items():
        for k, (step, d, tags) in enumerate(((100, 3.1, ["3"]), (300, 6.2, ["6"]), (700, 10.3, ["10"]),
                                             (900, 11.0, ["final"]))):
            c = c0 * (1 + 0.1 * k)
            drift_rows.append({"recipe": rec, "step": step, "token_positions": step * 65536, "nonpad_tokens": 1,
                               "loss_tokens": 1, "epochs": step / 700, "dbpb_fresh": d, "dbpb_enwik8_ctrl": d,
                               "drift_knowledge": d + c / 2, "drift_middle": d, "drift_style": d - c / 2,
                               "drift_rare": d + c / 3, "drift_frequent": d - c / 3, "contrast_ks": c,
                               "contrast_rf": 2 * c / 3, "train_loss": 1.0})
            label = f"exp06__{rec}__step{step:05d}"
            lines.append(checkpoint_line(label, f"/r/{label}", rec, step, tags))
            evals[label] = summary(label, 0.6 * (1 - c / 100), 0.3 + 0.01 * ((k + len(rec)) % 3))
    return lines, pd.DataFrame(drift_rows), evals


def test_eval_scores_split_main_and_chat_groups() -> None:
    r = eval_scores(summary("x", 0.5, 0.25, 0.4))
    assert r["arc_easy"] == 0.5 and r["mmlu"] == 0.5
    assert r["ifeval_prompt_strict_chat"] == 0.25 and r["gsm8k_flex_chat"] == 0.4
    assert np.isnan(r["ifeval_prompt_strict"]) and r["failed_groups"] == ""


def test_checkpoint_table_and_tests() -> None:
    lines, drift, evals = fixture()
    t = checkpoint_table(lines, drift, evals, BASE, BASE_CHAT)
    assert list(t.recipe.unique()) == ["tulu", "smoltalk", "mathcot"]
    assert t.is_mark.sum() == 9 and t.is_final.sum() == 3
    m = t[(t.recipe == "mathcot") & (t.step == 300)].iloc[0]
    assert m.forgetting == pytest.approx(100 * (0.6 - 0.6 * (1 - 8.8 / 100)) / 0.6)
    assert m.gain_ifeval == pytest.approx(100 * (m.ifeval_prompt_strict_chat - 0.12))
    assert len(per_mark(t)) == 9
    corr = correlation_table(t)
    tests = rule_table(t, corr, h3_rows(t, BASE), "contrast_ks").set_index("id")
    assert tests.loc["H1", "value"] == pytest.approx(1.0) and tests.loc["H1", "verdict"] == "pass"
    assert tests.loc["H4", "verdict"] == "pass"
    assert corr[(corr.x == "contrast_ks") & (corr.y == "forgetting") & (corr.subset == MARK_SET)].n.iloc[0] == 9
    assert residual_corr(t[t.is_mark], "contrast_ks", "forgetting", "dbpb_fresh") > 0.9


def test_strictly_descending() -> None:
    order = ("mathcot", "smoltalk", "tulu")
    assert strictly_descending({"mathcot": 3, "smoltalk": 2, "tulu": 1}, order) is True
    assert strictly_descending({"mathcot": 3, "smoltalk": 3, "tulu": 1}, order) is False
    assert strictly_descending({"mathcot": 3, "tulu": 1}, order) is None

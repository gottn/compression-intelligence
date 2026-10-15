"""Benchmark scores from exp04's run_eval.py summaries (exp04, exp06). Pure.

general = mean of 6 primary accuracies (acc_norm where reported, else acc); general_chance rescales each by its
chance accuracy; general_se is the standard error of the 6-task mean from the per-task standard errors.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

GENERAL: tuple[tuple[str, str, float], ...] = (  # (task, primary metric key, chance accuracy)
    ("arc_easy", "acc_norm,none", 0.25), ("arc_challenge", "acc_norm,none", 0.25),
    ("hellaswag", "acc_norm,none", 0.25), ("piqa", "acc_norm,none", 0.5),
    ("winogrande", "acc,none", 0.5), ("mmlu", "acc,none", 0.25),
)
TARGET: tuple[tuple[str, str, str], ...] = (  # (column, task, metric key)
    ("gsm8k_flex", "gsm8k", "exact_match,flexible-extract"), ("gsm8k_strict", "gsm8k", "exact_match,strict-match"),
    ("ifeval_prompt_strict", "ifeval", "prompt_level_strict_acc,none"),
    ("ifeval_inst_strict", "ifeval", "inst_level_strict_acc,none"),
)


def score_row(summary: Mapping[str, object], chat: Mapping[str, object] | None = None) -> dict[str, object]:
    """Flat per-model scores from a run_eval.py summary.json: primary accuracy per general task, target
    metrics, the GPU, total wall time and the groups that did not finish; with the chat-condition summary
    (summary_chat.json), the same target metrics as *_chat columns."""
    tasks, wall, bad = {}, 0.0, []
    for g, rec in summary["groups"].items():
        wall += float(rec.get("wall_s", 0.0))
        if rec.get("status") != "ok":
            bad.append(g)
        tasks.update(rec.get("tasks", {}))
    row: dict[str, object] = {"model": summary["model"], "gpu": summary.get("gpu", ""), "eval_wall_s": wall,
                              "failed_groups": ",".join(bad)}
    for task, key, _ in GENERAL:
        m = tasks.get(task, {}).get("metrics", {})
        row[task] = m.get(key, np.nan)
        row[f"{task}_se"] = m.get(key.replace(",", "_stderr,", 1), np.nan)
    for col, task, key in TARGET:
        row[col] = tasks.get(task, {}).get("metrics", {}).get(key, np.nan)
    ctasks, cwall = {}, 0.0
    for g, rec in (chat or {}).get("groups", {}).items():
        cwall += float(rec.get("wall_s", 0.0))
        if rec.get("status") != "ok":
            bad.append(g)
        ctasks.update(rec.get("tasks", {}))
    for col, task, key in TARGET:
        row[f"{col}_chat"] = ctasks.get(task, {}).get("metrics", {}).get(key, np.nan)
    row["chat_wall_s"], row["failed_groups"] = cwall if chat else np.nan, ",".join(bad)
    return row


def general_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Adds `general` (mean primary accuracy) and `general_chance` (mean of (acc − c) / (1 − c))."""
    out = scores.copy()
    out["general"] = out[[t for t, _, _ in GENERAL]].mean(axis=1, skipna=False)
    adj = [(out[t] - c) / (1 - c) for t, _, c in GENERAL]
    out["general_chance"] = pd.concat(adj, axis=1).mean(axis=1, skipna=False)
    se = out[[f"{t}_se" for t, _, _ in GENERAL]]
    out["general_se"] = np.sqrt((se**2).sum(axis=1, skipna=False)) / len(GENERAL)
    return out

from __future__ import annotations

from pathlib import Path

from exp04_benchmarks.run_eval import (
    GROUPS,
    MMLU_SUBJECTS,
    lm_eval_argv,
    newest_results,
    samples_spec,
    summarize_results,
    task_lines,
)


def test_samples_spec_reproduces_per_task_limits() -> None:
    spec = samples_spec()
    assert len(MMLU_SUBJECTS) == len(set(MMLU_SUBJECTS)) == 57
    assert spec["hellaswag"] == list(range(2000))
    assert len(spec) == 58 and all(spec[f"mmlu_{s}"] == list(range(40)) for s in MMLU_SUBJECTS)
    assert sum(len(v) for v in spec.values()) == 2000 + 57 * 40


def test_lm_eval_argv_is_identical_across_models_except_the_id() -> None:
    loglik, gsm8k, ifeval = GROUPS
    a = lm_eval_argv("py", "o/base", gsm8k, "/out/g", None)
    b = lm_eval_argv("py", "o/post", gsm8k, "/out/g", None)
    assert [x for x in a if "o/base" not in x] == [x for x in b if "o/post" not in x]
    assert "--apply_chat_template" not in a and "--log_samples" not in a and "--num_fewshot" not in a
    assert a[a.index("--limit") + 1] == "300" and a[a.index("--tasks") + 1] == "gsm8k"
    assert "pretrained=o/base,dtype=float16,max_length=2048" in a
    assert a[a.index("--batch_size") + 1] == "auto" and a[a.index("--seed") + 1] == "0,1234,1234,1234"
    ll = lm_eval_argv("py", "o/base", loglik, "/out/l", "/out/s.json")
    assert "--limit" not in ll and ll[ll.index("--samples") + 1] == "/out/s.json"
    assert ll[ll.index("--tasks") + 1] == "arc_easy,arc_challenge,hellaswag,piqa,winogrande,mmlu"
    assert lm_eval_argv("py", "m", ifeval, "/o", None)[-1] == "200"


RES = {
    "results": {
        "arc_easy": {"alias": "arc_easy", "acc,none": 0.7, "acc_stderr,none": 0.01, "acc_norm,none": 0.65,
                     "acc_norm_stderr,none": 0.01, "sample_len": 2376},
        "mmlu_a": {"acc,none": 0.5}, "mmlu_b": {"acc,none": 0.6},
    },
    "groups": {"mmlu": {"alias": "mmlu", "acc,none": 0.55, "acc_stderr,none": 0.02}},
    "group_subtasks": {"mmlu": ["mmlu_stem"], "mmlu_stem": ["mmlu_a", "mmlu_b"], "arc_easy": []},
    "versions": {"arc_easy": 1.0, "mmlu": 2, "mmlu_a": 1.0, "mmlu_b": 1.0},
    "n-shot": {"arc_easy": 0, "mmlu_a": 0, "mmlu_b": 0},
    "n-samples": {"arc_easy": {"original": 2376, "effective": 2376}, "mmlu_a": {"original": 100, "effective": 40},
                  "mmlu_b": {"original": 120, "effective": 40}},
}


def test_summarize_results_handles_tasks_and_groups() -> None:
    s = summarize_results(RES, ["arc_easy", "mmlu", "hellaswag"])
    assert set(s) == {"arc_easy", "mmlu"}  # missing tasks are left out, not invented
    assert s["arc_easy"]["metrics"] == {"acc,none": 0.7, "acc_stderr,none": 0.01, "acc_norm,none": 0.65,
                                        "acc_norm_stderr,none": 0.01}
    assert (s["arc_easy"]["n_effective"], s["arc_easy"]["n_shot"], s["arc_easy"]["n_leaves"]) == (2376, [0], 1)
    m = s["mmlu"]
    assert m["metrics"]["acc,none"] == 0.55 and m["version"] == 2 and m["leaf_versions"] == ["1.0"]
    assert (m["n_effective"], m["n_original"], m["n_leaves"]) == (80, 220, 2)
    assert m["leaf_effective"] == {"mmlu_a": 40, "mmlu_b": 40}


def test_task_lines_print_headline_metric_with_stderr() -> None:
    lines = task_lines("o/m", "loglik", summarize_results(RES, ["arc_easy"]), 61.2)
    assert lines == ["SCORE o/m arc_easy: acc_norm[none]=0.6500±0.0100 n=2376 shot=[0] (loglik group, 61 s)"]


def test_newest_results_ignores_files_from_before_the_call() -> None:
    old, new, newer = Path("a/results_1.json"), Path("a/results_2.json"), Path("b/results_3.json")
    mt = {old: 10.0, new: 100.0, newer: 101.0}
    assert newest_results([old, new, newer], 50.0, mt) == newer
    assert newest_results([old], 50.0, mt) is None

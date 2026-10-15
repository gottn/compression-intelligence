from __future__ import annotations

import pytest

from exp04_benchmarks.run_eval import (
    GROUPS,
    MMLU_SUBJECTS,
    TASK_DIRS,
    eval_kwargs,
    model_kwargs,
    neutral_generation_config,
    samples_spec,
    summarize_results,
    summary_name,
    task_lines,
    thinking_switch,
)


def test_samples_spec_reproduces_per_task_limits() -> None:
    spec = samples_spec()
    assert len(MMLU_SUBJECTS) == len(set(MMLU_SUBJECTS)) == 57
    assert spec["hellaswag"] == list(range(2000))
    assert len(spec) == 58 and all(spec[f"mmlu_{s}"] == list(range(40)) for s in MMLU_SUBJECTS)
    assert sum(len(v) for v in spec.values()) == 2000 + 57 * 40


def test_model_and_eval_kwargs_are_identical_across_models_except_the_id() -> None:
    a, b = model_kwargs("o/base"), model_kwargs("o/post")
    assert {k: v for k, v in a.items() if k != "pretrained"} == {k: v for k, v in b.items() if k != "pretrained"}
    assert a == {"pretrained": "o/base", "dtype": "float16", "max_length": 2048, "batch_size": "auto",
                 "device": "cuda"}
    loglik, gsm8k, ifeval, gsm8k_chat, ifeval_chat = GROUPS
    g = eval_kwargs(gsm8k, None)
    assert g["tasks"] == ["gsm8k"] and g["limit"] == 300 and "samples" not in g and "num_fewshot" not in g
    assert g["apply_chat_template"] is False and g["log_samples"] is False
    assert (g["random_seed"], g["numpy_random_seed"], g["torch_random_seed"], g["fewshot_random_seed"]) == \
        (0, 1234, 1234, 1234)
    ll = eval_kwargs(loglik, samples_spec())
    assert "limit" not in ll and len(ll["samples"]) == 58
    assert ll["tasks"] == ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "mmlu"]
    assert eval_kwargs(ifeval, None)["limit"] == 200
    gc = eval_kwargs(gsm8k_chat, None)
    assert gc["apply_chat_template"] is True and gc["fewshot_as_multiturn"] is True
    assert {k: v for k, v in gc.items() if k not in ("apply_chat_template", "fewshot_as_multiturn")} == \
        {k: v for k, v in g.items() if k not in ("apply_chat_template", "fewshot_as_multiturn")}
    assert eval_kwargs(ifeval_chat, None)["limit"] == 200 and g["fewshot_as_multiturn"] is False


def test_chat_helpers() -> None:
    assert thinking_switch("{% if enable_thinking is defined and enable_thinking is false %}") == {"enable_thinking": False}
    assert thinking_switch("{{ messages }}") == {} and thinking_switch(None) == {}
    raw, chat = [g for g in GROUPS if not g.chat], [g for g in GROUPS if g.chat]
    assert summary_name(raw) == "summary.json" and summary_name(chat) == "summary_chat.json"


def test_task_dirs_name_one_directory_per_task_family() -> None:
    # arc_easy and arc_challenge share arc/; the default 0-shot mmlu lives in mmlu/default/.
    family = {"arc_easy": "arc", "arc_challenge": "arc", "mmlu": "mmlu/default"}
    tasks = [t for g in GROUPS for t in g.tasks]
    assert sorted({family.get(t, t) for t in tasks}) == sorted(TASK_DIRS)


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


def test_neutral_generation_config_keeps_only_token_ids() -> None:
    qwen_instruct = {"bos_token_id": 151643, "eos_token_id": [151645, 151643], "pad_token_id": 151643,
                     "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05,
                     "max_new_tokens": 2048, "transformers_version": "4.37.0"}
    kept, dropped = neutral_generation_config(qwen_instruct)
    assert kept == {"bos_token_id": 151643, "eos_token_id": [151645, 151643], "pad_token_id": 151643}
    assert list(dropped) == ["do_sample", "max_new_tokens", "repetition_penalty", "temperature", "top_k", "top_p"]
    assert neutral_generation_config({"eos_token_id": 2, "pad_token_id": None}) == ({"eos_token_id": 2}, {})


def test_run_target_keeps_exp04_indices_and_adds_model_paths() -> None:
    from cilab.stages import model_at
    from exp04_benchmarks.run_eval import run_target
    assert run_target(4, None, None) == (model_at(4), "Qwen__Qwen2.5-0.5B")
    assert run_target(None, "/r/tulu/ckpt/step00340/", "exp06__tulu__step00340") == \
        ("/r/tulu/ckpt/step00340/", "exp06__tulu__step00340")
    assert run_target(None, "allenai/OLMo-2-0425-1B", None) == ("allenai/OLMo-2-0425-1B", "allenai__OLMo-2-0425-1B")
    for bad in ((None, None, None), (1, "/p", None), (1, None, "lbl")):
        with pytest.raises(ValueError):
            run_target(*bad)
    assert model_kwargs("o/base", None) == model_kwargs("o/base")
    assert model_kwargs("o/base", "o/inst") == {**model_kwargs("o/base"), "tokenizer": "o/inst"}

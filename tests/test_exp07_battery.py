from __future__ import annotations

from pathlib import Path

import pytest

from cilab.lmeval import MMLU_SUBJECTS
from cilab.sft import checkpoint_line
from exp07_arc_format.run_battery import (
    CHAT_GROUP,
    GROUPS,
    PUBLIC,
    TASK_DIRS,
    attempt_suffix,
    bits,
    eval_kwargs,
    group_named,
    group_samples,
    item_outcomes,
    list_lines,
    parse_entry,
    plan,
    public_lines,
    reference_chat_folder,
    template_partners,
    truncation_counts,
)

HERE = Path(__file__).resolve().parents[1] / "exp07_arc_format"


def test_public_list_is_the_committed_file_and_the_18_stage_rows() -> None:
    lines = public_lines()
    assert (HERE / "checkpoints_public.tsv").read_text() == "".join(lines)
    assert len(lines) == 18 and all(ln.split("\t")[2] == PUBLIC for ln in lines)
    assert lines[0].split("\t")[:2] == ["allenai__OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B"]


def test_group_samples_reuse_exp04_selection_under_both_mmlu_formats() -> None:
    letter, cont = group_samples(group_named("b4_mmlu_letter")), group_samples(group_named("b4_mmlu_cont"))
    assert len(letter) == len(cont) == 57 and all(v == list(range(40)) for v in letter.values())
    assert sorted(cont) == sorted(f"mmlu_{s}_continuation" for s in MMLU_SUBJECTS)
    assert [letter[f"mmlu_{s}"] for s in MMLU_SUBJECTS] == [cont[f"mmlu_{s}_continuation"] for s in MMLU_SUBJECTS]
    assert group_samples(group_named("b5_hellaswag")) == {"hellaswag": list(range(2000))}
    assert group_samples(group_named("b1_arc_0shot")) is None
    assert all(len(v) == 3 for v in group_samples(group_named("b4_mmlu_cont"), limit=3).values())


def test_eval_kwargs_differ_only_in_shots_template_and_selection() -> None:
    b1, b2, b3 = (eval_kwargs(group_named(n)) for n in ("b1_arc_0shot", "b2_arc_25shot", CHAT_GROUP))
    assert b1["tasks"] == b2["tasks"] == b3["tasks"] == ["arc_easy", "arc_challenge"]
    assert "num_fewshot" not in b1 and b2["num_fewshot"] == 25 and "num_fewshot" not in b3
    assert b1["apply_chat_template"] is False and b2["apply_chat_template"] is False and b3["apply_chat_template"] is True
    assert all(k["log_samples"] is True and "samples" not in k and "limit" not in k for k in (b1, b2, b3))
    assert (b1["random_seed"], b1["numpy_random_seed"], b1["torch_random_seed"], b1["fewshot_random_seed"]) == \
        (0, 1234, 1234, 1234)
    strip = ("num_fewshot", "apply_chat_template", "fewshot_as_multiturn")
    assert {k: v for k, v in b1.items() if k not in strip} == {k: v for k, v in b2.items() if k not in strip} == \
        {k: v for k, v in b3.items() if k not in strip}
    hs = eval_kwargs(group_named("b5_hellaswag"))
    assert "limit" not in hs and hs["samples"] == {"hellaswag": list(range(2000))}
    dry = eval_kwargs(group_named("b1_arc_0shot"), limit=10)
    assert dry["limit"] == 10 and eval_kwargs(group_named("b5_hellaswag"), limit=10)["samples"]["hellaswag"] == \
        list(range(10))


def test_task_dirs_cover_every_battery_task() -> None:
    fam = {"arc_easy": "arc", "arc_challenge": "arc", "mmlu": "mmlu/default", "mmlu_continuation": "mmlu/continuation"}
    assert sorted({fam.get(t, t) for g in GROUPS for t in g.tasks}) == sorted(TASK_DIRS)


def test_bases_run_b3_with_every_post_trained_template_of_their_family() -> None:
    assert template_partners("Qwen/Qwen2.5-1.5B") == ["Qwen/Qwen2.5-1.5B-Instruct", "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo",
                                                      "Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]
    base = parse_entry(public_lines()[7])
    assert base.is_base and base.model == "Qwen/Qwen2.5-1.5B"
    runs = plan(base)
    folders = [r.folder for r in runs]
    assert folders[:4] == ["b1_arc_0shot", "b4_mmlu_letter", "b4_mmlu_cont", "b5_hellaswag"]
    assert folders[-1] == "b2_arc_25shot" and "b0_piqa_wino" not in folders
    chat = [r for r in runs if r.group.chat]
    assert [r.tokenizer for r in chat] == template_partners("Qwen/Qwen2.5-1.5B")
    assert chat[3].folder == f"{CHAT_GROUP}__deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B"
    total = sum(len(plan(parse_entry(ln))) for ln in public_lines())
    assert total == 12 * 6 + 6 * 5 + 12  # 6 groups each; a base runs B3 once per post-trained model of its family


def test_post_trained_and_retrained_entries_use_their_own_template_against_the_matched_base_run() -> None:
    post = parse_entry(public_lines()[15])
    assert post.model == "HuggingFaceTB/SmolLM2-1.7B-Instruct" and not post.is_base
    assert post.base == "HuggingFaceTB/SmolLM2-1.7B"
    assert [r.tokenizer for r in plan(post)] == [None] * 6
    assert reference_chat_folder(post) == f"{CHAT_GROUP}__HuggingFaceTB__SmolLM2-1.7B-Instruct"
    line = checkpoint_line("exp07__mathcot__step01500", "/r/mathcot/ckpt/step01500", "mathcot", 1500, ["final"])
    rt = parse_entry(line)
    assert (rt.label, rt.base, rt.recipe, rt.step, rt.is_base) == \
        ("exp07__mathcot__step01500", "allenai/OLMo-2-0425-1B", "mathcot", 1500, False)
    assert [r.folder for r in plan(rt)] == ["b1_arc_0shot", "b4_mmlu_letter", "b4_mmlu_cont", "b5_hellaswag",
                                            "b0_piqa_wino", CHAT_GROUP, "b2_arc_25shot"]
    assert reference_chat_folder(rt) == f"{CHAT_GROUP}__allenai__OLMo-2-0425-1B-Instruct"
    again = parse_entry(line.replace("/r/mathcot/", "/r/mathcot-attempt2/"))
    assert again.label == "exp07__mathcot__step01500__attempt2" and attempt_suffix("/r/tulu/ckpt/x") == ""
    with pytest.raises(ValueError):
        parse_entry(checkpoint_line("x", "/p", "other", 0, []))
    with pytest.raises(ValueError):
        reference_chat_folder(parse_entry(public_lines()[0]))


def test_list_lines_keep_file_order_and_skip_blanks() -> None:
    text = "".join(public_lines()[:2]) + "\n" + checkpoint_line("exp07__tulu__step01500", "/p", "tulu", 1500, ["final"])
    lines = list_lines(text)
    assert len(lines) == 3 and lines[2].startswith("exp07__tulu")


def test_item_outcomes_are_sorted_compact_and_strict() -> None:
    samples = {"arc_easy": [{"doc_id": 2, "acc": 1.0, "acc_norm": 0.0}, {"doc_id": 0, "acc": 0.0, "acc_norm": 1.0}],
               "mmlu_x": [{"doc_id": 5, "acc": 1.0}]}
    out = item_outcomes(samples)
    assert out == {"arc_easy": {"doc_id": [0, 2], "acc": "01", "acc_norm": "10"}, "mmlu_x": {"doc_id": [5], "acc": "1"}}
    with pytest.raises(ValueError):
        bits([0.5])


def test_truncation_counts_use_the_longest_request_of_each_doc() -> None:
    samples = {"arc_easy": [{"arguments": [("a b c", " d"), ("a b c", " d e f g")]},
                            {"arguments": [("a", " b")]}]}
    n_words = lambda s: len(s.split())  # noqa: E731
    assert truncation_counts(samples, n_words, 5) == {"arc_easy": {"n_docs": 2, "n_over": 1, "max_tokens": 7}}

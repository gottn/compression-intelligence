from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from cilab.sft import (
    Message,
    assistant_spans,
    checkpoint_label,
    checkpoint_line,
    drop_reason,
    epoch_order,
    first_crossings,
    fits,
    loss_mask,
    lr_at,
    marks_due,
    pack_greedy,
    parse_checkpoint_line,
    row_arrays,
    row_stream,
    shuffled_order,
    stop_reason,
    take_until_tokens,
    warmup_steps,
)

BOS = EOS = "<|endoftext|>"
CONV: tuple[Message, ...] = ({"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
                             {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "2+2?"},
                             {"role": "assistant", "content": "4"})


def olmo_render(messages: Sequence[Message], gen: bool) -> str:
    """The OLMo-2-Instruct chat template, written out in Python."""
    out = BOS
    for i, m in enumerate(messages):
        last = i == len(messages) - 1
        if m["role"] in ("system", "user"):
            out += f"<|{m['role']}|>\n{m['content']}\n"
        elif m["role"] == "assistant":
            out += f"<|assistant|>\n{m['content']}{EOS}" + ("" if last else "\n")
        if last and gen:
            out += "<|assistant|>\n"
    return out


def test_assistant_spans_cover_reply_and_eos_only() -> None:
    full = olmo_render(CONV, False)
    spans = assistant_spans(olmo_render, CONV)
    assert [full[a:b] for a, b in spans] == ["Hello." + EOS, "4" + EOS]
    with pytest.raises(ValueError):
        assistant_spans(olmo_render, ({"role": "assistant", "content": "x"},))
    with pytest.raises(ValueError):  # a template that does not grow by appending
        assistant_spans(lambda m, g: "".join(x["content"] for x in reversed(m)) + ("!" if g else ""), CONV[1:3])


def test_loss_mask_from_token_starts() -> None:
    offsets = [(0, 3), (3, 5), (5, 9), (9, 12), (12, 14)]
    assert loss_mask(offsets, [(5, 12)]).tolist() == [False, False, True, True, False]
    assert loss_mask(offsets, [(4, 5), (12, 20)]).tolist() == [False, False, False, False, True]
    assert loss_mask(offsets, []).sum() == 0


def test_drop_reasons() -> None:
    assert drop_reason(CONV, (EOS, "<|pad|>")) is None
    assert drop_reason([], ()) == "empty"
    assert drop_reason([{"role": "user", "content": "a"}], ()) == "no_assistant"
    assert drop_reason([{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}], ()) == \
        "assistant_first"
    assert drop_reason([{"role": "tool", "content": "t"}, *CONV], ()) == "unknown_role"
    assert drop_reason([{"role": "user", "content": " "}, {"role": "assistant", "content": "a"}], ()) == \
        "empty_content"
    assert drop_reason([{"role": "user", "content": "q"}, {"role": "assistant", "content": f"a{EOS}"}], (EOS,)) == \
        "reserved_text"
    assert fits(2048, 1, 2048) and not fits(2049, 5, 2048) and not fits(10, 0, 2048)


def test_selection_is_seeded_and_token_matched() -> None:
    assert shuffled_order(10, 0).tolist() == shuffled_order(10, 0).tolist() != shuffled_order(10, 1).tolist()
    assert sorted(shuffled_order(10, 0).tolist()) == list(range(10))
    assert take_until_tokens([5, 5, 5], 10) == 2 and take_until_tokens([5, 5, 5], 11) == 3
    assert take_until_tokens([5, 5], 100) == 2 and take_until_tokens([7], 1) == 1


def test_pack_greedy_fills_rows_in_order() -> None:
    assert pack_greedy([3, 4, 2, 5, 6], 8) == [(0, 1), (2, 3), (4,)]
    assert pack_greedy([8, 8], 8) == [(0,), (1,)]
    with pytest.raises(ValueError):
        pack_greedy([9], 8)


def test_row_stream_reshuffles_each_epoch_and_covers_every_example() -> None:
    lengths = [3, 4, 2, 5, 6, 1, 7]
    rows = list(islice(row_stream(lengths, 8, 0), 40))
    for epoch in (0, 1):
        got = [i for e, r in rows if e == epoch for i in r]
        assert sorted(got) == list(range(len(lengths)))  # each example exactly once per epoch
        assert got == epoch_order(len(lengths), 0, epoch).tolist()  # in the epoch's order
    assert all(sum(lengths[i] for i in r) <= 8 for _, r in rows)
    assert epoch_order(7, 0, 0).tolist() != epoch_order(7, 0, 1).tolist()


def test_row_arrays_restart_positions_and_pad() -> None:
    a = (np.array([10, 11, 12]), np.array([False, True, True]))
    b = (np.array([20, 21]), np.array([False, True]))
    ids, pos, tgt = row_arrays([a, b], 8, 99)
    assert ids.tolist() == [10, 11, 12, 20, 21, 99, 99, 99]
    assert pos.tolist() == [0, 1, 2, 0, 1, 0, 1, 2]
    assert tgt.tolist() == [False, True, True, False, True, False, False, False]
    with pytest.raises(ValueError):
        row_arrays([a, a, a], 8, 99)


def test_lr_schedule() -> None:
    w = warmup_steps(1500, 0.03)
    assert w == 45
    assert lr_at(0, 2e-5, w, 1500) == pytest.approx(2e-5 / 45)
    assert lr_at(44, 2e-5, w, 1500) == pytest.approx(2e-5)
    assert lr_at(45, 2e-5, w, 1500) == pytest.approx(2e-5)
    mid = 45 + (1500 - 45) // 2
    assert lr_at(mid, 2e-5, w, 1500) == pytest.approx(2e-5 * 0.5 * (1 + math.cos(math.pi * (mid - 45) / 1455)))
    assert lr_at(1500, 2e-5, w, 1500) == pytest.approx(0.0, abs=1e-20)
    assert lr_at(2000, 2e-5, w, 1500) == pytest.approx(0.0, abs=1e-20)


def test_stop_reason() -> None:
    assert stop_reason(10, 1500, 100.0, 18000.0) is None
    assert stop_reason(1500, 1500, 100.0, 18000.0) == "max_steps"
    assert stop_reason(10, 1500, 18000.0, 18000.0) == "max_wall"


def test_marks_from_the_drift_log() -> None:
    marks = (3.0, 6.0, 10.0)
    log = [(0, 0.0), (20, 1.5), (40, 3.2), (60, 2.9), (80, 7.0), (100, 12.0)]
    assert first_crossings(log, marks) == {3.0: 40, 6.0: 80, 10.0: 100}
    assert [marks_due(log[:k], marks) for k in range(1, 7)] == [(), (), (3.0,), (), (6.0,), (10.0,)]
    assert marks_due([(0, 0.0), (20, 11.0)], marks) == (3.0, 6.0, 10.0)  # one checkpoint, three marks
    assert marks_due([], marks) == ()


def test_checkpoint_lines_round_trip() -> None:
    label = checkpoint_label("exp06", "tulu", 340)
    assert label == "exp06__tulu__step00340"
    line = checkpoint_line(label, "/r/tulu/ckpt/step00340", "tulu", 340, ["3", "6"])
    assert parse_checkpoint_line(line) == {"label": label, "path": "/r/tulu/ckpt/step00340", "recipe": "tulu",
                                           "step": "340", "marks": "3,6"}
    assert parse_checkpoint_line(checkpoint_line(label, "/p", "tulu", 340, []))["marks"] == "-"


INSTRUCT = "allenai/OLMo-2-0425-1B-Instruct"


def _cached(model: str) -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model.replace('/', '--')}"
    return hub.exists()


@pytest.mark.skipif(not _cached(INSTRUCT), reason="OLMo-2-Instruct tokenizer not in the HF cache")
def test_real_template_and_offsets_give_reply_and_eos_tokens() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(INSTRUCT, local_files_only=True)

    def render(m: Sequence[Message], gen: bool) -> str:
        return tok.apply_chat_template(list(m), tokenize=False, add_generation_prompt=gen)

    assert render(CONV, False) == olmo_render(CONV, False)
    assert render(CONV[:2], True) == olmo_render(CONV[:2], True)
    text = render(CONV, False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    mask = loss_mask([tuple(o) for o in enc["offset_mapping"]], assistant_spans(render, CONV))
    targets = [tok.decode([i]) for i, m in zip(enc["input_ids"], mask, strict=True) if m]
    assert "".join(targets) == "Hello." + EOS + "4" + EOS
    assert enc["input_ids"][0] == tok.bos_token_id and targets[-1] == EOS


def test_stage2_helpers() -> None:
    from cilab.sft import final_checkpoint, parse_marks, remaining_marks, saved_marks
    lines = [checkpoint_line("exp06__tulu__step00300", "/r/tulu/ckpt/step00300", "tulu", 300, ["3"]),
             checkpoint_line("exp06__smoltalk__step01500", "/r/smoltalk/ckpt/step01500", "smoltalk", 1500,
                             ["3", "final"]),
             checkpoint_line("exp06__tulu__step01500", "/r/tulu/ckpt/step01500", "tulu", 1500, ["final"]),
             "\n"]
    assert parse_marks("10,3,6") == (3.0, 6.0, 10.0) and parse_marks("") == ()
    assert final_checkpoint(lines, "tulu") == ("/r/tulu/ckpt/step01500", 1500)
    assert final_checkpoint(lines, "smoltalk") == ("/r/smoltalk/ckpt/step01500", 1500)
    with pytest.raises(ValueError):
        final_checkpoint(lines, "mathcot")
    assert saved_marks(lines, "tulu") == {3.0} and saved_marks(lines, "mathcot") == frozenset()
    assert remaining_marks((3.0, 6.0, 10.0), saved_marks(lines, "tulu")) == (6.0, 10.0)
    assert remaining_marks((3.0, 6.0, 10.0), frozenset()) == (3.0, 6.0, 10.0)

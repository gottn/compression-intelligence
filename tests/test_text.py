from __future__ import annotations

from itertools import pairwise

import pytest
from transformers import PreTrainedTokenizerBase

from cilab.io import read_tokenizer_config
from cilab.pairs import model_roles, model_slug
from cilab.text import cut_to_bytes, decode, doc_start_token, encode, make_windows, window_positions

MULTIBYTE = "Naïve café — 東京 🙂 مرحبا 𝔘nicode. " * 12  # 2-, 3- and 4-byte characters


@pytest.mark.parametrize("size", [3, 5, 8, 64])
def test_make_windows_round_trips_and_never_splits_a_char(gpt2_tok: PreTrainedTokenizerBase, size: int) -> None:
    ids = encode(gpt2_tok, MULTIBYTE)
    wins = make_windows(gpt2_tok, ids, MULTIBYTE, size)
    assert "".join(s for _, _, s in wins) == MULTIBYTE
    assert wins[0][0] == 0 and wins[-1][1] == len(ids)
    assert all(e0 == b1 for (_, e0, _), (b1, _, _) in pairwise(wins))
    assert all(0 < e - b <= size for b, e, _ in wins)
    for b, e, s in wins:
        assert "�" not in s
        assert decode(gpt2_tok, ids[b:e]) == s
    if size <= 8:  # small windows really do hit multi-byte boundaries and get shortened
        assert any(e - b < size for b, e, _ in wins[:-1])


def test_make_windows_rejects_a_char_longer_than_the_window(gpt2_tok: PreTrainedTokenizerBase) -> None:
    text = "東京"  # 4 GPT-2 tokens, each character spans 2
    with pytest.raises(ValueError):
        make_windows(gpt2_tok, encode(gpt2_tok, text), text, 1)


def test_window_positions_restart_at_each_window(gpt2_tok: PreTrainedTokenizerBase) -> None:
    pos = window_positions(gpt2_tok, MULTIBYTE, 16, "raw")
    assert pos.size == len(encode(gpt2_tok, MULTIBYTE))
    assert pos[0] == 0 and pos.max() <= 15


def test_doc_start_token_is_endoftext_for_gpt2(gpt2_tok: PreTrainedTokenizerBase) -> None:
    assert doc_start_token(gpt2_tok, read_tokenizer_config("openai-community/gpt2")) == (50256, "<|endoftext|>")
    assert doc_start_token(gpt2_tok, {}) == (50256, "<|endoftext|>")


def test_cut_to_bytes_drops_partial_char() -> None:
    assert cut_to_bytes("aé", 2) == "a"
    assert cut_to_bytes("aé", None) == "aé"


def test_model_roles_and_slug() -> None:
    roles = model_roles((("p", "org/base", "org/post"),))
    assert roles == [{"model": "org/base", "pair": "p", "role": "base"},
                     {"model": "org/post", "pair": "p", "role": "post"}]
    assert model_slug("Qwen/Qwen3-1.7B") == "Qwen__Qwen3-1.7B"

"""Tokenization, windowing, document-start and chat-prefix rules. Pure: the tokenizer is a parameter.

Window rule (exp01): consecutive non-overlapping windows of <= size tokens; a boundary moves left by a
token or two if it would split a multi-byte UTF-8 character, so the window texts concatenate to the
corpus exactly and bytes are counted once.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

CHAT_USER = "Continue the text."
PLACEHOLDER = "@@PLACEHOLDER@@"
EOT = "<|endoftext|>"

Window = tuple[int, int, str]  # (start token, end token, window text)


def cut_to_bytes(text: str, max_bytes: int | None) -> str:
    """First max_bytes UTF-8 bytes of text, dropping a trailing partial character; no-op if falsy."""
    if not max_bytes:
        return text
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def encode(tok: PreTrainedTokenizerBase, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def decode(tok: PreTrainedTokenizerBase, ids: Sequence[int]) -> str:
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def make_windows(tok: PreTrainedTokenizerBase, ids: list[int], text: str, size: int) -> list[Window]:
    """Consecutive (start, end, text) windows of <= size tokens whose texts concatenate to `text`."""
    wins: list[Window] = []
    b, pos = 0, 0
    while b < len(ids):
        e = min(b + size, len(ids))
        s = decode(tok, ids[b:e])
        while e < len(ids) and not (text.startswith(s, pos) and not s.endswith("�")):
            e -= 1
            s = decode(tok, ids[b:e])
        if e <= b or not text.startswith(s, pos):
            raise ValueError(f"window {len(wins)} does not decode to a prefix of the remaining corpus")
        wins.append((b, e, s))
        pos += len(s)
        b = e
    if pos != len(text):
        raise ValueError(f"windows cover {pos} of {len(text)} chars")
    return wins


def window_positions(tok: PreTrainedTokenizerBase, text: str, window: int, condition: str) -> np.ndarray:
    """In-window position (0 = first scored token) of every scored token, concatenated over windows."""
    wins = make_windows(tok, encode(tok, text), text, window)
    if condition == "raw":
        return np.concatenate([np.arange(e - b) for b, e, _ in wins])
    return np.concatenate([np.arange(len(encode(tok, s))) for _, _, s in wins])


def adds_bos(tok: PreTrainedTokenizerBase, tok_config: Mapping[str, object]) -> bool:
    """True if the tokenizer config asks for a BOS in front of every sequence."""
    return bool(tok_config.get("add_bos_token")) and tok.bos_token_id is not None


def doc_start_token(tok: PreTrainedTokenizerBase, tok_config: Mapping[str, object]) -> tuple[int, str]:
    """(id, string) of the RAW document-start token: BOS if the config adds one, else <|endoftext|>,
    else EOS. `tok_config` is the model's tokenizer_config.json ({} if absent)."""
    if adds_bos(tok, tok_config):
        return tok.bos_token_id, tok.bos_token
    eot = tok.convert_tokens_to_ids(EOT)
    # compare by string, not against unk_token_id: SmolLM2-Instruct's unk token IS <|endoftext|> (id 0)
    if eot is not None and tok.convert_ids_to_tokens(eot) == EOT:
        return eot, EOT
    if tok.eos_token_id is not None:
        return tok.eos_token_id, tok.eos_token
    raise ValueError("no document-start token")


def chat_prefix(tok: PreTrainedTokenizerBase, model_id: str) -> tuple[list[int], str] | None:
    """Token ids and text of the chat template up to the assistant turn; None if no template."""
    if tok.chat_template is None:
        return None
    kw = {"enable_thinking": False} if "Qwen3" in model_id else {}
    msgs = [{"role": "user", "content": CHAT_USER}, {"role": "assistant", "content": PLACEHOLDER}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, **kw)
    if rendered.count(PLACEHOLDER) != 1:
        raise ValueError("placeholder not rendered exactly once")
    pre = rendered.split(PLACEHOLDER)[0]
    return encode(tok, pre), pre  # special-token strings in `pre` are parsed as special tokens

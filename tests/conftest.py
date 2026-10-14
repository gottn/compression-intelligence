"""Shared fixtures: the cached GPT-2 tokenizer (no network; run download_models.py once if missing)."""
from __future__ import annotations

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

GPT2 = "openai-community/gpt2"


@pytest.fixture(scope="session")
def gpt2_tok() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(GPT2, local_files_only=True)

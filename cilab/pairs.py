"""Model registry shared by the experiments: base -> post-trained pairs and the corpus names. Pure."""
from __future__ import annotations

PAIRS: tuple[tuple[str, str, str], ...] = (
    ("qwen2.5", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen3", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B"),
    ("smollm2", "HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("qwen2.5-math", "Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
)
CORPORA: tuple[str, ...] = ("wiki2026", "arxiv2026", "gutenberg2026", "enwik8_ctrl")
FRESH_CORPORA: tuple[str, ...] = ("wiki2026", "arxiv2026", "gutenberg2026")


def model_roles(pairs: tuple[tuple[str, str, str], ...] = PAIRS) -> list[dict[str, str]]:
    """One {model, pair, role} record per model, base before post, in pair order."""
    return [{"model": m, "pair": pair, "role": role}
            for pair, base, post in pairs for role, m in (("base", base), ("post", post))]


def model_slug(model_id: str) -> str:
    """File-name-safe model id: 'Qwen/Qwen3-1.7B' -> 'Qwen__Qwen3-1.7B'."""
    return model_id.replace("/", "__")

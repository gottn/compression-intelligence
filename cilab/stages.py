"""Staged post-training checkpoints (exp03, exp04): each model's family, stage, recipe and parent. Pure.

A family is one base model and everything trained from it. `parent` is the checkpoint a stage was
trained from (None for the family base), so a family is a tree: OLMo-2 is a chain base → SFT → DPO →
RLVR; Qwen2.5-1.5B branches into Instruct, SimpleRL-Zoo (GRPO straight from the base) and the Math
continued-pretraining line that R1-Distill was distilled onto.
`tokenizer` names the vocabulary; models with the same name must encode the corpora to identical ids.

exp03 scored the first 14 rows on AAU (`EXP03_STAGES`); exp04 benchmarks all 18. The last 4 rows are
exp01's SmolLM2 and Qwen3 pairs, whose bpb was scored on the Mac in exp01.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    model: str
    family: str
    stage: str
    recipe: str
    parent: str | None
    tokenizer: str


OLMO_1B = "allenai/OLMo-2-0425-1B"
Q05, Q15, Q3 = "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"
QWEN_INSTRUCT = "Qwen2.5 post-training: SFT (~1M examples), offline DPO, online GRPO"
SMOL, QW3_BASE = "HuggingFaceTB/SmolLM2-1.7B", "Qwen/Qwen3-1.7B-Base"
SIMPLERL = "zero-RL: GRPO from the base on GSM8K/MATH problems, rule-based reward (SimpleRL-Zoo)"

STAGES: tuple[Stage, ...] = (
    Stage(OLMO_1B, "olmo2-1b", "base", "pretraining (OLMo-mix) + Dolmino mid-training", None, "olmo2"),
    Stage(f"{OLMO_1B}-SFT", "olmo2-1b", "sft", "Tülu 3 SFT mix", OLMO_1B, "olmo2"),
    Stage(f"{OLMO_1B}-DPO", "olmo2-1b", "dpo", "Tülu 3 on-policy preference DPO", f"{OLMO_1B}-SFT", "olmo2"),
    Stage(f"{OLMO_1B}-Instruct", "olmo2-1b", "rlvr", "RLVR (GRPO) on verifiable math and instruction rewards",
          f"{OLMO_1B}-DPO", "olmo2"),
    Stage(Q05, "qwen2.5-0.5b", "base", "pretraining (18T tokens)", None, "qwen2.5"),
    Stage(f"{Q05}-Instruct", "qwen2.5-0.5b", "instruct", QWEN_INSTRUCT, Q05, "qwen2.5"),
    Stage("hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo", "qwen2.5-0.5b", "rl-zero", SIMPLERL, Q05, "qwen2.5"),
    Stage(Q15, "qwen2.5-1.5b", "base", "pretraining (18T tokens)", None, "qwen2.5"),
    Stage(f"{Q15}-Instruct", "qwen2.5-1.5b", "instruct", QWEN_INSTRUCT, Q15, "qwen2.5"),
    Stage("hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo", "qwen2.5-1.5b", "rl-zero", SIMPLERL, Q15, "qwen2.5"),
    Stage("Qwen/Qwen2.5-Math-1.5B", "qwen2.5-1.5b", "math-cpt",
          "continued pretraining on Qwen Math Corpus v2", Q15, "qwen2.5"),
    Stage("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "qwen2.5-1.5b", "r1-distill",
          "SFT on ~800k DeepSeek-R1 reasoning traces", "Qwen/Qwen2.5-Math-1.5B", "qwen2.5"),
    Stage(Q3, "qwen2.5-3b", "base", "pretraining (18T tokens)", None, "qwen2.5"),
    Stage(f"{Q3}-Instruct", "qwen2.5-3b", "instruct", QWEN_INSTRUCT, Q3, "qwen2.5"),
    Stage(SMOL, "smollm2", "base", "pretraining (11T tokens)", None, "smollm2"),
    Stage(f"{SMOL}-Instruct", "smollm2", "instruct", "SFT on SmolTalk, then DPO on UltraFeedback", SMOL, "smollm2"),
    Stage(QW3_BASE, "qwen3", "base", "pretraining (36T tokens)", None, "qwen3"),
    Stage("Qwen/Qwen3-1.7B", "qwen3", "instruct",
          "strong-to-weak distillation (off- and on-policy) from Qwen3 flagship models, thinking + non-thinking",
          QW3_BASE, "qwen3"),
)
FAMILIES: tuple[str, ...] = ("olmo2-1b", "qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "smollm2", "qwen3")
EXP03_FAMILIES: tuple[str, ...] = FAMILIES[:4]
EXP03_STAGES: tuple[Stage, ...] = tuple(s for s in STAGES if s.family in EXP03_FAMILIES)
RL_STAGES: tuple[str, ...] = ("rlvr", "rl-zero")  # stages whose last step is pure RL on verifiable rewards


def stage_of(model: str, stages: tuple[Stage, ...] = STAGES) -> Stage:
    """The Stage record of a model id."""
    hits = [s for s in stages if s.model == model]
    if len(hits) != 1:
        raise KeyError(f"{model}: {len(hits)} stage records")
    return hits[0]


def model_at(index: int, stages: tuple[Stage, ...] = STAGES) -> str:
    """Model id at a Slurm array index (0-based, table order). Negative or too-large indices raise."""
    if not 0 <= index < len(stages):
        raise IndexError(f"array index {index} outside 0..{len(stages) - 1}")
    return stages[index].model


def family_base(family: str, stages: tuple[Stage, ...] = STAGES) -> str:
    """Model id of the family's root (the one stage without a parent)."""
    roots = [s.model for s in stages if s.family == family and s.parent is None]
    if len(roots) != 1:
        raise ValueError(f"{family}: {len(roots)} roots")
    return roots[0]


def comparisons(stages: tuple[Stage, ...] = STAGES) -> list[tuple[str, str, str]]:
    """(model, reference, kind) for every non-root stage: versus the family base ('vs_base') and, when the
    parent is not the base, versus the parent ('vs_parent'). Ordered as `stages`."""
    out = []
    for s in stages:
        if s.parent is None:
            continue
        base = family_base(s.family, stages)
        out.append((s.model, base, "vs_base"))
        if s.parent != base:
            out.append((s.model, s.parent, "vs_parent"))
    return out


def tokenizer_groups(stages: tuple[Stage, ...] = STAGES) -> dict[str, list[str]]:
    """{tokenizer name: model ids in table order}, groups sorted by name."""
    names = sorted({s.tokenizer for s in stages})
    return {n: [s.model for s in stages if s.tokenizer == n] for n in names}

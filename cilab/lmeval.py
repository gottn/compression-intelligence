"""lm-evaluation-harness driver helpers shared by exp04 (run_eval.py), exp06 and exp07 (run_battery.py).

Pure helpers first (no lm_eval import at module level, so they run on the Mac without the harness); the effects
(TaskManager, GPU name, generation config) sit below the marker and import lm_eval / torch / transformers lazily.
Moved unchanged from exp04_benchmarks/run_eval.py; exp04's behaviour and outputs do not change.
"""
from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from pathlib import Path

MMLU_SUBJECTS: tuple[str, ...] = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics", "econometrics", "electrical_engineering",
    "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy",
    "prehistory", "professional_accounting", "professional_law", "professional_medicine",
    "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
)
HELLASWAG_N, MMLU_N = 2000, 40
SEEDS: dict[str, int] = {"random_seed": 0, "numpy_random_seed": 1234, "torch_random_seed": 1234,
                         "fewshot_random_seed": 1234}  # lm-eval's defaults, passed explicitly
MAX_LENGTH = 2048  # tokens
GEN_KEEP: tuple[str, ...] = ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id")


# ----------------------------------------------------------------------------- pure
def samples_spec(hellaswag_n: int = HELLASWAG_N, mmlu_n: int = MMLU_N,
                 subjects: tuple[str, ...] = MMLU_SUBJECTS) -> dict[str, list[int]]:
    """lm-eval `--samples` mapping that reproduces `--limit` per task: the first n doc indices."""
    spec = {"hellaswag": list(range(hellaswag_n))}
    spec.update({f"mmlu_{s}": list(range(mmlu_n)) for s in subjects})
    return spec


def model_kwargs(model: str, tokenizer: str | None = None) -> dict[str, object]:
    """HFLM arguments; identical for every checkpoint except the id (and, for exp06, an optional tokenizer)."""
    kw: dict[str, object] = {"pretrained": model, "dtype": "float16", "max_length": MAX_LENGTH, "batch_size": "auto",
                             "device": "cuda"}
    return {**kw, "tokenizer": tokenizer} if tokenizer else kw


def neutral_generation_config(diff: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """(kept, dropped) from a GenerationConfig.to_diff_dict(): keep only token ids; drop every length,
    sampling and penalty default the checkpoint ships, so decoding is lm-eval's greedy setting for all models."""
    kept = {k: diff[k] for k in GEN_KEEP if diff.get(k) is not None}
    dropped = {k: v for k, v in sorted(diff.items()) if k not in GEN_KEEP and k != "transformers_version"}
    return kept, dropped


def thinking_switch(chat_template: str | None) -> dict[str, object]:
    """Chat-template kwargs that turn thinking mode off where the template has the switch (Qwen3)."""
    return {"enable_thinking": False} if chat_template and "enable_thinking" in chat_template else {}


def _metrics(entry: Mapping[str, object]) -> dict[str, float]:
    return {k: float(v) for k, v in entry.items() if "," in k and isinstance(v, int | float)}


def summarize_results(res: Mapping[str, object], tasks: Sequence[str]) -> dict[str, dict[str, object]]:
    """Per top-level task: metrics ('metric,filter' -> value, stderr included), version, n-shot and doc
    counts from an lm-eval results JSON. For a group (mmlu), counts are summed over its subtasks and the
    per-subtask effective counts are kept, so the 40-per-subject selection can be checked."""
    results = {**res.get("results", {}), **res.get("groups", {})}
    subtasks = res.get("group_subtasks", {})
    versions, nshot, nsamp = res.get("versions", {}), res.get("n-shot", {}), res.get("n-samples", {})
    out = {}
    for t in tasks:
        if t not in results:
            continue
        leaves = _leaf_tasks(t, subtasks)
        eff = {leaf: int(nsamp[leaf]["effective"]) for leaf in leaves if leaf in nsamp}
        orig = {leaf: int(nsamp[leaf]["original"]) for leaf in leaves if leaf in nsamp}
        out[t] = {"metrics": _metrics(results[t]), "version": versions.get(t),
                  "leaf_versions": sorted({str(versions.get(leaf)) for leaf in leaves}),
                  "n_shot": sorted({int(nshot[leaf]) for leaf in leaves if leaf in nshot}),
                  "n_effective": sum(eff.values()), "n_original": sum(orig.values()),
                  "n_leaves": len(leaves)}
        if len(leaves) > 1:
            out[t]["leaf_effective"] = dict(sorted(eff.items()))
    return out


def _leaf_tasks(task: str, subtasks: Mapping[str, Sequence[str]]) -> list[str]:
    """Leaf task names under a task or group. lm-eval lists plain tasks in group_subtasks with an empty list;
    a task that is absent or has no subtasks is its own leaf."""
    if not subtasks.get(task):
        return [task]
    return [leaf for s in subtasks[task] for leaf in _leaf_tasks(s, subtasks)]


# ----------------------------------------------------------------------------- effects
def gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception as e:  # report, never fail the run on it
        return f"unknown ({e})"


def env_versions() -> dict[str, str]:
    import importlib.metadata as md
    out = {"python": platform.python_version()}
    for pkg in ("lm_eval", "transformers", "torch", "datasets", "accelerate", "nltk"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "missing"
    return out


def task_manager(task_dirs: Sequence[str]):  # -> lm_eval.tasks.TaskManager
    """TaskManager over only the given lm-eval task directories (fast to index over a network file system)."""
    import lm_eval.tasks
    from lm_eval.tasks import TaskManager
    root = Path(lm_eval.tasks.__file__).parent
    return TaskManager(include_defaults=False, include_path=[str(root / d) for d in task_dirs])


def set_neutral_generation_config(lm: object) -> tuple[dict[str, object], dict[str, object]]:
    from transformers import GenerationConfig
    kept, dropped = neutral_generation_config(lm.model.generation_config.to_diff_dict())
    lm.model.generation_config = GenerationConfig(**kept)
    return kept, dropped


def free_cuda() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception as e:  # report, never fail the run on it
        print(f"empty_cache failed: {e}", flush=True)

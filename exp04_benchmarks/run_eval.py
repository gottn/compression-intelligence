"""exp04 driver, run INSIDE the AAU jobs (never on the Mac): lm-evaluation-harness on one checkpoint.

    python exp04_benchmarks/run_eval.py prefetch                      # setup job: datasets + nltk data
    python exp04_benchmarks/run_eval.py run --index I --out DIR [--groups loglik,gsm8k,ifeval]
    python exp04_benchmarks/run_eval.py run --index I --out DIR --groups gsm8k_chat,ifeval_chat   # chat addendum
    python exp04_benchmarks/run_eval.py run --model-path P [--tokenizer T] [--label L] --out DIR --groups ...   # exp06

`run` maps the Slurm array index to a model (cilab.stages.model_at), loads it once (lm-eval HFLM) and calls
`lm_eval.simple_evaluate` once per task group, each in its own try/except:
  loglik  arc_easy, arc_challenge, hellaswag, piqa, winogrande, mmlu   (log-likelihood, harness few-shot defaults)
  gsm8k   gsm8k, 5-shot (task default), first 300 test problems        (generative)
  ifeval  ifeval, 0-shot, first 200 prompts                            (generative)
Every call: HF backend, fp16, batch_size auto, max_length 2048, NO chat template (base and post-trained models
see identical prompts), seeds 0 / 1234 / 1234 / 1234, log_samples off.

In-process, not one CLI process per group: the dry run (job 1080307) spent ~6.5 min per CLI call indexing all
of lm-eval's task YAMLs over the network file system with the GPU idle. Here the TaskManager indexes only the
7 task directories used (TASK_DIRS), once, and the model loads once. simple_evaluate is what the CLI calls.

`limit` is global to one call, so the loglik group passes the brief's per-task limits as `samples` (doc
indices): hellaswag docs 0..1999 and docs 0..39 of each of the 57 MMLU subjects; tasks not listed run in full.
lm-eval selects `--limit N` as docs 0..N-1, so this is the same selection.
Generation config: transformers 5 merges the checkpoint's generation_config.json into every generate() call,
and an explicit max_new_tokens there (Qwen: 2048) beats the max_length lm-eval passes, so gsm8k ran past
lm-eval's 256-token default in the dry run; instruct checkpoints also carry sampling defaults (temperature,
top_p, repetition_penalty). The driver replaces it with a config holding only the token ids
(neutral_generation_config), so every model decodes greedily under the harness's own limits.
max_length 2048: lm-eval's auto batch size for generation probes a forward pass at the model's max length
(131072 for Qwen2.5-1.5B), which would force batch 1; 2048 fits every prompt here (ifeval: ctx + 1280 gen).

Chat condition (addendum, post-trained checkpoints only): gsm8k_chat and ifeval_chat repeat the two generative
groups with the model's own chat template and the few-shot examples as separate turns
(apply_chat_template, fewshot_as_multiturn); limits, seeds and generation caps are unchanged. Templates that
read `enable_thinking` (Qwen3) get enable_thinking=False. Its summary is summary_chat.json.

exp06 addition: --model-path evaluates a local checkpoint directory (or any hub id) instead of a stage-table index,
--tokenizer loads the tokenizer (and its chat template) from elsewhere, and --label names the output folder (default:
the slug of the path). Everything else is identical; with --index the behaviour is exp04's, unchanged.

Outputs under DIR/<model slug>/: <group>/results.json (lm-eval's results dict), <group>/done.json,
loglik_samples.json, summary.json. A group with a done.json is skipped (resumable after a Slurm requeue).
A failed group does not stop the others; the process exits 1 at the end if any group failed.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cilab`

from cilab.pairs import model_slug
from cilab.stages import model_at

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
TASK_DIRS: tuple[str, ...] = ("arc", "hellaswag", "piqa", "winogrande", "mmlu/default", "gsm8k", "ifeval")


@dataclass(frozen=True)
class Group:
    name: str
    tasks: tuple[str, ...]
    limit: int | None  # docs per task; None = full task (or the samples file, for loglik)
    chat: bool = False  # apply the model's chat template, few-shot as multi-turn


GROUPS: tuple[Group, ...] = (
    Group("loglik", ("arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "mmlu"), None),
    Group("gsm8k", ("gsm8k",), 300),
    Group("ifeval", ("ifeval",), 200),
    Group("gsm8k_chat", ("gsm8k",), 300, chat=True),
    Group("ifeval_chat", ("ifeval",), 200, chat=True),
)
DEFAULT_GROUPS = "loglik,gsm8k,ifeval"
# Metrics printed per task as the group finishes (all metrics are kept in summary.json).
HEADLINE: dict[str, tuple[str, ...]] = {
    "arc_easy": ("acc_norm,none",), "arc_challenge": ("acc_norm,none",), "hellaswag": ("acc_norm,none",),
    "piqa": ("acc_norm,none",), "winogrande": ("acc,none",), "mmlu": ("acc,none",),
    "gsm8k": ("exact_match,flexible-extract", "exact_match,strict-match"),
    "ifeval": ("prompt_level_strict_acc,none", "inst_level_strict_acc,none"),
}


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


def run_target(index: int | None, model_path: str | None, label: str | None) -> tuple[str, str]:
    """(model id or path, output folder name): exp04's stage-table index, or exp06's --model-path with a label."""
    if (index is None) == (model_path is None):
        raise ValueError("give exactly one of --index and --model-path")
    if index is not None:
        if label:
            raise ValueError("--label only goes with --model-path")
        model = model_at(index)
        return model, model_slug(model)
    return model_path, label or model_slug(model_path.strip("/"))


GEN_KEEP: tuple[str, ...] = ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id")


def neutral_generation_config(diff: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """(kept, dropped) from a GenerationConfig.to_diff_dict(): keep only token ids; drop every length,
    sampling and penalty default the checkpoint ships, so decoding is lm-eval's greedy setting for all models."""
    kept = {k: diff[k] for k in GEN_KEEP if diff.get(k) is not None}
    dropped = {k: v for k, v in sorted(diff.items()) if k not in GEN_KEEP and k != "transformers_version"}
    return kept, dropped


def thinking_switch(chat_template: str | None) -> dict[str, object]:
    """Chat-template kwargs that turn thinking mode off where the template has the switch (Qwen3)."""
    return {"enable_thinking": False} if chat_template and "enable_thinking" in chat_template else {}


def summary_name(groups: Sequence[Group]) -> str:
    """summary.json for the main run; summary_chat.json when only chat groups run, so the two never collide."""
    return "summary_chat.json" if groups and all(g.chat for g in groups) else "summary.json"


def eval_kwargs(group: Group, samples: Mapping[str, list[int]] | None) -> dict[str, object]:
    """simple_evaluate arguments of one task group (no chat template, no sample logging, fixed seeds)."""
    kw: dict[str, object] = {"tasks": list(group.tasks), "log_samples": False, "apply_chat_template": group.chat,
                             "fewshot_as_multiturn": group.chat, **SEEDS}
    if group.limit is not None:
        kw["limit"] = group.limit
    elif samples is not None:
        kw["samples"] = dict(samples)
    return kw


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


def task_lines(model: str, group: str, summary: Mapping[str, Mapping[str, object]], wall_s: float) -> list[str]:
    """One printable line per task: headline metric(s), stderr, docs evaluated, group wall time."""
    lines = []
    for t, s in summary.items():
        m = s["metrics"]
        cells = [f"{k.split(',')[0]}[{k.split(',')[1]}]={m[k]:.4f}±{m.get(k.replace(',', '_stderr,', 1), float('nan')):.4f}"
                 for k in HEADLINE.get(t, ()) if k in m]
        lines.append(f"SCORE {model} {t}: " + " ".join(cells) + f" n={s['n_effective']} shot={s['n_shot']} "
                     f"({group} group, {wall_s:.0f} s)")
    return lines


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


def task_manager():  # -> lm_eval.tasks.TaskManager
    """TaskManager over only the task directories this experiment uses (fast to index)."""
    import lm_eval.tasks
    from lm_eval.tasks import TaskManager
    root = Path(lm_eval.tasks.__file__).parent
    return TaskManager(include_defaults=False, include_path=[str(root / d) for d in TASK_DIRS])


def run_group(lm: object, tm: object, model: str, group: Group, root: Path) -> dict[str, object]:
    from lm_eval import simple_evaluate
    from lm_eval.utils import handle_non_serializable
    gdir = root / group.name
    gdir.mkdir(parents=True, exist_ok=True)
    samples = samples_spec() if group.name == "loglik" else None
    if samples is not None:
        (root / "loglik_samples.json").write_text(json.dumps(samples))
    kw = eval_kwargs(group, samples)
    print(f"[{time.strftime('%H:%M:%S')}] {group.name}: simple_evaluate({ {k: v for k, v in kw.items() if k != 'samples'} }"
          f"{', samples=loglik_samples.json' if samples else ''})", flush=True)
    t0 = time.time()
    rec: dict[str, object] = {"group": group.name, "kwargs": {k: v for k, v in kw.items() if k != "samples"},
                              "samples_file": "loglik_samples.json" if samples else None}
    try:
        res = simple_evaluate(model=lm, task_manager=tm, **kw)
    except Exception:
        rec.update(status="failed", wall_s=round(time.time() - t0, 1), error=traceback.format_exc())
        print(f"FAILED {model} {group.name}:\n{rec['error']}", flush=True)
        _free_cuda()
        return rec
    wall = time.time() - t0
    res.pop("samples", None)
    (gdir / "results.json").write_text(json.dumps(res, indent=1, default=handle_non_serializable))
    rec.update(status="ok", wall_s=round(wall, 1), results_file=f"{group.name}/results.json",
               tasks=summarize_results(res, group.tasks))
    rec["config"] = {k: res.get("config", {}).get(k) for k in ("batch_size", "batch_sizes", "model_dtype",
                                                                "random_seed", "numpy_seed", "torch_seed",
                                                                "fewshot_seed", "limit", "model_num_parameters")}
    rec["lm_eval_version"], rec["transformers_version"] = res.get("lm_eval_version"), res.get("transformers_version")
    rec["max_length"], rec["chat_template"] = res.get("max_length"), res.get("chat_template")
    missing = [t for t in group.tasks if t not in rec["tasks"]]
    if missing:
        rec["status"], rec["missing_tasks"] = "incomplete", missing
    (gdir / "done.json").write_text(json.dumps(rec, indent=1, default=handle_non_serializable))
    for line in task_lines(model, group.name, rec["tasks"], wall):
        print(line, flush=True)
    return rec


def set_neutral_generation_config(lm: object) -> tuple[dict[str, object], dict[str, object]]:
    from transformers import GenerationConfig
    kept, dropped = neutral_generation_config(lm.model.generation_config.to_diff_dict())
    lm.model.generation_config = GenerationConfig(**kept)
    return kept, dropped


def _free_cuda() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception as e:  # report, never fail the run on it
        print(f"empty_cache failed: {e}", flush=True)


def cmd_run(args: argparse.Namespace) -> int:
    model, folder = run_target(args.index, args.model_path, args.label)
    root = Path(args.out) / folder
    root.mkdir(parents=True, exist_ok=True)
    wanted = [g for g in GROUPS if g.name in args.groups.split(",")]
    print(f"exp04 index {args.index}: {model} on {platform.node()} gpu={gpu_name()} {env_versions()}"
          f"{f' tokenizer={args.tokenizer}' if args.tokenizer else ''}", flush=True)
    todo = [g for g in wanted if not (root / g.name / "done.json").exists()]
    gen_kept: dict[str, object] = {}
    gen_dropped: dict[str, object] = {}
    chat_info: dict[str, object] = {}
    recs = {g.name: json.loads((root / g.name / "done.json").read_text()) for g in wanted if g not in todo}
    for name, rec in recs.items():
        print(f"skip {name}: done.json exists ({rec.get('status')})", flush=True)
    if todo:
        t0 = time.time()
        tm = task_manager()
        print(f"[{time.strftime('%H:%M:%S')}] task index built in {time.time() - t0:.0f} s", flush=True)
        try:
            from lm_eval.models.huggingface import HFLM
            t0 = time.time()
            lm = HFLM(**model_kwargs(model, args.tokenizer))
            print(f"[{time.strftime('%H:%M:%S')}] model loaded in {time.time() - t0:.0f} s", flush=True)
            gen_kept, gen_dropped = set_neutral_generation_config(lm)
            print(f"generation config: kept {gen_kept}, dropped {gen_dropped}", flush=True)
            if any(g.chat for g in todo):
                template = getattr(lm.tokenizer, "chat_template", None)
                lm.chat_template_args = {**(lm.chat_template_args or {}), **thinking_switch(template)}
                chat_info = {"has_chat_template": bool(template), "chat_template_args": lm.chat_template_args}
                print(f"chat: {chat_info}", flush=True)
        except Exception:
            err = traceback.format_exc()
            print(f"FAILED {model} model load:\n{err}", flush=True)
            recs.update({g.name: {"group": g.name, "status": "failed", "error": err} for g in todo})
            lm = None
        if lm is not None:
            for g in todo:
                recs[g.name] = run_group(lm, tm, model, g, root)
    summary = {"model": model, "index": args.index, "host": platform.node(), "gpu": gpu_name(),
               "versions": env_versions(), "model_kwargs": model_kwargs(model, args.tokenizer),
               "generation_config_kept": gen_kept, "generation_config_dropped": gen_dropped, "chat": chat_info,
               "groups": {g.name: recs[g.name] for g in wanted}}
    (root / summary_name(wanted)).write_text(json.dumps(summary, indent=1, default=str))
    bad = [g.name for g in wanted if recs[g.name].get("status") != "ok"]
    print(f"exp04 index {args.index} {model}: groups ok {[g.name for g in wanted if g.name not in bad]}, "
          f"failed {bad}", flush=True)
    return 1 if bad else 0


def cmd_prefetch(args: argparse.Namespace) -> int:
    """Build every task once (downloads its dataset into HF_HOME) and the ifeval nltk data. No model, no GPU."""
    from lm_eval.tasks import get_task_dict
    names = [t for g in GROUPS for t in g.tasks]
    tasks = get_task_dict(names, task_manager())
    print(f"prefetched {len(tasks)} task entries for {names}", flush=True)
    import lm_eval.tasks.ifeval.instructions_util  # noqa: F401  (downloads nltk punkt_tab on import)
    print("nltk punkt_tab ok; versions", env_versions(), flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--index", type=int, default=None, help="Slurm array index into cilab.stages.STAGES")
    r.add_argument("--model-path", default=None, help="exp06: local checkpoint directory or hub id, instead of --index")
    r.add_argument("--tokenizer", default=None, help="exp06: tokenizer (and chat template) to load instead of the model's")
    r.add_argument("--label", default=None, help="exp06: output folder name for --model-path")
    r.add_argument("--out", required=True, help="results root, e.g. $CI_OUT/results")
    r.add_argument("--groups", default=DEFAULT_GROUPS)
    sub.add_parser("prefetch")
    args = ap.parse_args()
    sys.exit({"run": cmd_run, "prefetch": cmd_prefetch}[args.cmd](args))


if __name__ == "__main__":
    main()

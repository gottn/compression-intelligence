"""exp04 driver, run INSIDE the AAU jobs (never on the Mac): lm-evaluation-harness on one checkpoint.

    python exp04_benchmarks/run_eval.py prefetch                      # setup job: datasets + nltk data
    python exp04_benchmarks/run_eval.py run --index I --out DIR [--groups loglik,gsm8k,ifeval]

`run` maps the Slurm array index to a model (cilab.stages.model_at) and calls lm_eval once per task group:
  loglik  arc_easy, arc_challenge, hellaswag, piqa, winogrande, mmlu   (log-likelihood, harness few-shot defaults)
  gsm8k   gsm8k, 5-shot (task default), first 300 test problems        (generative)
  ifeval  ifeval, 0-shot, first 200 prompts                            (generative)
Every call: HF backend, fp16, batch_size auto, max_length 2048, NO chat template (base and post-trained models
see identical prompts), --seed 0,1234,1234,1234, no --log_samples.

`--limit` is global to one lm_eval call, so the loglik group passes the brief's per-task limits as `--samples`
(a JSON file of doc indices): hellaswag docs 0..1999 and docs 0..39 of each of the 57 MMLU subjects; tasks
not in the file run in full. lm-eval selects `--limit N` as docs 0..N-1, so this is the same selection.
max_length 2048: lm-eval's auto batch size for generation probes a forward pass at the model's max length
(131072 for Qwen2.5-1.5B), which would force batch 1; 2048 fits every prompt here (ifeval: ctx + 1280 gen).

Outputs under DIR/<model slug>/: <group>/<sanitized model>/results_*.json (lm-eval), <group>/done.json,
loglik_samples.json, summary.json. A group with a done.json is skipped (resumable after a Slurm requeue).
A failed group does not stop the others; the process exits 1 at the end if any group failed.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
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
SEED_ARG = "0,1234,1234,1234"  # python random, numpy, torch, few-shot sampler
MAX_LENGTH = 2048  # tokens


@dataclass(frozen=True)
class Group:
    name: str
    tasks: tuple[str, ...]
    limit: int | None  # docs per task; None = full task (or the samples file, for loglik)


GROUPS: tuple[Group, ...] = (
    Group("loglik", ("arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "mmlu"), None),
    Group("gsm8k", ("gsm8k",), 300),
    Group("ifeval", ("ifeval",), 200),
)
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


def lm_eval_argv(python: str, model: str, group: Group, out_dir: str, samples_path: str | None) -> list[str]:
    """The lm_eval command line of one task group (identical prompting for every model: no chat template)."""
    argv = [python, "-m", "lm_eval", "run", "--model", "hf",
            "--model_args", f"pretrained={model},dtype=float16,max_length={MAX_LENGTH}",
            "--tasks", ",".join(group.tasks), "--batch_size", "auto", "--device", "cuda",
            "--seed", SEED_ARG, "--output_path", out_dir]
    if group.limit is not None:
        argv += ["--limit", str(group.limit)]
    elif samples_path is not None:
        argv += ["--samples", samples_path]
    return argv


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


def newest_results(paths: Sequence[Path], since: float, mtimes: Mapping[Path, float]) -> Path | None:
    """The newest results_*.json written at or after `since` (epoch s), or None."""
    fresh = [p for p in paths if mtimes[p] >= since]
    return max(fresh, key=lambda p: (mtimes[p], p.name)) if fresh else None


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


def run_group(model: str, group: Group, root: Path) -> dict[str, object]:
    gdir = root / group.name
    gdir.mkdir(parents=True, exist_ok=True)
    samples = None
    if group.name == "loglik":
        samples = root / "loglik_samples.json"
        samples.write_text(json.dumps(samples_spec()))
    argv = lm_eval_argv(sys.executable, model, group, str(gdir), str(samples) if samples else None)
    print(f"[{time.strftime('%H:%M:%S')}] {group.name}: {' '.join(argv)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(argv, check=False).returncode  # streams into the Slurm log
    wall = time.time() - t0
    paths = sorted(gdir.glob("**/results_*.json"))
    found = newest_results(paths, t0 - 1, {p: p.stat().st_mtime for p in paths})
    rec = {"group": group.name, "returncode": rc, "wall_s": round(wall, 1), "argv": argv[1:],
           "results_file": str(found.relative_to(root)) if found else None}
    if rc == 0 and found is not None:
        res = json.loads(found.read_text())
        rec["status"] = "ok"
        rec["tasks"] = summarize_results(res, group.tasks)
        rec["config"] = {k: res.get("config", {}).get(k) for k in ("batch_size", "batch_sizes", "model_dtype",
                                                                    "random_seed", "numpy_seed", "torch_seed",
                                                                    "fewshot_seed", "limit", "model_num_parameters")}
        rec["lm_eval_version"], rec["transformers_version"] = res.get("lm_eval_version"), res.get("transformers_version")
        rec["max_length"], rec["chat_template"] = res.get("max_length"), res.get("chat_template")
        missing = [t for t in group.tasks if t not in rec["tasks"]]
        if missing:
            rec["status"], rec["missing_tasks"] = "incomplete", missing
        (gdir / "done.json").write_text(json.dumps(rec, indent=1))
        for line in task_lines(model, group.name, rec["tasks"], wall):
            print(line, flush=True)
    else:
        rec["status"] = "failed"
        print(f"FAILED {model} {group.name}: returncode {rc}, results file {found}", flush=True)
    return rec


def cmd_run(args: argparse.Namespace) -> int:
    model = model_at(args.index)
    root = Path(args.out) / model_slug(model)
    root.mkdir(parents=True, exist_ok=True)
    wanted = [g for g in GROUPS if g.name in args.groups.split(",")]
    print(f"exp04 index {args.index}: {model} on {platform.node()} gpu={gpu_name()} {env_versions()}", flush=True)
    recs = []
    for g in wanted:
        done = root / g.name / "done.json"
        if done.exists():
            rec = json.loads(done.read_text())
            print(f"skip {g.name}: done.json exists ({rec.get('status')})", flush=True)
        else:
            rec = run_group(model, g, root)
        recs.append(rec)
    summary = {"model": model, "index": args.index, "host": platform.node(), "gpu": gpu_name(),
               "versions": env_versions(), "groups": {r["group"]: r for r in recs}}
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    bad = [r["group"] for r in recs if r.get("status") != "ok"]
    print(f"exp04 index {args.index} {model}: groups ok "
          f"{[r['group'] for r in recs if r.get('status') == 'ok']}, failed {bad}", flush=True)
    return 1 if bad else 0


def cmd_prefetch(args: argparse.Namespace) -> int:
    """Build every task once (downloads its dataset into HF_HOME) and the ifeval nltk data. No model, no GPU."""
    from lm_eval.tasks import TaskManager, get_task_dict
    names = [t for g in GROUPS for t in g.tasks]
    tasks = get_task_dict(names, TaskManager())
    print(f"prefetched {len(tasks)} task entries for {names}", flush=True)
    import lm_eval.tasks.ifeval.instructions_util  # noqa: F401  (downloads nltk punkt_tab on import)
    print("nltk punkt_tab ok; versions", env_versions(), flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--index", type=int, required=True, help="Slurm array index into cilab.stages.STAGES")
    r.add_argument("--out", required=True, help="results root, e.g. $CI_OUT/results")
    r.add_argument("--groups", default=",".join(g.name for g in GROUPS))
    sub.add_parser("prefetch")
    args = ap.parse_args()
    sys.exit({"run": cmd_run, "prefetch": cmd_prefetch}[args.cmd](args))


if __name__ == "__main__":
    main()

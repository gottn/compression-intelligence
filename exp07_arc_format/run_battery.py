"""exp07 driver, run INSIDE the AAU eval jobs (never on the Mac or the front end): the ARC format battery on one
checkpoint of a checkpoints list.

    python exp07_arc_format/run_battery.py run --list FILE --index I --out DIR [--limit N] [--groups G,...]
    python exp07_arc_format/run_battery.py public-list > exp07_arc_format/checkpoints_public.tsv     (Mac; committed)
    python exp07_arc_format/run_battery.py prefetch                    (CPU job; only if a dataset is not cached)

The list has one tab-separated line per checkpoint (cilab.sft.checkpoint_line: label, path, recipe, step, marks):
the 18 public rows of cilab.stages (recipe "public", the hub id as path) first, then the lines exp07's training job
appends (recipe tulu or mathcot, a local checkpoint directory). Line I+1 is evaluated; an index past the end exits 0.

Groups (definitions in RESULTS.md), run in this order, one lm_eval.simple_evaluate call each, the model loaded once:
  b1_arc_0shot        arc_easy, arc_challenge   0-shot, no template (exp04's default)          acc, acc_norm
  b4_mmlu_letter      mmlu                      0-shot letter choice, 40 per subject (exp04)   acc
  b4_mmlu_cont        mmlu_continuation         0-shot answer-text continuation, same 2,280    acc
  b5_hellaswag        hellaswag                 0-shot, first 2,000 validation docs (exp04)    acc, acc_norm
  b0_piqa_wino        piqa, winogrande          0-shot (exp04); retrained checkpoints only     acc(_norm)
  b3_arc_0shot_chat   arc_easy, arc_challenge   0-shot with a chat template (apply_chat_template)
  b2_arc_25shot       arc_easy, arc_challenge   25-shot, shots from the train split (seed 1234)
B3's template: post-trained and retrained checkpoints use their own tokenizer and template; a base runs B3 once per
post-trained model of its family, with that model's tokenizer and template (folder b3_arc_0shot_chat__<model slug>),
so every B3 comparison sees identical prompt tokens. Templates with `enable_thinking` get it False (Qwen3, as exp04).
Harness settings are exp04's (cilab.lmeval): fp16, batch_size auto, max_length 2048, seeds 0/1234/1234/1234, greedy
generation config. New: per-item outcomes are logged (log_samples) and kept compactly in <group>/items.json
(doc ids and 0/1 strings of acc and acc_norm per task), with a count of contexts longer than max_length (the harness
left-truncates those).

Outputs under DIR/<label>/: <group>/{results.json, items.json, done.json}, summary.json. A group with a done.json
is skipped (resumable after a requeue or a timeout); a failed group does not stop the others; exit 1 if any failed.
--limit N (dry runs) evaluates the first N docs of every task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cilab`

from cilab import lmeval
from cilab.lmeval import MAX_LENGTH, SEEDS, model_kwargs, samples_spec, summarize_results, thinking_switch
from cilab.pairs import model_slug
from cilab.sft import checkpoint_line, parse_checkpoint_line
from cilab.stages import STAGES, Stage, family_base, stage_of

TASK_DIRS: tuple[str, ...] = ("arc", "hellaswag", "piqa", "winogrande", "mmlu/default", "mmlu/continuation")
PUBLIC = "public"  # recipe field of the public rows
RETRAINED_RECIPES: tuple[str, ...] = ("tulu", "smoltalk", "mathcot")
RETRAINED_BASE = "allenai/OLMo-2-0425-1B"
RETRAINED_TEMPLATE = "allenai/OLMo-2-0425-1B-Instruct"  # the template the exp06/exp07 checkpoints carry
ITEM_METRICS: tuple[str, ...] = ("acc", "acc_norm")
ARC_SHOTS = 25


@dataclass(frozen=True)
class BatteryGroup:
    name: str
    tasks: tuple[str, ...]
    num_fewshot: int | None  # None = the task default (0 for every task here)
    chat: bool = False
    samples: str | None = None  # "hellaswag", "mmlu" or "mmlu_continuation": exp04's per-task doc selection
    retrained_only: bool = False


GROUPS: tuple[BatteryGroup, ...] = (
    BatteryGroup("b1_arc_0shot", ("arc_easy", "arc_challenge"), None),
    BatteryGroup("b4_mmlu_letter", ("mmlu",), None, samples="mmlu"),
    BatteryGroup("b4_mmlu_cont", ("mmlu_continuation",), None, samples="mmlu_continuation"),
    BatteryGroup("b5_hellaswag", ("hellaswag",), None, samples="hellaswag"),
    BatteryGroup("b0_piqa_wino", ("piqa", "winogrande"), None, retrained_only=True),
    BatteryGroup("b3_arc_0shot_chat", ("arc_easy", "arc_challenge"), None, chat=True),
    BatteryGroup("b2_arc_25shot", ("arc_easy", "arc_challenge"), ARC_SHOTS),
)
CHAT_GROUP = "b3_arc_0shot_chat"
HEADLINE: dict[str, tuple[str, ...]] = {"arc_easy": ("acc_norm,none", "acc,none"),
                                        "arc_challenge": ("acc_norm,none", "acc,none"), "mmlu": ("acc,none",),
                                        "mmlu_continuation": ("acc,none",), "hellaswag": ("acc_norm,none", "acc,none"),
                                        "piqa": ("acc_norm,none",), "winogrande": ("acc,none",)}


@dataclass(frozen=True)
class Entry:
    label: str  # output folder
    model: str  # hub id or local checkpoint directory
    recipe: str  # "public", or the retrained recipe
    step: int
    marks: str
    base: str  # family base (the reference)
    is_base: bool


@dataclass(frozen=True)
class Run:
    folder: str  # output folder of this group run, under DIR/<label>/
    group: BatteryGroup
    tokenizer: str | None  # tokenizer (and template) to evaluate with; None = the checkpoint's own


# ----------------------------------------------------------------------------- pure
def group_named(name: str, groups: tuple[BatteryGroup, ...] = GROUPS) -> BatteryGroup:
    hits = [g for g in groups if g.name == name]
    if len(hits) != 1:
        raise KeyError(name)
    return hits[0]


def continuation_name(task: str) -> str:
    """'mmlu_anatomy' -> 'mmlu_anatomy_continuation' (lm-eval's mmlu/continuation leaf names)."""
    return f"{task}_continuation"


def group_samples(group: BatteryGroup, limit: int | None = None) -> dict[str, list[int]] | None:
    """lm-eval `samples` (doc indices per leaf task) of a group: exp04's spec for hellaswag and MMLU (the continuation
    format gets the same doc indices under its own task names); with `limit`, the first `limit` of each."""
    if group.samples is None:
        return None
    spec = samples_spec()
    if group.samples == "hellaswag":
        out = {"hellaswag": spec["hellaswag"]}
    elif group.samples == "mmlu":
        out = {k: v for k, v in spec.items() if k.startswith("mmlu_")}
    elif group.samples == "mmlu_continuation":
        out = {continuation_name(k): v for k, v in spec.items() if k.startswith("mmlu_")}
    else:
        raise ValueError(f"unknown samples spec {group.samples}")
    return {k: v[:limit] if limit is not None else list(v) for k, v in sorted(out.items())}


def eval_kwargs(group: BatteryGroup, limit: int | None = None) -> dict[str, object]:
    """simple_evaluate arguments of one group: exp04's seeds, per-item logging on, the group's shots and template."""
    kw: dict[str, object] = {"tasks": list(group.tasks), "log_samples": True, "apply_chat_template": group.chat,
                             "fewshot_as_multiturn": group.chat, **SEEDS}
    if group.num_fewshot is not None:
        kw["num_fewshot"] = group.num_fewshot
    samples = group_samples(group, limit)
    if samples is not None:
        kw["samples"] = samples
    elif limit is not None:
        kw["limit"] = limit
    return kw


def public_lines(stages: tuple[Stage, ...] = STAGES) -> list[str]:
    """The 18 public rows of the checkpoints list, in stage-table order (label = exp04's folder name)."""
    return [checkpoint_line(model_slug(s.model), s.model, PUBLIC, 0, []) for s in stages]


def attempt_suffix(path: str) -> str:
    """'__attempt2' when a checkpoint comes from a Slurm requeue's <recipe>-attempt2/ directory, else ''."""
    m = re.search(r"/[a-z]+-attempt(\d+)/ckpt/", path)
    return f"__attempt{m.group(1)}" if m else ""


def parse_entry(line: str, stages: tuple[Stage, ...] = STAGES) -> Entry:
    """A checkpoints-list line -> Entry. Public rows take base/post from the stage table; retrained rows are
    OLMo-2-0425-1B fine-tunes."""
    c = parse_checkpoint_line(line)
    if c["recipe"] == PUBLIC:
        st = stage_of(c["path"], stages)
        return Entry(c["label"], c["path"], PUBLIC, int(c["step"]), c["marks"], family_base(st.family, stages),
                     st.parent is None)
    if c["recipe"] not in RETRAINED_RECIPES:
        raise ValueError(f"unknown recipe {c['recipe']!r} in {line!r}")
    return Entry(c["label"] + attempt_suffix(c["path"]), c["path"], c["recipe"], int(c["step"]), c["marks"],
                 RETRAINED_BASE, False)


def template_partners(base: str, stages: tuple[Stage, ...] = STAGES) -> list[str]:
    """Post-trained models of a base's family, in table order: the tokenizers (and templates) the base runs B3 with."""
    fam = stage_of(base, stages).family
    return [s.model for s in stages if s.family == fam and s.parent is not None]


def chat_folder(partner: str | None) -> str:
    return CHAT_GROUP if partner is None else f"{CHAT_GROUP}__{model_slug(partner)}"


def plan(entry: Entry, stages: tuple[Stage, ...] = STAGES,
         groups: tuple[BatteryGroup, ...] = GROUPS) -> list[Run]:
    """Group runs of one checkpoint in execution order."""
    runs: list[Run] = []
    for g in groups:
        if g.retrained_only and entry.recipe == PUBLIC:
            continue
        if g.chat and entry.is_base:
            runs += [Run(chat_folder(p), g, p) for p in template_partners(entry.model, stages)]
        else:
            runs.append(Run(chat_folder(None) if g.chat else g.name, g, None))
    return runs


def reference_chat_folder(entry: Entry) -> str:
    """The base's B3 folder that a post-trained or retrained checkpoint is compared with."""
    if entry.is_base:
        raise ValueError("a base has no B3 reference")
    return chat_folder(RETRAINED_TEMPLATE if entry.recipe != PUBLIC else entry.model)


def bits(values: Sequence[float]) -> str:
    """0/1 metric values as a compact string ('0110...'); anything else raises."""
    out = []
    for v in values:
        if v not in (0, 1):
            raise ValueError(f"not a 0/1 outcome: {v}")
        out.append("1" if v else "0")
    return "".join(out)


def item_outcomes(samples: Mapping[str, Sequence[Mapping[str, object]]],
                  metrics: Sequence[str] = ITEM_METRICS) -> dict[str, dict[str, object]]:
    """{leaf task: {doc_id: [...], <metric>: '0101...'}} from lm-eval's logged samples, sorted by doc id; metrics a
    task does not report are left out."""
    out: dict[str, dict[str, object]] = {}
    for task in sorted(samples):
        ex = sorted(samples[task], key=lambda e: int(e["doc_id"]))
        rec: dict[str, object] = {"doc_id": [int(e["doc_id"]) for e in ex]}
        for m in metrics:
            if ex and all(m in e for e in ex):
                rec[m] = bits([float(e[m]) for e in ex])
        out[task] = rec
    return out


def longest_request(example: Mapping[str, object]) -> str:
    """Context + continuation of a logged sample's longest request (multiple choice: one request per choice)."""
    texts = ["".join(str(a) for a in args) for args in example["arguments"]]
    return max(texts, key=len)


def truncation_counts(samples: Mapping[str, Sequence[Mapping[str, object]]], n_tokens: Callable[[str], int],
                      max_len: int) -> dict[str, dict[str, int]]:
    """Per leaf task: docs, docs whose longest request exceeds max_len tokens (left-truncated by the harness), and
    the longest request in tokens."""
    out = {}
    for task in sorted(samples):
        lens = [n_tokens(longest_request(e)) for e in samples[task]]
        out[task] = {"n_docs": len(lens), "n_over": sum(n > max_len for n in lens), "max_tokens": max(lens, default=0)}
    return out


def md5(text: str | None) -> str | None:
    return hashlib.md5(text.encode()).hexdigest() if text else None


def task_lines(label: str, folder: str, summary: Mapping[str, Mapping[str, object]], wall_s: float) -> list[str]:
    lines = []
    for t, s in summary.items():
        m = s["metrics"]
        cells = [f"{k.split(',')[0]}={m[k]:.4f}±{m.get(k.replace(',', '_stderr,', 1), float('nan')):.4f}"
                 for k in HEADLINE.get(t, ()) if k in m]
        lines.append(f"SCORE {label} {folder} {t}: " + " ".join(cells) + f" n={s['n_effective']} shot={s['n_shot']} "
                     f"({wall_s:.0f} s)")
    return lines


def list_lines(text: str) -> list[str]:
    """Non-blank lines of a checkpoints list, in file order (the eval array index is a position in this list)."""
    return [ln for ln in text.splitlines(keepends=True) if ln.strip()]


# ----------------------------------------------------------------------------- effects
def read_entry(list_path: Path, index: int) -> Entry | None:
    lines = list_lines(list_path.read_text())
    return parse_entry(lines[index]) if 0 <= index < len(lines) else None


def chat_lm(lm: object, tokenizer: str | None) -> object:
    """The HFLM to run B3 with: the checkpoint's own (tokenizer None), or the same weights with another model's
    tokenizer and template. Thinking is switched off where the template has the switch."""
    from lm_eval.models.huggingface import HFLM
    out = lm if tokenizer is None else HFLM(pretrained=lm.model, tokenizer=tokenizer, max_length=MAX_LENGTH,
                                             batch_size="auto")
    template = getattr(out.tokenizer, "chat_template", None)
    out.chat_template_args = {**(out.chat_template_args or {}), **thinking_switch(template)}
    return out


def run_group(lm: object, tm: object, entry: Entry, run: Run, root: Path, limit: int | None) -> dict[str, object]:
    from lm_eval import simple_evaluate
    from lm_eval.utils import handle_non_serializable
    gdir = root / run.folder
    gdir.mkdir(parents=True, exist_ok=True)
    kw = eval_kwargs(run.group, limit)
    shown = {k: v for k, v in kw.items() if k != "samples"}
    print(f"[{time.strftime('%H:%M:%S')}] {run.folder}: simple_evaluate({shown}"
          f"{', samples=<exp04 spec>' if 'samples' in kw else ''}) tokenizer={run.tokenizer or 'own'}", flush=True)
    rec: dict[str, object] = {"group": run.group.name, "folder": run.folder, "kwargs": shown,
                              "samples": {k: len(v) for k, v in kw.get("samples", {}).items()} or None,
                              "tokenizer": run.tokenizer or "own"}
    t0 = time.time()
    try:
        use = chat_lm(lm, run.tokenizer) if run.group.chat else lm
        if run.group.chat:
            rec["chat_template_args"] = use.chat_template_args
            rec["template_md5"] = md5(getattr(use.tokenizer, "chat_template", None))
        res = simple_evaluate(model=use, task_manager=tm, **kw)
        logged = res.pop("samples", {}) or {}
        items = item_outcomes(logged)
        trunc = truncation_counts(logged, lambda s: len(use.tok_encode(s)), MAX_LENGTH)
    except Exception:
        rec.update(status="failed", wall_s=round(time.time() - t0, 1), error=traceback.format_exc())
        print(f"FAILED {entry.label} {run.folder}:\n{rec['error']}", flush=True)
        lmeval.free_cuda()
        return rec
    wall = time.time() - t0
    (gdir / "results.json").write_text(json.dumps(res, indent=1, default=handle_non_serializable))
    (gdir / "items.json").write_text(json.dumps(items))
    rec.update(status="ok", wall_s=round(wall, 1), tasks=summarize_results(res, run.group.tasks), truncation=trunc,
               lm_eval_version=res.get("lm_eval_version"), transformers_version=res.get("transformers_version"),
               max_length=res.get("max_length"), chat_template_md5=md5(res.get("chat_template")),
               config={k: res.get("config", {}).get(k) for k in ("batch_size", "batch_sizes", "model_dtype",
                                                                 "random_seed", "numpy_seed", "torch_seed",
                                                                 "fewshot_seed", "limit")})
    missing = [t for t in run.group.tasks if t not in rec["tasks"]]
    if missing:
        rec["status"], rec["missing_tasks"] = "incomplete", missing
    (gdir / "done.json").write_text(json.dumps(rec, indent=1, default=handle_non_serializable))
    for line in task_lines(entry.label, run.folder, rec["tasks"], wall):
        print(line, flush=True)
    over = {t: v["n_over"] for t, v in trunc.items() if v["n_over"]}
    if over:
        print(f"TRUNCATED contexts > {MAX_LENGTH} tokens in {run.folder}: {over}", flush=True)
    return rec


def cmd_run(args: argparse.Namespace) -> int:
    entry = read_entry(Path(args.list), args.index)
    if entry is None:
        print(f"no checkpoint at index {args.index} in {args.list}: nothing to evaluate", flush=True)
        return 0
    runs = plan(entry)
    if args.groups:
        wanted = set(args.groups.split(","))
        runs = [r for r in runs if r.group.name in wanted]
        if "b0_piqa_wino" in wanted and entry.recipe == PUBLIC:  # dry runs exercise B0 on a public checkpoint
            runs.insert(0, Run("b0_piqa_wino", group_named("b0_piqa_wino"), None))
    root = Path(args.out) / entry.label
    root.mkdir(parents=True, exist_ok=True)
    print(f"exp07 index {args.index}: {entry} on {platform.node()} gpu={lmeval.gpu_name()} {lmeval.env_versions()}; "
          f"runs {[r.folder for r in runs]}", flush=True)
    todo = [r for r in runs if not (root / r.folder / "done.json").exists()]
    recs = {r.folder: json.loads((root / r.folder / "done.json").read_text()) for r in runs if r not in todo}
    for name, rec in recs.items():
        print(f"skip {name}: done.json exists ({rec.get('status')})", flush=True)
    gen_kept: dict[str, object] = {}
    gen_dropped: dict[str, object] = {}
    if todo:
        t0 = time.time()
        tm = lmeval.task_manager(TASK_DIRS)
        print(f"[{time.strftime('%H:%M:%S')}] task index built in {time.time() - t0:.0f} s", flush=True)
        try:
            from lm_eval.models.huggingface import HFLM
            t0 = time.time()
            lm = HFLM(**model_kwargs(entry.model))
            print(f"[{time.strftime('%H:%M:%S')}] model loaded in {time.time() - t0:.0f} s", flush=True)
            gen_kept, gen_dropped = lmeval.set_neutral_generation_config(lm)
        except Exception:
            err = traceback.format_exc()
            print(f"FAILED {entry.label} model load:\n{err}", flush=True)
            recs.update({r.folder: {"group": r.group.name, "status": "failed", "error": err} for r in todo})
            lm = None
        if lm is not None:
            for r in todo:
                recs[r.folder] = run_group(lm, tm, entry, r, root, args.limit)
    summary = {"entry": asdict(entry), "index": args.index, "list": args.list, "host": platform.node(),
               "gpu": lmeval.gpu_name(), "versions": lmeval.env_versions(), "model_kwargs": model_kwargs(entry.model),
               "limit": args.limit, "generation_config_kept": gen_kept, "generation_config_dropped": gen_dropped,
               "runs": {r.folder: recs[r.folder] for r in runs}}
    (root / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    bad = [r.folder for r in runs if recs[r.folder].get("status") != "ok"]
    print(f"exp07 index {args.index} {entry.label}: ok {[r.folder for r in runs if r.folder not in bad]}, "
          f"failed {bad}", flush=True)
    return 1 if bad else 0


def cmd_public_list(args: argparse.Namespace) -> int:
    sys.stdout.write("".join(public_lines()))
    return 0


def cmd_prefetch(args: argparse.Namespace) -> int:
    """Build every battery task once (downloads its dataset into HF_HOME). No model, no GPU."""
    from lm_eval.tasks import get_task_dict
    names = sorted({t for g in GROUPS for t in g.tasks})
    tasks = get_task_dict(names, lmeval.task_manager(TASK_DIRS))
    print(f"prefetched {len(tasks)} task entries for {names}; versions {lmeval.env_versions()}", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--list", required=True, help="checkpoints list (public rows, then the training job's lines)")
    r.add_argument("--index", type=int, required=True, help="0-based line of the list (the Slurm array index)")
    r.add_argument("--out", required=True, help="results root, e.g. $CI_OUT/results")
    r.add_argument("--limit", type=int, default=None, help="dry runs: first N docs of every task")
    r.add_argument("--groups", default=None, help="dry runs: only these groups (comma-separated names)")
    sub.add_parser("public-list")
    sub.add_parser("prefetch")
    args = ap.parse_args()
    sys.exit({"run": cmd_run, "public-list": cmd_public_list, "prefetch": cmd_prefetch}[args.cmd](args))


if __name__ == "__main__":
    main()

"""exp06 data, run INSIDE the AAU CPU setup job (never on the front end): download, format, tokenize and select the
token-matched SFT subsets of the three recipes.

    python exp06_matched_drift/prep_data.py --out DIR [--recipes tulu smoltalk mathcot] [--budget 45000000]

Per recipe (definitions in RESULTS.md; helpers in cilab/sft.py):
  1. candidates in the order numpy.random.default_rng(0).permutation(rows), in chunks;
  2. each `messages` list is dropped for a cilab.sft.drop_reason, else rendered with the OLMo-2-0425-1B-Instruct chat
     template and tokenized once with character offsets; the loss mask marks tokens that start inside an assistant
     turn (reply + closing <|endoftext|>); examples over 2,048 tokens are dropped, never truncated;
  3. fitting examples are kept until they hold `budget` template tokens (the one that crosses it is kept).
If a recipe's fitting pool runs out first, the budget for every recipe becomes that pool's total: recipes are prepared
in the order given, a short pool lowers the budget, and recipes already prepared are cut to the new budget (they are a
prefix of the same seeded order, so the cut is exact).
Datasets are loaded at the revisions pinned in RECIPES (the ones exp06 recorded).
Writes DIR/<recipe>.npz (ids int32, mask bool, offsets int64 of each example in the flat arrays, source index of each
example), DIR/<recipe>.prov.json (dataset id, config, revision, counts, drop reasons, token stats, source mix) and
DIR/summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `cilab`

import numpy as np

from cilab.sft import assistant_spans, drop_reason, fits, loss_mask, shuffled_order, take_until_tokens

TEMPLATE_TOKENIZER = "allenai/OLMo-2-0425-1B-Instruct"
MAX_LEN = 2048  # tokens
BUDGET = 45_000_000  # template tokens per recipe (RESULTS.md: D)
SEED = 0
CHUNK = 4096  # candidates per tokenizer batch
RESERVED: tuple[str, ...] = ("<|endoftext|>", "<|pad|>")


@dataclass(frozen=True)
class Recipe:
    name: str
    dataset: str
    config: str | None
    split: str
    source_cols: tuple[str, ...]  # columns tallied for the source mix
    revision: str | None = None  # hub commit; None resolves the current head (exp06's setup did, on 2026-10-15)


# Revisions pinned to the ones exp06's setup job recorded (logs/setup/data/*.prov.json), so a rerun (exp07) makes the
# same subsets.
RECIPES: tuple[Recipe, ...] = (
    Recipe("tulu", "allenai/tulu-3-sft-mixture", None, "train", ("source",),
           "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"),
    Recipe("smoltalk", "HuggingFaceTB/smoltalk", "all", "train", ("source",),
           "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"),
    Recipe("mathcot", "open-r1/OpenR1-Math-220k", "all", "train", ("source", "problem_type"),
           "e4e141ec9dea9f8326f4d347be56105859b2bd68"),
)


# ----------------------------------------------------------------------------- pure
def recipe_named(name: str, recipes: tuple[Recipe, ...] = RECIPES) -> Recipe:
    hits = [r for r in recipes if r.name == name]
    if len(hits) != 1:
        raise KeyError(name)
    return hits[0]


def length_stats(lengths: Sequence[int]) -> dict[str, float]:
    a = np.asarray(lengths, dtype=np.int64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "sum": int(a.sum()), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)), "max": int(a.max())}


# ----------------------------------------------------------------------------- effects
def make_render(tok: object):
    def render(messages: Sequence[Mapping[str, str]], gen: bool) -> str:
        return tok.apply_chat_template([dict(m) for m in messages], tokenize=False, add_generation_prompt=gen)
    return render


def load_recipe(recipe: Recipe) -> tuple[object, str, float]:
    """(dataset split, hub revision it was loaded at, load wall s); downloads into HF_HOME. A recipe without a pinned
    revision is loaded at the hub's current head."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    t0 = time.time()
    revision = recipe.revision or HfApi().dataset_info(recipe.dataset).sha
    ds = load_dataset(recipe.dataset, recipe.config, split=recipe.split, revision=revision)
    return ds, revision, time.time() - t0


def prepare(recipe: Recipe, ds: object, revision: str, tok: object, budget: int) -> dict[str, object]:
    """Scan candidates of `ds` in seeded order until the kept examples reach `budget` tokens or the dataset ends."""
    t0 = time.time()
    cols = [c for c in recipe.source_cols if c in ds.column_names]
    ds = ds.select_columns(["messages", *cols])
    order = shuffled_order(len(ds), SEED)
    render = make_render(tok)
    drops: Counter[str] = Counter()
    ids_l, mask_l, src_idx, src_mix = [], [], [], []
    total, scanned, all_lengths = 0, 0, []
    print(json.dumps({"recipe": recipe.name, "rows": len(ds), "revision": revision}), flush=True)
    for lo in range(0, len(order), CHUNK):
        idx = order[lo:lo + CHUNK]
        batch = ds.select(idx.tolist())
        msgs = batch["messages"]
        srcs = list(zip(*(batch[c] for c in cols), strict=True)) if cols else [()] * len(idx)
        reasons: list[str | None] = []
        texts, spans = [], []
        for m in msgs:
            why = drop_reason(m, RESERVED)
            if why is None:
                try:
                    spans.append(assistant_spans(render, m))
                    texts.append(render(m, False))
                except ValueError:
                    why = "template_not_prefix"
            reasons.append(why)
        enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
        j = -1
        for k, why in enumerate(reasons):
            scanned = lo + k + 1
            if why is not None:
                drops[why] += 1
                continue
            j += 1
            n = len(enc["input_ids"][j])
            mask = loss_mask([tuple(o) for o in enc["offset_mapping"][j]], spans[j])
            all_lengths.append(n)
            if not fits(n, int(mask[1:].sum()), MAX_LEN):
                drops["too_long" if n > MAX_LEN else "no_target"] += 1
                continue
            ids_l.append(np.asarray(enc["input_ids"][j], dtype=np.int32))
            mask_l.append(mask)
            src_idx.append(int(idx[k]))
            src_mix.append(srcs[k])
            total += n
            if total >= budget:
                break
        if total >= budget:
            break
        if lo // CHUNK % 10 == 0:
            print(json.dumps({"recipe": recipe.name, "scanned": scanned, "kept": len(ids_l), "tokens": total,
                              "wall_s": round(time.time() - t0)}), flush=True)
    return {"recipe": recipe, "revision": revision, "rows": len(ds), "ids": ids_l, "mask": mask_l,
            "src_idx": src_idx, "src_mix": src_mix, "src_cols": cols, "drops": drops, "scanned": scanned,
            "rendered_lengths": all_lengths, "wall_s": time.time() - t0}


def write_recipe(p: Mapping[str, object], n_keep: int, budget: int, out: Path) -> dict[str, object]:
    """Write the first n_keep kept examples of a prepared recipe; returns its provenance record."""
    r: Recipe = p["recipe"]
    ids, mask = p["ids"][:n_keep], p["mask"][:n_keep]
    lengths = [len(x) for x in ids]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.savez(out / f"{r.name}.npz", ids=np.concatenate(ids).astype(np.int32), mask=np.concatenate(mask),
             offsets=offsets, source_index=np.asarray(p["src_idx"][:n_keep], dtype=np.int64))
    loss = [int(m[1:].sum()) for m in mask]
    mix = Counter("/".join(str(v) for v in s) for s in p["src_mix"][:n_keep])
    prov = {"recipe": r.name, "dataset": r.dataset, "config": r.config, "split": r.split,
            "revision": p["revision"], "rows": p["rows"], "seed": SEED, "order": "numpy default_rng(0).permutation",
            "template_tokenizer": TEMPLATE_TOKENIZER, "max_len": MAX_LEN, "budget_tokens": budget,
            "candidates_scanned": p["scanned"], "kept": n_keep, "kept_before_budget_cut": len(p["ids"]),
            "drops": dict(sorted(p["drops"].items())), "tokens": length_stats(lengths),
            "loss_tokens": length_stats(loss), "loss_token_share": sum(loss) / max(1, sum(lengths)),
            "rendered_tokens_scanned": length_stats(p["rendered_lengths"]),
            "fit_rate_scanned": float(np.mean(np.asarray(p["rendered_lengths"]) <= MAX_LEN))
            if p["rendered_lengths"] else None,
            "source_columns": p["src_cols"], "source_mix_top30": dict(mix.most_common(30)),
            "versions": {k: _version(k) for k in ("datasets", "transformers", "tokenizers")},
            "date": time.strftime("%Y-%m-%d"), "wall_s": round(p["wall_s"], 1)}
    (out / f"{r.name}.prov.json").write_text(json.dumps(prov, indent=1))
    return prov


def _version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def cmd_prep(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TEMPLATE_TOKENIZER)
    budget = args.budget
    prepared = {}
    for name in args.recipes:
        ds, revision, load_s = load_recipe(recipe_named(name))
        print(json.dumps({"recipe": name, "loaded_s": round(load_s)}), flush=True)
        p = prepare(recipe_named(name), ds, revision, tok, budget)
        pool = sum(len(x) for x in p["ids"])
        if pool < budget:
            print(json.dumps({"recipe": name, "short_pool_tokens": pool, "budget_lowered_from": budget}), flush=True)
            budget = pool
        prepared[name] = p
    provs = {}
    for name, p in prepared.items():
        n_keep = take_until_tokens([len(x) for x in p["ids"]], budget)
        provs[name] = write_recipe(p, n_keep, budget, out)
        print(json.dumps({k: provs[name][k] for k in ("recipe", "kept", "candidates_scanned", "drops", "tokens",
                                                      "loss_token_share")}), flush=True)
    (out / "summary.json").write_text(json.dumps({"budget_tokens": budget, "requested_budget": args.budget,
                                                  "recipes": provs}, indent=1))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipes", nargs="*", default=[r.name for r in RECIPES])
    ap.add_argument("--budget", type=int, default=BUDGET)
    raise SystemExit(cmd_prep(ap.parse_args()))


if __name__ == "__main__":
    main()

"""Supervised fine-tuning helpers (exp06): chat formatting with an assistant-only loss mask, example selection,
sequence packing, the learning-rate schedule and drift-mark checkpointing. Pure: the chat-template renderer, the
tokenizer's outputs, lengths and seeds are parameters; datasets, tokenizer calls and the training loop live in the
experiment scripts.

Loss rule: every example is rendered with one chat template; a token is a loss target when its character span starts
inside an assistant span = the characters a turn adds after the generation prompt (the reply and the end-of-sequence
token the template appends). Spans come from rendering prefixes of the conversation, so the rendered text must grow by
appending (true for the OLMo-2 template); anything else raises.

Packing rule: examples, in a seeded per-epoch order, fill rows of `capacity` tokens greedily and in order; an example
never splits, and a row closes when the next example does not fit. Position ids restart at 0 for every example and for
the padding tail, so the model (transformers' packed-sequence mask) attends only within an example.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np

Message = Mapping[str, str]
Render = Callable[[Sequence[Message], bool], str]  # (messages, add_generation_prompt) -> rendered text
Span = tuple[int, int]  # [start, end) in characters

ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})


# ----------------------------------------------------------------------------- formatting and loss mask
def drop_reason(messages: Sequence[Message], reserved: Sequence[str]) -> str | None:
    """Why an example cannot be formatted (None if it can): no assistant turn, an assistant turn first (no prompt to
    condition on), a role the template does not render, an empty or non-string content, or a reserved special-token
    string (e.g. the end-of-sequence text) inside a content."""
    if not messages:
        return "empty"
    roles = [m.get("role") for m in messages]
    if "assistant" not in roles:
        return "no_assistant"
    if any(r not in ROLES for r in roles):
        return "unknown_role"
    if next(r for r in roles if r != "system") == "assistant":
        return "assistant_first"
    contents = [m.get("content") for m in messages]
    if any(not isinstance(c, str) or not c.strip() for c in contents):
        return "empty_content"
    if any(s in c for c in contents for s in reserved):
        return "reserved_text"
    return None


def assistant_spans(render: Render, messages: Sequence[Message]) -> tuple[Span, ...]:
    """Character spans of every assistant turn in render(messages): from the end of the prefix rendered with the
    generation prompt to the end of the prefix that includes the turn. Raises unless both prefixes are string prefixes
    of the full rendering and the turn adds characters."""
    full = render(messages, False)
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        if i == 0:
            raise ValueError("assistant turn without a preceding prompt")
        pre, upto = render(messages[:i], True), render(messages[: i + 1], False)
        if not (full.startswith(pre) and full.startswith(upto) and len(upto) > len(pre)):
            raise ValueError(f"turn {i}: the rendering does not grow by appending")
        spans.append((len(pre), len(upto)))
    return tuple(spans)


def loss_mask(offsets: Sequence[Span], spans: Sequence[Span]) -> np.ndarray:
    """Per token, True when its character span starts inside an assistant span (a loss target)."""
    starts = np.asarray([s for s, _ in offsets], dtype=np.int64)
    out = np.zeros(len(starts), dtype=bool)
    for a, b in spans:
        out |= (starts >= a) & (starts < b)
    return out


def fits(n_tokens: int, n_loss: int, max_len: int) -> bool:
    """An example is kept when it fits in one row and has at least one loss target after its first token."""
    return 0 < n_tokens <= max_len and n_loss > 0


# ----------------------------------------------------------------------------- selection
def shuffled_order(n: int, seed: int) -> np.ndarray:
    """Candidate order: a permutation of range(n), a pure function of the seed."""
    return np.random.default_rng(seed).permutation(n)


def take_until_tokens(lengths: Sequence[int], budget: int) -> int:
    """Number of leading examples whose cumulative length first reaches `budget` tokens (all of them if never)."""
    c = np.cumsum(np.asarray(lengths, dtype=np.int64))
    hit = np.flatnonzero(c >= budget)
    return int(hit[0]) + 1 if hit.size else len(c)


# ----------------------------------------------------------------------------- packing
def pack_greedy(lengths: Sequence[int], capacity: int) -> list[tuple[int, ...]]:
    """Rows of positions into `lengths` (taken in order): each row is filled until the next example does not fit."""
    rows: list[tuple[int, ...]] = []
    cur: list[int] = []
    used = 0
    for i, n in enumerate(lengths):
        if not 0 < n <= capacity:
            raise ValueError(f"example {i} has {n} tokens, capacity {capacity}")
        if used + n > capacity:
            rows.append(tuple(cur))
            cur, used = [], 0
        cur.append(i)
        used += n
    if cur:
        rows.append(tuple(cur))
    return rows


def epoch_order(n: int, seed: int, epoch: int) -> np.ndarray:
    """Example order of one epoch: a permutation seeded by (seed, epoch)."""
    return np.random.default_rng([seed, epoch]).permutation(n)


def row_stream(lengths: Sequence[int], capacity: int, seed: int) -> Iterator[tuple[int, tuple[int, ...]]]:
    """Endless (epoch, example indices) rows: each epoch reshuffles every example and packs it greedily."""
    lens = np.asarray(lengths, dtype=np.int64)
    epoch = 0
    while True:
        order = epoch_order(len(lens), seed, epoch)
        for row in pack_greedy(lens[order].tolist(), capacity):
            yield epoch, tuple(int(order[j]) for j in row)
        epoch += 1


def row_arrays(examples: Sequence[tuple[np.ndarray, np.ndarray]], capacity: int,
               pad_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(input ids, position ids, loss-target flags), each of length `capacity`, for one packed row of
    (ids, mask) examples. Positions restart at 0 per example and for the padding tail; padding is never a target."""
    n = sum(len(i) for i, _ in examples)
    if n > capacity:
        raise ValueError(f"row holds {n} tokens, capacity {capacity}")
    pad = capacity - n
    ids = np.concatenate([*(i for i, _ in examples), np.full(pad, pad_id)]).astype(np.int64)
    pos = np.concatenate([*(np.arange(len(i)) for i, _ in examples), np.arange(pad)]).astype(np.int64)
    tgt = np.concatenate([*(np.asarray(m, dtype=bool) for _, m in examples), np.zeros(pad, dtype=bool)])
    return ids, pos, tgt


# ----------------------------------------------------------------------------- schedule and stopping
def warmup_steps(total: int, frac: float) -> int:
    return max(1, math.ceil(frac * total))


def lr_at(step: int, peak: float, warmup: int, total: int) -> float:
    """Learning rate of optimizer update `step` (0-based): linear warmup to `peak` over `warmup` updates, then cosine
    decay to 0 at update `total`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def stop_reason(step: int, max_steps: int, wall_s: float, max_wall_s: float) -> str | None:
    """'max_steps' or 'max_wall' once either budget is spent after `step` updates, else None."""
    if step >= max_steps:
        return "max_steps"
    return "max_wall" if wall_s >= max_wall_s else None


# ----------------------------------------------------------------------------- drift marks
def first_crossings(log: Sequence[tuple[int, float]], marks: Sequence[float]) -> dict[float, int]:
    """{mark: first logged step whose drift (%) is at or above it}, for every mark the log has reached.
    `log` holds (step, drift) in step order."""
    out: dict[float, int] = {}
    for step, d in log:
        for m in marks:
            if m not in out and d >= m:
                out[m] = step
    return out


def marks_due(log: Sequence[tuple[int, float]], marks: Sequence[float]) -> tuple[float, ...]:
    """Marks first crossed at the last logged step (a checkpoint is saved for them now)."""
    if not log:
        return ()
    last = log[-1][0]
    return tuple(m for m, s in sorted(first_crossings(log, marks).items()) if s == last)


def checkpoint_label(prefix: str, recipe: str, step: int) -> str:
    return f"{prefix}__{recipe}__step{step:05d}"


def parse_checkpoint_line(line: str) -> dict[str, str]:
    """One line of a checkpoints list: label, path, recipe, step, marks (tab-separated)."""
    label, path, recipe, step, marks = line.rstrip("\n").split("\t")
    return {"label": label, "path": path, "recipe": recipe, "step": step, "marks": marks}


def checkpoint_line(label: str, path: str, recipe: str, step: int, marks: Sequence[str]) -> str:
    return "\t".join((label, path, recipe, str(step), ",".join(marks) or "-")) + "\n"

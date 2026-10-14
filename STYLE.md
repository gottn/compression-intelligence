# Style: functional code, structured commits

## Code

1. **Pure by default.** A function's output depends only on its arguments. No hidden state, no reads of module-level mutables, no mutation of arguments. Return new objects.
2. **Effects at the edges.** File I/O, network, model loading and GPU calls live in thin `cmd_*` / `run_*` functions at the bottom of a module. Everything above them is pure and runs without a model.
3. **Explicit state.** Tokenizer, model, device, paths and seeds are parameters. Constants are UPPERCASE tuples or frozen dataclasses. `global` is banned (ruff PLW0603).
4. **Compute and write are separate functions.** A function that returns a table never also writes it.
5. **Determinism.** Fixed seeds, sorted iteration over dicts and directories, pinned dependencies. Every number in a RESULTS.md is reproducible by one script named there.
6. **Units in names or docstrings.** `bpb`, `bits`, `bytes`, `wall_s`. No bare floats whose unit the reader must guess.
7. **Small, typed, tested.** Type hints on every signature. Pure helpers get a pytest in `tests/`. `./check.sh` must pass before every commit.
8. **Shared code lives in `cilab/`.** Experiments under `expNN_*/` import from it and never from each other.

## Experiments

- One folder per experiment: `expNN_slug/` with `README` header in its `run_all.sh`, `data/`, `results/`, `RESULTS.md`.
- RESULTS.md leads with the verdict, then method, table, robustness, failures, next steps.
- Corpora are dated snapshots with a `*.prov.json`. Never re-collect in place; make a new name.
- Sanity checks run before the main run and their outputs are committed.

## Commits (Conventional Commits, enforced by `.githooks/commit-msg`)

```
<type>(<scope>): <what changed> — <measured effect when there is one>

<why, in one or two sentences>
<how it was verified>

Co-Authored-By: ...
```

- Types: `feat`, `fix`, `exp`, `data`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`.
- Scope is the experiment or package: `exp01`, `exp02`, `cilab`, `docs`.
- Subject ≤ 100 characters, imperative, no trailing period.
- `exp` and `feat` subjects state the measured effect: `exp(exp02): temperature rescaling — recovers 62% of Qwen3 bpb gap`.
- `refactor` subjects state what was proven unchanged: `refactor(cilab): move window helpers — exp01 bpb bit-identical on 4 rows`.
- One logical change per commit. Data snapshots are their own `data(...)` commit.

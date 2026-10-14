# compression-intelligence

Does lossless compression of fresh text track what post-training does to a language model?

Start with `STATE.md` (status, key numbers, how to rerun), then `docs/research-thread.md` (the plan and its gates) and `STYLE.md` (code and commit rules). The survey that started this is `docs/` → report in `~/Documents/Personal/reports/`.

| folder | what |
|---|---|
| `cilab/` | shared pure helpers: tokenization, windows, scoring, stats |
| `exp01_posttraining_bpc/` | pilot: base vs post-trained bits per byte on Aug–Oct 2026 text, 4 pairs ≤ 2B |
| `docs/` | research thread, one-sheet, prior-art note |
| `tests/` | pytest for the pure helpers |

Setup on a Mac (Apple silicon):

```bash
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r exp01_posttraining_bpc/requirements.txt ruff pytest && ./check.sh
```

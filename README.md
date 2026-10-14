# compression-intelligence

Does lossless compression of fresh text track what post-training does to a language model?

Start with `STATE.md` (status, key numbers, how to rerun), then `docs/research-thread.md` (the plan and its gates) and `STYLE.md` (code and commit rules). The survey that started this is `docs/` → report in `~/Documents/Personal/reports/`.

| folder | what |
|---|---|
| `cilab/` | shared helpers: `pairs` (models), `stages` (staged checkpoints), `text` (tokenization, windows), `score`, `stats`, `calib`, `io` |
| `exp01_posttraining_bpc/` | pilot: base vs post-trained bits per byte on Aug–Oct 2026 text, 4 pairs ≤ 2B |
| `exp02_temperature/` | does one temperature undo the post-training bpb gap? (35% for Qwen3, 19% for R1) |
| `exp03_staged_drift/` | Δbpb per post-training stage, 14 public checkpoints on AAU (OLMo-2 SFT +7.0%, DPO +6.6%, RLVR +0.5%; pure GRPO +0.1–0.2%) |
| `infra/` | AAU cluster submit script and Slurm jobs |
| `docs/` | research thread, one-sheet, prior-art note |
| `tests/` | pytest for the pure helpers |

Setup on a Mac (Apple silicon):

```bash
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r exp01_posttraining_bpc/requirements.txt ruff pytest && ./check.sh
```

# compression-intelligence: state

**Goal.** Test empirically whether "compression = intelligence" survives post-training. Concretely: does bits-per-byte (bpb) on fresh, post-cutoff text track capability across base and post-trained LLMs? Huang et al. 2024 found r ≈ −0.93, but only across base models.

## Next thread

See `docs/research-thread.md` (2026-10-14): fresh-text bpb as a forgetting meter for post-training. exp03 (staged public checkpoints, AAU) is done; next is exp04 (benchmarks for the same checkpoints on AAU).

## Status (2026-10-13)

**exp01_posttraining_bpc: done.** The full write-up is `exp01_posttraining_bpc/RESULTS.md`.

Setup:
- 4 base → post-trained pairs, all ≤2B parameters, with identical tokenizers within each pair.
- 3 fresh corpora (Aug–Oct 2026) of about 100 KB each, plus an enwik8 control.
- RAW and CHAT scoring conditions.
- Ran on the Mac with MPS in fp16, verified against fp32.

**Verdict.** Post-training worsens fresh-text bpb in every pair and every corpus. The size of the loss does not track the capability gain.

Key numbers: Δbpb, RAW, mean of the 3 fresh corpora (window-bootstrap 95% CIs all exclude 0):

| pair | Δbpb RAW |
|---|---|
| Qwen2.5-1.5B → Instruct | +0.9% |
| SmolLM2-1.7B → Instruct | +3.6% |
| Qwen3-1.7B-Base → Qwen3-1.7B | +16.2% |
| Qwen2.5-Math-1.5B → R1-Distill-1.5B | +15.8% |

- In the chat template, post vs base is +1.8%, n/a, +37.0% and +16.0% for the same four pairs. SmolLM2's base has no template.
- The signs hold with a plain `\n\n` document start: +0.9%, +4.1%, +17.0%, +20.9%.
- The gap is broad (the worst 1% of tokens explain 11–19% of it for Qwen3 and R1) and persists at window positions ≥128.
- GPT-2 sanity check: 1.243 bpb on enwik8.

Gotchas that are fixed but still worth knowing:
- The document-start token can move bpb by 5–30% for some models: Qwen2.5-Math with `<|endoftext|>`, SmolLM2 on enwik8, and R1 given its EOS.
- An in-process `model.half()` after an fp32 load keeps the fp32 copy alive on MPS. The fix is to run the fp32 reference in a subprocess, with one process per model.
- The Wikipedia API returns 429 after about 3 quick anonymous calls; pace at 1 call per 6 s.
- en.wikinews.org has been dead since May 2026.
- `.half()` also casts the rotary `inv_freq` buffer to fp16. The 8 exp01 rows from the first run (Qwen2.5-1.5B and -Instruct × arxiv/enwik8) carry a drift of 2–3e-5 relative. They were kept and are flagged in exp01's RESULTS.

## exp02_temperature: done (2026-10-14)

The full write-up is `exp02_temperature/RESULTS.md`.

**Question.** How much of the base → post fresh-text bpb gap does one scalar temperature undo?

**Method.** RAW condition and the same windows as exp01. T* is fit on the even windows, with a 23-point grid over 0.50–1.60 plus a parabola in log T, and evaluated on the odd windows.

**Verdict, against the rule stated before the run: mixed.** It is mostly not a calibration shift.

| pair | T* post (base) | gap recovered | residual gap at own T* |
|---|---|---|---|
| Qwen2.5 → Instruct | 1.00–1.06 (1.00–1.05) | 15% | +0.9% |
| SmolLM2 → Instruct | 1.04–1.08 (1.00–1.05) | 6% | +3.6% |
| Qwen3-Base → Qwen3 | 1.22–1.28 (1.00–1.04) | **35%** (CIs 28–42%) | **+10.4%** |
| Qwen2.5-Math → R1-Distill | 1.15–1.18 (1.02–1.07) | **19%** | **+12.7%** |

- Each figure is the mean of the 3 fresh corpora.
- Qwen3 and R1 are overconfident: T* is about 1.25 and 1.16, while every base model sits at 0.995–1.07.
- Undoing that sharpening still leaves a 10–13% bpb penalty. So "forgetting/drift" survives and "only calibration" is weakened.
- The caveat is that one global T is the crudest calibrator. The next step is entropy-binned T and a mixture with base or unigram.
- T=1 matches exp01's per-token bits to ≤ 1.6e-6 per window (≤ 1.1e-6 after re-scoring the 4 stale `.half()` rows).
- Total scoring time was 25 min.

## exp03_staged_drift: done (2026-10-14)

The full write-up is `exp03_staged_drift/RESULTS.md`.

**Question.** How far does each post-training stage move fresh-text bpb from the family base, and does pure RL move it at all?

**Setup.**
- 14 public checkpoints, in the stage table `cilab/stages.py`.
- RAW condition, with exp01's corpora and rules.
- One AAU job: 1080281, Tesla T4, fp16, 18 min 42 s.

**Verdict.** SFT and DPO move bpb a lot. Pure RL barely moves it, and no RL stage lowered bpb on any corpus. Steps vs their predecessor, mean of the 3 fresh corpora:

| step | Δbpb |
|---|---|
| OLMo-2-1B SFT | +7.0% |
| OLMo-2-1B DPO (on top of SFT) | **+6.6%** |
| OLMo-2-1B RLVR (on top of DPO) | +0.5% |
| SimpleRL-Zoo pure GRPO, 0.5B | +0.18% |
| SimpleRL-Zoo pure GRPO, 1.5B | +0.12% |
| Qwen2.5 Instruct, 0.5B / 1.5B / 3B | +1.6 / +0.9 / +2.1% |
| Qwen2.5-Math vs Qwen2.5-1.5B | +23.8% |
| R1-Distill vs Math | +15.8% |
| R1-Distill vs plain 1.5B | +43.0% |

- **Expectations.** E1 (SFT +3 to +8%) and E3 (RLVR ±1%) were met. E2 (DPO +0 to +2%) and E4/E5 (SimpleRL +0.5 to +3%) were not.
- **Control.** Against the base, the enwik8 control drifts 1.3–3× more than fresh text.
- **Reproduction.** exp01's 4 shared models reproduce on the T4 within 5e-5 bpb.

## How to rerun

```bash
cd ~/Documents/Personal/compression-intelligence/exp01_posttraining_bpc
bash run_all.sh        # resumable; ~1 h after the ~27 GB model download (HF cache in ~/.cache/huggingface)
```

- The environment is `.venv` (Python 3.12, created by uv), with exact versions pinned in `exp01_posttraining_bpc/requirements.txt`.
- Keep `data/`: it is a dated snapshot, and re-collecting would fetch different text.
- To redo the scoring only: `rm -rf results` and rerun.
- Individual steps (all run with `../.venv/bin/python`):
  - `score_bpb.py tokcheck`
  - `score_bpb.py sanity`
  - `score_bpb.py run`
  - `score_bpb.py docstart`
  - `score_bpb.py audit ...`
  - `analyze_positions.py`
  - `plot_results.py`
- exp02: `cd exp02_temperature && bash run_all.sh` (~30 min, resumable; needs the exp01 models in the HF cache and exp01's `results/tokens/`).
- exp03:
  - Scoring runs on AAU: `ssh aau 'cd ~/projects/compression-intelligence && infra/cluster/submit.sh exp03'` (~20 min on a T4), then rsync `results/` back.
  - Analysis runs on the Mac: `bash exp03_staged_drift/run_all.sh analyze`.
  - Mac preflight: `run_all.sh preflight`.

## Layout

- `exp01_posttraining_bpc/`:
  - `RESULTS.md`: verdict, tables and caveats.
  - `results/`: CSVs, figure, sanity and position audits, per-token bits.
  - `data/`: corpora with provenance JSON.
  - `logs/`: run logs, `notes.txt` incident log, model cards.
- `exp02_temperature/`: `RESULTS.md`; `results/` (runs.csv, windows.csv, grid/*.npy per-window bits on the T grid, summary.md, figure).
- `exp03_staged_drift/`: `RESULTS.md`; `results/` (bpb.csv, windows.csv, tokens/*.npy from the AAU job; stage_deltas.csv, summary.md, fig_stages.png from the analysis); `logs/` (Slurm log, Mac preflight).
- `infra/`: cluster submit script and Slurm jobs (`smoke`, `exp03`); rules in `docs/cluster-rules.md`.
- `cilab/`: shared helpers (pairs, stages, text, score, stats, calib, io); `tests/`: pytest for them.

## Next (from RESULTS.md)

1. ~~Separate the effect of each stage and of RL vs distillation.~~ Done in exp03: DPO is as large a step as SFT; pure RL is +0.1–0.5%.
   - Still open: larger corpora (about 3M characters per domain), and a pre-cutoff same-domain control to separate "fresh" from "old text with markup".
2. ~~Test calibration vs knowledge with one temperature.~~ Done in exp02: T > 1 recovers 35% (Qwen3) and 19% (R1). Next: richer calibrators, and a per-token decomposition of the residual (see exp02 RESULTS).
3. Run one benchmark harness across all models (cloze plus GSM8K / MATH-500), extend to 10–20 pairs, and test whether base bpb predicts post-trained scores.

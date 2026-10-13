# compression-intelligence: state

**Goal.** Test empirically whether "compression = intelligence" survives post-training. Concretely: does bits-per-byte (bpb) on fresh, post-cutoff text track capability across base and post-trained LLMs? Huang et al. 2024 found r ≈ −0.93, but only across base models.

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

## Layout

- `exp01_posttraining_bpc/`:
  - `RESULTS.md`: verdict, tables and caveats.
  - `results/`: CSVs, figure, sanity and position audits, per-token bits.
  - `data/`: corpora with provenance JSON.
  - `logs/`: run logs, `notes.txt` incident log, model cards.

## Next (from RESULTS.md)

1. Separate the effect of each stage and of RL vs distillation:
   - OLMo-2-1B staged checkpoints;
   - RL-zero checkpoints from Qwen2.5-Math-1.5B vs R1-Distill;
   - about 3M characters per domain.
2. Test calibration vs knowledge: fit a temperature per post-trained model and log entropy. Does T > 1 recover the 16%?
3. Run one benchmark harness across all models (cloze plus GSM8K / MATH-500), extend to 10–20 pairs, and test whether base bpb predicts post-trained scores.

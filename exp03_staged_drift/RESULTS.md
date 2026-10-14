# exp03: fresh-text bpb drift per post-training stage

Status: **pre-registered, not yet run.** This file holds the expectations, written and committed on 2026-10-14 before the AAU job was submitted. The verdict, tables and figure are added after the run, below the expectations, which stay unedited.

## Question

How far does each post-training stage move a model from its pretraining distribution, measured as Δbpb on fresh text against the family base? And does pure RL move it at all?

## Expectations (pre-registered 2026-10-14, before the run)

Δbpb is the RAW condition, averaged over the 3 fresh corpora (wiki2026, arxiv2026, gutenberg2026), with paired window-bootstrap 95% CIs. A range counts as **met** when the point estimate falls inside it.

| # | comparison | expected Δbpb |
|---|---|---|
| E1 | OLMo-2-0425-1B-SFT vs OLMo-2-0425-1B (base) | +3 to +8%, and the largest single step in the OLMo-2 chain |
| E2 | OLMo-2-0425-1B-DPO vs -SFT (on top) | +0 to +2% |
| E3 | OLMo-2-0425-1B-Instruct (RLVR) vs -DPO (on top) | within ±1% |
| E4 | Qwen-2.5-0.5B-SimpleRL-Zoo vs Qwen2.5-0.5B (pure GRPO from the base) | +0.5 to +3% |
| E5 | Qwen-2.5-1.5B-SimpleRL-Zoo vs Qwen2.5-1.5B (pure GRPO from the base) | +0.5 to +3% |

**Headline rule.** If any RL stage (E3 RLVR vs DPO; E4 or E5 SimpleRL-Zoo vs base) is *below* its predecessor with the whole 95% CI under zero, that is the headline: RL lowered fresh-text bpb.

**Reported without a pre-stated number:**
- Qwen2.5 Instruct vs base at 0.5B, 1.5B and 3B (the scale control; exp01 measured +0.9% at 1.5B on the Mac).
- Qwen2.5-Math-1.5B vs Qwen2.5-1.5B.
- R1-Distill-Qwen-1.5B vs both of its references: Qwen2.5-Math-1.5B (its parent) and plain Qwen2.5-1.5B.
- The enwik8 control for every comparison.

**Reproduction check.** Four models were already scored on the Mac (MPS) in exp01: Qwen2.5-1.5B, Qwen2.5-1.5B-Instruct, Qwen2.5-Math-1.5B and R1-Distill-Qwen-1.5B. Their AAU bpb must match exp01 within 1e-3 bpb on every corpus, despite the different GPU and torch build.

## Method (fixed before the run)

- **Models.** 14 public checkpoints in 4 families, listed with stage, recipe and parent in `cilab/stages.py` (`STAGES`).
- **Corpora.** exp01's dated snapshots (`../exp01_posttraining_bpc/data`): wiki2026, arxiv2026, gutenberg2026 and the enwik8_ctrl control, about 100 KB each.
- **Scoring.** RAW condition only, identical to exp01:
  - 1024-token windows;
  - input = [doc-start, *window];
  - every text token is scored;
  - bpb = Σ bits / UTF-8 bytes.
- **Doc-start token.** The tokenizer's BOS if its config adds one, else `<|endoftext|>`, else EOS:
  - OLMo-2: its tokenizer config sets no `add_bos_token`, so `<|endoftext|>` (100257), which is also its BOS and EOS;
  - Qwen2.5 family: `<|endoftext|>` (151643);
  - R1-Distill: `<｜begin▁of▁sentence｜>` (151646).
- **Tokenizer identity.** The Mac preflight checked that the 4 OLMo-2 stages encode all 4 corpora to identical ids. So do all 10 Qwen2.5-derived models. All round-trip exactly. The job repeats the check (`results/tokcheck.json`). Identical ids make the windows identical, so the bootstrap pairs windows.
- **Compute.** One AAU job, one GPU, fp16 weights, float32 logits, one model at a time (one child process per model).
  - Each model is checked against fp32 on window 0. If the two disagree by ≥ 5e-3 relative, or fp16 gives non-finite bits, that model is scored in fp32.
  - Job: `infra/slurm/exp03.sbatch`. Scorer: `score_stages.py`.
- **Statistics** (`analyze_stages.py`). Paired window bootstrap, 5,000 resamples:
  - each corpus has its own fixed seed, so every comparison on a corpus uses the same resamples;
  - the fresh-corpus mean resamples windows within each corpus independently (`cilab.stats.stratified_mean_ci`).

## Preflight (Mac, 2026-10-14)

`bash run_all.sh preflight`, outputs in `logs/preflight/`:
- Qwen2.5-0.5B on enwik8_ctrl: **0.83937 bpb** on MPS (fp16). Window 0 in fp16 is within 1.3e-5 relative of fp32, and the score is deterministic.
- Tokenizer identity holds in both groups.
- A second run skipped the finished row (resume works).

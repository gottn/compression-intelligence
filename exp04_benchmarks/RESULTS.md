# exp04: does fresh-text drift predict benchmark forgetting, but not gain?

**Status: pre-registered, no results yet.** Everything below was written and committed before the AAU jobs ran. The outcome sections are added after the run, above this text; the rules are not edited.

## Question

exp01 and exp03 measured Δbpb, the fresh-text bits-per-byte drift of each post-trained checkpoint from its family base. Does that drift predict how much general capability the checkpoint lost (forgetting), while being unrelated to what it gained on the target task (one-sidedness)? This decides the thesis in `docs/research-thread.md` (chain steps 1 and 2).

## Checkpoints

All 18 rows of `cilab/stages.py` (`STAGES`): 6 family bases and 12 post-trained checkpoints. Array index = row.

| family | base | post-trained checkpoints (stage) |
|---|---|---|
| olmo2-1b | OLMo-2-0425-1B | -SFT (sft), -DPO (dpo), -Instruct (rlvr) |
| qwen2.5-0.5b | Qwen2.5-0.5B | -Instruct (instruct), SimpleRL-Zoo (rl-zero) |
| qwen2.5-1.5b | Qwen2.5-1.5B | -Instruct (instruct), SimpleRL-Zoo (rl-zero), Qwen2.5-Math-1.5B (math-cpt), R1-Distill-Qwen-1.5B (r1-distill) |
| qwen2.5-3b | Qwen2.5-3B | -Instruct (instruct) |
| smollm2 | SmolLM2-1.7B | -Instruct (instruct) |
| qwen3 | Qwen3-1.7B-Base | Qwen3-1.7B (instruct) |

Every post-trained checkpoint is compared with its **family base** (R1-Distill and Qwen2.5-Math with plain Qwen2.5-1.5B), the same reference as the predictor.

## Benchmarks (AAU, `infra/slurm/exp04_eval.sbatch`, driver `run_eval.py`)

- lm-evaluation-harness 0.4.13 (pinned in `infra/cluster/requirements-exp04.txt`), HF backend, fp16, `batch_size auto`, `max_length 2048`, `--seed 0,1234,1234,1234`, greedy decoding (task defaults), no `--log_samples`.
- **Decoding is the harness's greedy default for every model.** transformers 5 merges each checkpoint's `generation_config.json` into `generate()`; Qwen's sets `max_new_tokens: 2048`, which overrode lm-eval's 256-token gsm8k limit in the first dry run (job 1080307, cancelled), and instruct checkpoints also ship sampling and repetition-penalty defaults. The driver replaces the generation config with one holding only the token ids (the dropped fields are recorded per model in `summary.json`). gsm8k therefore generates at most 256 new tokens (lm-eval's default; the task sets no `max_gen_toks`) and ifeval at most 1,280 (task setting).
- **One GPU type for all checkpoints:** the array job pins `--gres=gpu:a10:1`, so fp16 kernels are the same for every model.
- **No chat template for any model.** Base and post-trained checkpoints see byte-identical prompts. This favours base models on generative tasks and is a stated confound, not a hidden one.
- General side (log-likelihood, harness default few-shot = 0 for all six): arc_easy (test, 2,376), arc_challenge (test, 1,172), hellaswag (first 2,000 validation docs), piqa (validation, 1,838), winogrande (validation, 1,267), mmlu (first 40 test questions of each of 57 subjects = 2,280).
- Target side (generative): gsm8k 5-shot (task default), first 300 test problems, strict-match and flexible-extract both recorded; ifeval 0-shot, first 200 prompts (installed via `lm-eval[ifeval]`).
- 11,433 documents per checkpoint. Per-task limits are passed to the single log-likelihood call as `--samples` (doc indices 0..N−1, which is exactly what `--limit N` selects), because `--limit` is global to a call.
- One `lm_eval.simple_evaluate` call per task group (log-likelihood, gsm8k, ifeval), in one process per checkpoint with the model loaded once; a failed group does not stop the others and the array task exits non-zero. (The dry run used one CLI process per group; that spent ~6.5 min per call indexing every lm-eval task YAML with the GPU idle, so the driver now indexes only the 7 task directories it uses.)

## Definitions (fixed before the run)

- **Primary accuracy per general task:** `acc_norm` where the task reports it (arc_easy, arc_challenge, hellaswag, piqa), else `acc` (winogrande, mmlu).
- **General score** = mean of the 6 primary accuracies.
- **Forgetting** (headline) = 100 × (general_base − general_post) / general_base, in percent of the base score; positive = capability lost.
- **Forgetting, chance-adjusted** (secondary) = the same relative drop on the mean of chance-adjusted accuracies (acc − c) / (1 − c), with c = 0.5 for piqa and winogrande and 0.25 for the other four (ARC has a few 3- and 5-choice items; 0.25 is used for all). Harmon et al. (2510.17776) use a chance-adjusted, *sample-wise* forgetting metric; it needs per-sample outcomes, which this run does not log (`--log_samples` off), so we cite it and report the aggregate chance-adjusted drop instead.
- **Gain** (headline) = gsm8k flexible-extract exact match, post − base, in points. Secondary: strict-match, and ifeval `prompt_level_strict_acc` (post − base) for the chat-style checkpoints.
- **Predictors** (vs the family base, RAW condition, point estimates):
  - Δbpb_fresh = mean over wiki2026, arxiv2026, gutenberg2026 of 100 × (Σ bits_post − Σ bits_base) / Σ bits_base;
  - Δbpb_enwik8 = the same on the enwik8 control.
  - 10 checkpoints from `exp03_staged_drift/results/stage_deltas.csv` (AAU T4, `kind == vs_base`); SmolLM2-Instruct and Qwen3-1.7B from `exp01_posttraining_bpc/results/bpb.csv` (Mac MPS). Mixing is fine: exp03 reproduced exp01's 4 shared models on the T4 within 5e-5 bpb.
- **Statistics** (`analyze_forgetting.py`, over the 12 post-trained checkpoints): Pearson r (the gate statistic) and Spearman ρ of each predictor with each outcome; 95% percentile CIs from 10,000 bootstrap resamples of checkpoints (seed 0; resamples with a constant column are dropped); leave-one-out range of r; within-family OLS slopes for OLMo-2-1B (3 post-trained points) and Qwen2.5-1.5B (4 points). n = 12 is small and the CIs will be wide.

## Pre-registered rules

| rule | statistic | pass | soft pass | fail |
|---|---|---|---|---|
| **Gate 1**: drift predicts forgetting | Pearson r(Δbpb_fresh, forgetting), n = 12 | r ≥ 0.8 | 0.6 ≤ r < 0.8 | r < 0.6 |
| **One-sidedness**: drift does not predict gain | Pearson r(Δbpb_fresh, gsm8k gain) | holds if \|r\| < 0.3 **and** Gate 1 at least soft-passes | | otherwise not shown |

**Expected** (stated now, scored after):
- E1: r(Δbpb_fresh, forgetting) between 0.6 and 0.8 (soft pass).
- E2: r(Δbpb_fresh, gsm8k gain) near 0 or negative.
- E3: the enwik8 predictor does as well as or slightly better than fresh text (exp03: the control drifts 1.3–3× more).
- E4: R1-Distill has the largest drift (+43% vs Qwen2.5-1.5B) but only modest general loss, i.e. it sits below the fitted line.

How each expectation is scored (added 2026-10-14 in a separate commit, before any benchmark result existed; Pearson r throughout):
- E1 met if 0.6 ≤ r(Δbpb_fresh, forgetting) < 0.8.
- E2 met if r(Δbpb_fresh, gsm8k gain) < 0.3 (near zero or negative).
- E3 met if r(Δbpb_enwik8, forgetting) ≥ r(Δbpb_fresh, forgetting) − 0.05.
- E4 met if R1-Distill has the largest Δbpb_fresh of the 12 and a negative residual from the OLS line of forgetting on Δbpb_fresh fitted to all 12.
- "Chat-style" for the ifeval gain = stages sft, dpo, rlvr and instruct (9 checkpoints); ifeval gain is also reported over all 12.

## Confounds (stated before the run)

- **Prompt format.** No chat template anywhere. Chat-tuned and reasoning models (Qwen3, R1-Distill) are scored off their native format, which lowers their generative scores (gsm8k, and ifeval in particular) and can look like forgetting on log-likelihood tasks. This is the price of identical prompting.
- **Generation length.** gsm8k uses lm-eval's default cap of 256 new tokens for every model. Long chain-of-thought models (R1-Distill above all, and possibly SimpleRL-Zoo) can be cut off before their final answer, which understates their gsm8k gain. This is the cost of identical decoding; it is not corrected.
- **Subsets.** hellaswag 2,000 of 10,042; mmlu 40 per subject; gsm8k 300 of 1,319; ifeval 200 of 541. The binomial standard error of gsm8k at n = 300 is about ±2.8 points.
- **n = 12 checkpoints from different labs and recipes**, with family as a nuisance factor; two points (Qwen2.5-Math, R1-Distill) are continued pretraining and distillation on math, far from the others in Δbpb, and will carry leverage in Pearson r. Spearman and leave-one-out are reported for this reason.
- **Contamination.** Public benchmarks (gsm8k, mmlu, arc) may be in some post-training sets, inflating "gain" and masking "forgetting" unevenly across recipes.
- **Hardware.** All array tasks request an A10; the GPU is recorded per checkpoint. exp03's predictors came from a T4 (and exp01's from the Mac); benchmark and bpb hardware differ by design.

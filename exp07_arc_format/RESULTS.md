# exp07: is exp04/exp06's "forgetting" knowledge loss or an answer-format effect?

**Status: pre-registered, no results yet.** Everything below was written and committed before any exp07 job ran. The
outcome sections will be added above this text; the rules are not edited.

## Question

In exp04 and exp06, the forgetting of mathcot (exp06) and SmolLM2-1.7B-Instruct (exp04) is almost entirely an ARC
loss under lm-eval's default format: 0-shot, `Question: … \nAnswer:` with each answer text as the scored
continuation, `acc_norm`. Without ARC-e and ARC-c, mathcot forgets +0.4 to +1.5% and SmolLM2-Instruct −0.03%
(`docs/review-independent.md`, §3.4). The chat recipes' "no forgetting" (tulu, smoltalk, OLMo-2-SFT) is an MMLU gain
under a letter-choice format cancelling an ARC-e loss (§3.5). Nothing so far separates format from knowledge.

- **Hypothesis F (format).** The ARC loss shrinks below the noise floor when only the answer format changes.
- **Hypothesis K (knowledge).** The ARC loss persists across formats.

## Checkpoints

**(a) Public: the 18 rows of `cilab/stages.py`.** These are the 12 post-trained checkpoints and their 6 family bases,
with weights already in `~/hf_cache` on AAU. Each post-trained checkpoint is compared with its family base, as in
exp04.

**(b) Retrained exp06 recipes: tulu and mathcot, stage 1 only.** exp06 deleted these checkpoints, so they are trained
again.
- The trainer is `exp06_matched_drift/train_sft.py` with its stage-1 defaults, unchanged since exp06 stage 1 except
  for the new flags below:
  - full fine-tune of `allenai/OLMo-2-0425-1B`, AdamW 2e-5, 45 warmup updates, cosine decay to 0 at update 1,500;
  - 32 × 2,048 packed tokens per update, seed 0;
  - the drift hook every 20 updates, against update 0.
- **Data.** The data are re-made by exp06's `prep_data.py`, now pinned to the dataset revisions exp06 recorded:
  - tulu: `allenai/tulu-3-sft-mixture@b14afda60f1bbebe55d5d2fa1e4df5042f97f8be`;
  - mathcot: `open-r1/OpenR1-Math-220k@e4e141ec9dea9f8326f4d347be56105859b2bd68`.
  - The new `*.prov.json` must match exp06's (`exp06_matched_drift/logs/setup/data/`) in kept examples, candidates
    scanned, drop counts and token sums. If it does not, training is not started.
- **Checkpoints.** Each recipe saves one at update 1,500 (final), and one at the first scored update with
  Δbpb_fresh ≥ 3% if that is reached (`--marks 3`). In exp06 stage 1 neither recipe reached 3%.
- **Differences from exp06 stage 1.** None of these touches the optimisation:
  - the label prefix is `exp07`;
  - only the 3% mark is saved;
  - the training wall-time stop is 27,000 s (7.5 h) instead of 30,600 s, to fit the 8 h Slurm limit. Stage 1 needed
    23,400 s, so the stop should not bind. If it binds, the run is reported as not reproduced.
  - The GPU is an A40, as exp06. An L40S fallback would change the kernels and is recorded if used.
- smoltalk is not retrained: its exp06 forgetting was already inside the floor.
- The base OLMo-2-0425-1B is in (a).

## Format battery

The harness is the same as exp04:
- lm-evaluation-harness 0.4.13, HF backend, fp16, `batch_size auto`, `max_length 2048`;
- seeds 0 / 1234 / 1234 / 1234, no system prompt;
- A10 for every checkpoint;
- one process per checkpoint, the model loaded once, and one `simple_evaluate` call per group, resumable per group.

The driver is `exp07_arc_format/run_battery.py`, built from exp04's `run_eval.py` helpers (now in `cilab/lmeval.py`).
- **New in exp07:** per-item outcomes (`acc`, `acc_norm` per document) are saved, so base and checkpoint can be paired
  on the same items. exp04 did not save them.

| group | tasks | shots | format | metrics |
|---|---|---|---|---|
| **B1** | arc_easy (test, 2,376), arc_challenge (test, 1,172) | 0 | exp04's default: `Question: {q}\nAnswer:` + ` {answer text}` | acc and acc_norm |
| **B2** | arc_easy, arc_challenge | 25 | as B1, with 25 solved train-split examples (lm-eval's seeded sampler, seed 1234), the Open LLM Leaderboard convention | acc_norm and acc |
| **B3** | arc_easy, arc_challenge | 0 | `apply_chat_template`: the B1 question as one user turn, the template's generation prompt, then the answer text as the assistant's continuation (lm-eval drops the leading space under a chat template) | acc_norm and acc |
| **B4** | mmlu (default) and mmlu_continuation | 0 | letter choice (`… A. …\nD. …\nAnswer:` + ` A`) and answer-text continuation (`Question: …\nAnswer:` + ` {answer text}`) | acc |
| **B5** | hellaswag (first 2,000 validation docs) | 0 | default | acc and acc_norm |
| B0 | piqa, winogrande | 0 | default | as exp04 |

- **MMLU sample.** Both B4 formats use the first 40 test questions of each of the 57 subjects (2,280), via exp04's
  samples spec (`cilab.lmeval.samples_spec`, mapped to the `mmlu_<subject>_continuation` task names for the
  continuation format).
- **B0** runs only on the retrained checkpoints. It completes their 6-task general score; the public checkpoints have
  it from exp04.
- **Which chat template B3 uses.**
  - Post-trained and retrained checkpoints use their own tokenizer and template. The retrained checkpoints carry
    OLMo-2-0425-1B-Instruct's, as in exp06.
  - Each base is scored once per post-trained model of its family, with that model's tokenizer and template. This
    includes OLMo-2-0425-1B with OLMo-2-0425-1B-Instruct's, as in exp06. Every B3 comparison therefore sees
    identical prompt tokens on both sides.
  - Templates that read `enable_thinking` (Qwen3) get `enable_thinking=False`, as in exp04.
  - The md5 of every template used is recorded.
- **B2 truncation check.** 25-shot contexts longer than 2,048 tokens are left-truncated by the harness. Their count per
  task is logged.

## Definitions

- **ARC-E loss under Bk** (points) = 100 × (acc_norm of the base − acc_norm of the checkpoint) under group Bk.
  Positive means a loss. The reference is the family base; for the retrained checkpoints it is OLMo-2-0425-1B. Under
  B3 the base is the run with the checkpoint's own template. The same quantity on `acc`, and the ARC-C version, are
  reported alongside.
- **Normalisation gap** = acc_norm − acc (points), per checkpoint and group. **Normalisation amplification** = the loss
  on acc_norm − the loss on acc.
- **MMLU gain** (points) = 100 × (checkpoint − base) acc, under the letter format and under the continuation format.
- **Floors.**
  - **Unpaired floor** (exp04's formula, for one metric) = 1.96 × √(se_checkpoint² + se_base²) × 100, from lm-eval's
    standard errors.
  - **Paired 95% CI** (robustness, new) = the mean per-item difference ± 1.96 × its standard error, over the same
    documents.
- **General score and forgetting** are as exp04: the mean of the 6 primary accuracies, and 100 × (base − checkpoint) /
  base.
- **Forgetting without ARC.**
  - **5-task mean without ARC-E:** ARC-C, HellaSwag, PIQA, WinoGrande, MMLU. This is the mean named in the brief.
  - **4-task mean without both ARC tasks:** the review's reading.
  - Each has its floor from the exp04 formula with the n-task standard error.
  - Public checkpoints take these from exp04's `scores.csv` (the forgetting in question). Retrained checkpoints take
    them from B0, B1, B4 and B5.

## Pre-registered decision rules

**Rule 1: knowledge or format.** It is applied separately to mathcot at update 1,500 (retrained) and to
SmolLM2-1.7B-Instruct (against SmolLM2-1.7B). The metric is acc_norm, with acc reported.

| verdict | condition |
|---|---|
| **K** (knowledge) | ARC-E loss ≥ 5 points under B2 **and** under B3 |
| **F** (format) | ARC-E loss ≤ 2.2 points (the exp06 floor) under B2 **or** under B3 |
| **mixed** | otherwise |

**Rule 2: does the chat recipes' MMLU gain survive a format change?** It is applied to tulu at update 1,500
(retrained) and to the 8 chat-style public checkpoints:
- OLMo-2 SFT, DPO and Instruct;
- Qwen2.5-0.5B, 1.5B and 3B-Instruct;
- SmolLM2-1.7B-Instruct;
- Qwen3-1.7B.

| verdict | condition |
|---|---|
| **format** | the letter-format gain exceeds its unpaired floor and the continuation gain does not |
| **survives** | both exceed their floors |
| **no gain** | the letter-format gain is inside its floor |

The ratio of the continuation gain to the letter gain is reported for every checkpoint. "Halves" means a
continuation gain ≤ 0.5 × the letter gain.

**Rule 3: reproduction of exp06 stage 1.**
- Let d_t = Δbpb_fresh(exp07) − Δbpb_fresh(exp06) at scored update t.
- Let σ_run be the root mean square of d_t over the scored updates 760–1,480 (37 points): the run-to-run noise
  measured on the same pair of runs.
- A recipe reproduces if:
  - |d_1500| ≤ 2 σ_run, **and**
  - its 6-task general score at update 1,500 differs from exp06's step-1,500 score by less than the unpaired 95%
    floor of the two (exp04's formula, about 2.2 pp of the base).
- The exp06 values at update 1,500 are Δbpb_fresh +2.016% (tulu) and +2.496% (mathcot), with general scores 0.5974
  and 0.5639.
- If a recipe does not reproduce, Rules 1 and 2 are still scored on it, but flagged.

**Sanity check (reported, not a rule).** B1, B4-letter and B5 re-run exp04's ARC, MMLU and HellaSwag settings on the
18 public checkpoints. They should agree with exp04's `scores.csv` within 0.5 points: same kernels, possibly a
different batch size.

**Also reported for every checkpoint:**
- the ARC-C version of Rule 1's numbers;
- the HellaSwag change (B5, the control task, where no recipe showed a loss);
- forgetting on the 5- and 4-task means without ARC;
- the normalisation gap and amplification under B1, B2 and B3;
- paired CIs for every loss and gain.

## Expected (stated now, scored after)

- **E1.** mathcot at update 1,500: the ARC-E loss persists at 25-shot (B2 loss ≥ 5 points, the K side), but shrinks
  under the chat template (2.2 < B3 loss < 5). The verdict is **mixed**.
- **E2.** tulu at update 1,500: the MMLU gain halves under continuation (continuation gain ≤ 0.5 × letter gain).
- **E3.** Both retrained recipes reproduce exp06 (Rule 3).

## Figure (`results/fig_format.png`)

- **Left:** per checkpoint, the ARC-E loss against the base under B1, B2 and B3, as grouped bars, with the 2.2- and
  5-point lines.
- **Right:** per chat-style checkpoint, the MMLU gain under the letter format against the continuation format, with
  floors.

## Confounds (stated before the run)

- **B3 is not a neutral format for every model.**
  - mathcot's template turn was followed by `<think>` reasoning in training, so a bare answer after the generation
    prompt is still off-distribution for it.
  - R1-Distill's template opens `<think>` itself.
  - B3 changes the wrapper, not the "answer immediately" convention.
- **B2's shots come from ARC's train split**, the same distribution as the test questions. Few-shot also restores the
  base's in-context answer format, which is the point of the test.
- **mmlu_continuation and mmlu-letter share questions, not difficulty.** Continuation scoring is known to sit nearer
  chance for small models, which compresses gains.
- **One seed per recipe.** The reproduction gives one replicate pair per recipe, so σ_run is the only run-to-run noise
  estimate.
- **Unpaired floors are conservative for "lost" calls and liberal for "no difference" calls.** Paired CIs are added
  for that reason.

## Budget and jobs

- **Setup:** `exp06_setup` with `--recipes tulu mathcot` (CPU, about 25 min).
- **Training:** `exp07_train`, array 0–1, A40, `--time=08:00:00`, about 13 GPU-hours.
- **Evaluation:** `exp07_eval`, one task per checkpoint, A10, `--time=02:00:00`, about 12–16 A10-hours in total.
  - It reads `$CI_OUT/checkpoints.txt` in the training run: the 18 public rows first, then the retrained checkpoints
    appended by the training job.
  - The public part runs while the training does.
  - The rest is queued with `--dependency=afterany` on the training array.
- **At most 4 concurrent GPU tasks:** 2 trainings plus at most 2 evaluations.
- **A dry run comes first:** 2 public checkpoints with `--limit 10`.

# exp06: at matched drift, does the knowledge − style contrast predict forgetting and ignore the gain?

**Status: pre-registered, no results yet.** Everything below was written on 2026-10-15, before any training step, drift score or benchmark of an exp06 checkpoint existed. The outcome sections will be added above this text after the run; the definitions and rules are not edited.

## Question

exp04 found that fresh-text Δbpb measures the dose of post-training: it tracks forgetting (ρ = 0.67, n = 12) and the IFEval chat gain (r = 0.89, n = 8). exp05 found a post-hoc lead: the direction of the drift, contrast = drift_KNOWLEDGE − drift_STYLE, tracked forgetting (r = 0.81 without the math line) and not the gain (r = 0.17). Those 12 checkpoints came from different labs, bases and recipes, and differed in dose.

Here the base is fixed, the dose is fixed and only the recipe varies:
- one base, OLMo-2-0425-1B, and three SFT recipes that differ only in content;
- checkpoints saved when the drift first crosses the same marks (3%, 6%, 10% Δbpb_fresh);
- at a mark, total drift is equal by construction, so if forgetting still differs between recipes, the contrast has to explain it.

## Design

**Base:** `allenai/OLMo-2-0425-1B`, the exp03/exp04 base (exp04 general score 0.58890).

**Recipes.** Same hyperparameters, same chat format, same token budget and the same number of updates; only the content differs.

| recipe | dataset (config, split) | rows | role |
|---|---|---|---|
| R1 `tulu` | `allenai/tulu-3-sft-mixture` (default, train) | 939,343 | style-heavy: OLMo-2's own SFT data |
| R2 `smoltalk` | `HuggingFaceTB/smoltalk` (`all`, train) | 1,043,917 | SmolLM2-Instruct's SFT data (SmolLM2-Instruct forgot 3.8%, mostly ARC, at +3.6% drift) |
| R3 `mathcot` | `open-r1/OpenR1-Math-220k` (`all`, train) | 225,129 | knowledge-heavy: one R1 reasoning answer (`<think>` … answer) per problem |

**Format.**
- Every example is the dataset's `messages` field, rendered with the chat template of the `allenai/OLMo-2-0425-1B-Instruct` tokenizer, for all three recipes. The base tokenizer has no template; the two tokenizers have the same vocabulary, special tokens and ids on exp01's corpora (exp03 tokcheck).
- The template starts with `<|endoftext|>` and closes each assistant turn with `<|endoftext|>`.
- **Loss on assistant tokens only.** A token is a loss target when its character span starts inside an assistant turn's span. The span is the characters the turn adds after the generation prompt `<|assistant|>\n`: the reply plus its closing `<|endoftext|>`. Every assistant turn of a multi-turn example is trained.
- **Maximum length 2,048 tokens.** Longer examples are dropped, never truncated.
- **Other drops**, counted and reported: no assistant turn, an assistant turn before any prompt, a role other than system/user/assistant, an empty content, and a content containing the literal `<|endoftext|>` or `<|pad|>`.
- Helpers: `cilab/sft.py` (`drop_reason`, `assistant_spans`, `loss_mask`, `fits`).

**Selection: token-matched subsets (decided with the coordinator on 2026-10-15, before any data was prepared).**
- The brief's "20k random examples" conflicts with its "same token budget". On a 400-row sample tokenized with the template, examples that fit hold a mean of 323 tokens (tulu), 761 (smoltalk) and about 1,558 (OpenR1-Math), and only about 13% of OpenR1-Math answers fit 2,048 tokens (median 5.1k).
- 20k examples each would mean about 15 / 6.5 / 5 passes at the same update count. So the subsets are matched on tokens instead.
- **Rule.**
  - Each recipe's candidates are taken in the order `numpy.random.default_rng(0).permutation(rows)`.
  - Fitting examples are kept until the kept examples hold **D = 45,000,000 template tokens** (prompt and reply; the example that crosses D is kept).
  - If one recipe's fitting pool holds fewer than D tokens, D becomes that pool's total for all three.
  - The same rule is why `mathcot` uses the OpenR1-Math `all` config: the default config's fitting pool (about 19M tokens) would force about 5 passes.
- **Expected counts:** about 140k (tulu), 60k (smoltalk) and 29k (mathcot) examples, with mean tokens per example of about 323 / 761 / 1,558. The exact counts, means and drop counts go in this file after the data job.
- **Bias, stated now.** The 2,048-token cut keeps only the shorter math solutions, so `mathcot` is biased toward short reasoning chains and easier problems. Tulu and SmolTalk lose only their longest 3–5%.

**Training.** One Slurm array task per recipe, all identical apart from the data:
- **Weights.** Full fine-tune with fp32 master weights, bf16 autocast and SDPA attention.
- **Packing.**
  - Examples fill rows of 2,048 tokens greedily in a per-epoch order, `default_rng([0, epoch]).permutation`. An example never splits.
  - Position ids restart at 0 for each example and for the padding tail, so attention stays within an example (transformers' packed-sequence mask). Padding is never a target.
  - Helpers: `pack_greedy`, `row_stream`, `row_arrays`.
- **Batch.** 32 rows per update, i.e. 65,536 token positions. Micro-batch 2 with 16 accumulation steps; the micro-batch may change after the dry run if memory requires it, which leaves the update unchanged.
- **Loss.** Mean cross-entropy over the loss targets of the whole update.
- **Optimizer.** AdamW with β = (0.9, 0.999), ε = 1e-8, weight decay 0 and gradient-norm clipping at 1.0.
- **Learning rate.** Peak 2e-5, linear warmup over 45 updates (3% of 1,500), then cosine decay to 0 at update 1,500.
- **Stop.** At 1,500 updates (98.3M token positions, about 2.2 passes over D for every recipe) or after 5 h of training wall time, whichever comes first. The step count D implies at 32 × 2,048 per update is 687 per pass, so the 1,500-update cap stands.
- **Seed.** 0 everywhere.
- **Hardware.** `--gres=gpu:a40:1` (48 GB), `--time=06:00:00`, `--mem=64G`, `--cpus-per-task=8`. L40S if no A40 starts within 30 min.

**Drift during training.**
- **When.** Every 20 updates. This is a superset of the brief's every 100: a mark is only as well matched as the scoring grid, and one score takes about 10 s on the training GPU.
- **Weights scored.** The fp32 weights are copied into a bf16 copy of the model, exactly the weights a checkpoint would save. That copy scores exp01's four corpora in the RAW condition with exp03's mechanics:
  - the base tokenizer;
  - 1,024-token windows that never split a character;
  - input = [`<|endoftext|>`, *window];
  - −log2 p from float32 logits (`cilab.score.score_raw`).
- **Reference.** The base model, cast to bf16 the same way, scored on the same GPU before the first update (step 0). Sanity check: its bpb is within 0.5% (relative) of exp03's fp16 T4 bpb on each corpus.
- **Quantities** (`cilab/drift.py`; fresh = unweighted mean of wiki2026, arxiv2026 and gutenberg2026, as in exp03–exp05):
  - Δbpb per corpus and Δbpb_fresh;
  - drift per class for exp05's POS classes (knowledge / middle / style) and frequency classes (rare / frequent);
  - **contrast_ks = drift_knowledge − drift_style** (primary);
  - contrast_rf = drift_rare − drift_frequent (secondary).
- **Classes.** The per-token classes are exp05's committed `olmo2__<corpus>.npz`. The job raises unless their ids SHA-1 and window lengths equal the ids and windows it scores.
- **Outputs.** Per-token bits for every scored step go to the run directory as `.npy`; one row per scored step goes to `drift.csv`.

**Checkpoints.**
- A checkpoint is saved at the first scored update whose Δbpb_fresh is at or above each mark **{3%, 6%, 10%}** (`first_crossings`, `marks_due`), and at the final update.
- One checkpoint can carry several marks.
- Format: bf16 safetensors plus the Instruct tokenizer, about 3 GB each.
- A mark a recipe never reaches is reported as not reached.
- The list of checkpoints is `$CI_OUT/checkpoints.txt`.

**Evaluation** (exp04's `run_eval.py`, with a new `--model-path` option; exp04 behaviour unchanged):
- **Groups.**
  - `loglik`: arc_easy, arc_challenge, hellaswag (2,000), piqa, winogrande and mmlu (40 per subject);
  - `gsm8k_chat`: 300 problems, 5-shot as multi-turn;
  - `ifeval_chat`: 200 prompts.
- **Settings.** fp16, A10, `--time=01:30:00`, at most 4 tasks at once (training included).
- **Base.** The base weights are evaluated in the same three groups with the Instruct tokenizer (`--tokenizer`), because the base has no template of its own and exp04 has no chat-condition score for it. Sanity check: its `loglik` general score reproduces exp04's base (0.58890) within 0.5 points.

**Outcomes.**
- **forgetting** = 100 (g_base − g) / g_base, in % of the base. g is the mean of the 6 primary accuracies (acc_norm for ARC-e, ARC-c, HellaSwag and PIQA; acc for WinoGrande and MMLU), and g_base = exp04's base, 0.58890.
- **IFEval gain** = 100 × (prompt-level strict accuracy of the checkpoint with the chat template − the base's with the same template), in points.
  - Correlations do not depend on which base value is subtracted.
  - The exp04-style gain (against the base's plain-prompt 0.095) is reported too.
- **gsm8k chat gain** (flexible-extract, points) is reported, with no test.
- **Noise floor of a difference between two checkpoints:** 1.96 √(se₁² + se₂²) / g_base × 100, with each se the standard error of the 6-task mean (exp04's formula).

## Pre-registered tests

The primary set is the **mark checkpoints** (up to 3 recipes × 3 marks = 9).

Statistics:
- Pearson r, with 95% percentile CIs from 10,000 checkpoint-bootstrap resamples (seed 0; resamples with a constant column are dropped; `cilab.stats.bootstrap_corr`);
- Spearman ρ;
- the leave-one-recipe-out r (each recipe dropped in turn, n ≤ 6).

| id | statistic | pass | soft | fail |
|---|---|---|---|---|
| **H1** contrast predicts forgetting | Pearson r(contrast_ks, forgetting) over the mark checkpoints | r ≥ 0.8 | 0.6 ≤ r < 0.8 | r < 0.6 |
| **H2** contrast ignores the gain | Pearson r(contrast_ks, IFEval gain) over the mark checkpoints | \|r\| < 0.3 | | \|r\| ≥ 0.3 |
| **H3** at matched drift, dose does not decide forgetting | per mark, the range of forgetting across the recipes that reached it, against the 95% floor of a two-checkpoint difference | range > floor at ≥ 2 of the 3 marks | | otherwise |
| **H4** recipe ordering at the 6% mark | contrast_ks and forgetting of the three 6% checkpoints | contrast mathcot > smoltalk > tulu **and** forgetting in the same order | contrast order holds, forgetting order does not | contrast order violated |
| **Fail rule** | r(contrast_ks, Δbpb_fresh) over the mark checkpoints; H1 and H4 | | | r > 0.9, **or** H1 r < 0.6 with the H4 contrast order violated: the lead is not supported |

**Rules for the edge cases.**
- **H3.** r(Δbpb_fresh, forgetting) over the mark checkpoints is reported beside it. Within a mark, r is about 0 by construction.
- **H1 with few checkpoints.** If fewer than 6 mark checkpoints exist, the H1 verdict is reported with "underpowered (n < 6)".
- **H4 incomplete.** If a recipe never reaches 6%, H4 is "not evaluable", and the fail rule uses H1 alone.

**Secondary readings** (reported, not gates):
- every test with contrast_rf in place of contrast_ks;
- every test over the mark checkpoints plus the finals (up to 12);
- r of contrast_ks with the residual of forgetting after an OLS fit on Δbpb_fresh;
- Spearman for every Pearson test;
- per-recipe drift, contrast and loss trajectories over all scored updates;
- MMLU and ARC separately (SmolLM2-Instruct's loss was ARC-led).

## Expected (stated now, scored after)

| at the 6% mark | tulu | smoltalk | mathcot |
|---|---|---|---|
| forgetting, % of base | 0 ± 2 | 2 to 4 (ARC-led) | 3 to 8 |
| contrast_ks, pp | −3 to +1 | +2 to +5 | +5 to +12 |
| IFEval chat gain | highest | middle | lowest |

- **E0.** All three recipes reach 6% within 1,500 updates. `tulu` may not reach 10%: OLMo-2's own SFT on the full Tülu 3 mix ended at +7.0%.
- **E1.** H1 at least soft-passes and H2 passes. That is the one-sided meter exp04 failed to find with total drift.
- **E2.** The fail rule does not trigger, because at a matched dose the contrast varies by recipe while Δbpb_fresh does not.

## Confounds (stated before the run)

- **Length and difficulty.** The 2,048-token cut biases `mathcot` toward short chains. Token matching gives the three recipes different example counts (about 140k / 60k / 29k), so per-example repetition is equal (same passes) but the number of distinct prompts is not.
- **Marks are matched on a grid.** A checkpoint overshoots its mark by up to 20 updates of drift. The achieved Δbpb_fresh of every checkpoint is reported, and the within-mark spread of Δbpb_fresh is stated next to H3.
- **One seed and one base.** No seed replicate. Benchmark noise (about ±2% of base for one checkpoint) is of the same size as the expected tulu–smoltalk difference.
- **Precision.** Drift is scored in bf16 on the training GPU; benchmarks run in fp16 on A10s from the bf16 checkpoint. The base reference is scored both ways.
- **Template.** The base sees the OLMo-2-Instruct template only at evaluation (for the gain reference). Its chat IFEval score may be near its plain-prompt 0.095.
- **Dose-correlated contrast.** If every recipe's contrast grows with dose, the contrast across marks is partly total drift. That is what the fail rule's r(contrast, Δbpb_fresh) > 0.9 clause tests.

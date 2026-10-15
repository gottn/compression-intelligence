# Prior art, tools and products: has anyone shipped a post-training drift meter?

Searched 2026-10-14 to 2026-10-15. This is the companion to `prior-art.md`, which covers papers. This note covers implementations: fine-tuning platforms, training libraries, eval and monitoring tools, open-source repos, paper code releases, and work from Sept–Oct 2026.

How links and claims were checked:

- **Links.** Every link below returned HTTP 200 on 2026-10-15.
- **"(src-checked)".** For the closest items, the claim was also checked against the source file, README, PR metadata or PDF. These items are marked (src-checked).
- **Platform docs.** These are cited by their exact metric and field names rather than long prose quotes, so each claim can be checked at the URL.

Overlap ratings: **same thing / close / adjacent / unrelated**.

## Verdict

**Nobody ships it.** No platform, library, eval tool or repo found does all of the following:

1. scores a fixed general-text corpus;
2. reports the paired Δbpb of a post-trained checkpoint against its base;
3. gives a paired-bootstrap CI;
4. is used to rank or gate post-training candidates across recipes;
5. is validated against benchmark forgetting.

Each piece exists somewhere, but separately:

- **Fresh-text bpb corpus:** lm-eval `uncheatable_eval`, merged 2026-08-28. It gives an absolute score per model and its README says it is for base models only.
- **Paired reference-vs-candidate divergence on a fixed corpus:** llama.cpp `--kl-divergence` and exllamav3 `model_diff.py`. Both are built for quantization.
- **Two-model bpb gap report:** Marin `perplexity_gap`. It is built to compare one pretrained model with another.
- **Packaged fine-tune-vs-base log-prob delta with a bootstrap CI and pass/fail:** dlm-sway `external_perplexity`. It is built for LoRA adapters, has 0 stars and is unvalidated.

Elsewhere:

- **Platforms** report validation loss on the user's own data. OpenAI and Azure add a safety gate.
- **Amazon Nova Forge** is the only platform that names forgetting as a problem. It tells users to run MMLU, HumanEval and GSM8K on the base and on each candidate.
- **Monitoring products** cover data, prediction and app-output drift only.
- **Training libraries:** none of the 13 checked computes KL on anything other than its own rollouts or the training batch.

## What "the same thing" would have to do

| # | Property | Ours |
|---|---|---|
| P1 | Fixed general-text corpus, post-cutoff | About 100 KB of Wikipedia, arXiv abstracts and Gutenberg from Aug–Oct 2026, plus an enwik8 control |
| P2 | Paired against the base, signed | Δbpb vs base, same tokenizer family |
| P3 | Uncertainty | Paired window bootstrap |
| P4 | Cheap: no task data, no generation | About 2 s per checkpoint on a GPU |
| P5 | Purpose | Rank or gate candidates across SFT, DPO, RLVR, GRPO and distillation |
| P6 | One-sided and validated | Predicts general-ability loss, not target gain. Validation against benchmarks is pending (exp04). |

---

## 1. Fine-tuning platforms

**Answer: none reports a drift, forgetting or regression score against the base model, and none evaluates on held-out general text.**

- The base-relative metrics that do exist are measured on the user's own task data (Together's DPO KL) or are safety gates (OpenAI, Azure).
- Amazon Nova Forge is the only platform that names catastrophic forgetting. Its instrument is a full benchmark run.

| Platform | Docs | What it reports | Drift vs base? | Overlap |
|---|---|---|---|---|
| OpenAI SFT | [supervised-fine-tuning](https://developers.openai.com/api/docs/guides/supervised-fine-tuning) | `full_valid_loss` and `full_valid_mean_token_accuracy` on the user's validation file. After training, an automatic safety eval runs and can block deployment; the community forum names it `refusals_v3`. The docs tell users to build their own evals to compare against the base. | Safety gate only | adjacent |
| Azure AI Foundry | [fine-tuning](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/fine-tuning), [safety evaluation](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/fine-tuning-safety-evaluation) | `train_loss`, `full_valid_loss` and token accuracy. After training, an automatic harm eval runs, using the same method as for base models, and blocks deployment on failure. | Safety gate only | adjacent |
| Google Vertex AI (Gemini tuning) | [gemini-use-supervised-tuning](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-use-supervised-tuning) | `/train_total_loss`, `/eval_total_loss` and `/eval_fraction_of_correct_next_step_preds`. An optional `evaluationConfig` (pointwise, pairwise, BLEU, ROUGE) runs on the user's validation data. The RL tuning metrics show no KL. | No | unrelated |
| Together AI | [training metrics](https://docs.together.ai/docs/fine-tuning-training-metrics.md) | DPO jobs log `train/kl_divergence` against the reference model, computed on the preference training data. | Base-relative, but on task data | adjacent |
| Fireworks | [fine-tuning](https://docs.fireworks.ai/fine-tuning/fine-tuning-models), [evaluating](https://docs.fireworks.ai/fine-tuning/evaluating-fine-tuned-models), [quantization blog (2024-08-01)](https://fireworks.ai/blog/fireworks-quantization) | SFT evaluation uses a split carved from the training data, plus the user's own metric. The RL metric `train/inference_k3` only checks that training and inference numerics agree, and the default GRPO recipe has no reference KL. Separately, Fireworks uses KL between the reference and the quantized model to gate quantization quality. | No for fine-tuning. The quantization KL is the same mechanism as ours. | adjacent |
| AWS Bedrock customization | [evaluation](https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation.html) | `perplexity` and `validation_loss` on the user's validation set. Evaluations use built-in or custom datasets, an LLM judge or human raters. | No | adjacent |
| Amazon Nova Forge and SageMaker HyperPod recipes | [nova-forge-cpt](https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-forge-cpt.html), [nova-model-evaluation](https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-model-evaluation.html) | See the bullets below this table. | Same goal, slow benchmark instrument | adjacent |
| Mistral Forge | [announcement](https://mistral.ai/news/forge) | Watches for regressions on benchmarks the user chooses. | Benchmarks only | adjacent |
| Thinking Machines Tinker | [evaluations](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/evaluations/) | `NLLEvaluator` gives held-out NLL on data the user supplies. No comparison with the base (see §2). | No | adjacent |
| Lamini | [evaluation](https://docs.lamini.ai/tuning/evaluation/) | A dashboard compares the tuned model with its base. The metrics are not specified. | Unclear | adjacent |
| Nebius Token Factory | [how-to-fine-tune](https://docs.tokenfactory.nebius.com/fine-tuning/how-to-fine-tune) | `train_loss` and `valid_loss`. | No | unrelated |
| Anyscale | [finetuning intro](https://docs.anyscale.com/llms/finetuning/intro) | No eval metrics documented. LLMForge is being deprecated. | No | unrelated |

Amazon Nova Forge in more detail:

- Its docs describe mixing old pre-training data with new domain data to prevent catastrophic forgetting.
- Users are told to run the starting checkpoint and each candidate through MMLU, HumanEval and GSM8K, using the `*_general_text_benchmark_eval.yaml` recipes.
- An AWS blog post reports MMLU of 0.75 for the baseline, 0.47 after training on customer data only, and 0.74 with mixing.
- The `strong_reject` metric `deflection` is base-relative and comes with a significance test, but it covers safety only.
- This is the closest platform. It has the same goal as ours but uses a slow benchmark instrument.

Platforms that are gone, unmaintained or unverifiable:

- **Mistral fine-tuning API:** deprecated ([docs](https://docs.mistral.ai/capabilities/finetuning/)).
- **Cohere fine-tuning:** deprecated 2025-09-15. A search snippet claimed loss and accuracy were shown as the change from the default model, but the live page no longer shows it. **Unverified.**
- **HF AutoTrain:** marked unmaintained.
- **Predibase:** docs redirect to rubrik.com.
- **OpenPipe:** docs do not resolve.
- **NeMo Evaluator:** docs are archived.
- **Modal:** compute only.
- **Unsloth Studio "Model Arena":** a qualitative side-by-side chat with the base.

## 2. Training libraries

**Answer: none ships a held-out general-text-vs-base metric, callback or recipe.**

- **KL is logged only on the library's own rollouts or the training batch.** This held in all 13 libraries checked (shallow clones from 2026-10-14, grepped).
- **Eval hooks score only the task's validation split or a benchmark.**
- **A grep for `forgetting|retention|catastroph` found nothing relevant.** Hits were checkpoint-retention settings, documentation, and tinker-cookbook's SDFT recipe.

| Library | KL and eval hooks | General text vs base? | Overlap |
|---|---|---|---|
| [TRL](https://github.com/huggingface/trl) | GRPO and RLOO log `kl` on sampled completions. GRPO's default `beta=0.0` loads no reference model at all. Online DPO logs `objective/kl` on its own generations. DPO logs `rewards/*` and `logps/*` on the preference pairs. Callbacks: SyncRefModel, LogCompletions, Weave, BEMA, MergeModel (experimental). | No | adjacent |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | `kl` is the masked mean over rollout actions. A PPO-ptx pretraining-mix loss (`--ptx_coef`, default 0.05) existed through v0.10.1 and was removed by v0.10.4. It was mitigation, not measurement. | No | adjacent |
| [verl](https://github.com/volcengine/verl) | KL reward penalty and `kl_loss` on the rollout batch. `rollout_corr_helper.py` reports KL and perplexity between the inference engine and the trainer, which measures numeric mismatch, not drift. SFT reports `val/loss` on `val_files`. | No | adjacent |
| [axolotl](https://github.com/axolotl-ai-cloud/axolotl) | `do_bench_eval` runs MMLU (or "dharma") during training ([training_args_base.py](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/core/training_args_base.py)). It also has `do_causal_lm_eval` and an eval-set perplexity callback. | No. The nearest library retention check, but benchmark-based. | adjacent |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | [`cal_ppl.py`](https://github.com/hiyouga/LLaMA-Factory/blob/main/scripts/stat_utils/cal_ppl.py) gives one model's perplexity on a dataset, and raw text is allowed. Comparing two checkpoints means two runs and a manual diff. Also has `mix_strategy` and `llama_pro.py` (block expansion). | No | adjacent |
| [Unsloth](https://github.com/unslothai/unsloth) | Eval loss and perplexity on the eval set only. | No | unrelated |
| [torchtune](https://github.com/pytorch/torchtune) | Unmaintained. The `eleuther_eval` recipe could run lm-eval's wikitext bpb task, one model at a time. | No | adjacent |
| [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner) / [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) | `init_policy_kl` on rollouts, and `gen_kl_error` for train-vs-generation mismatch. NeMo-Aligner is unmaintained since 2025-05. | No | adjacent |
| [open-instruct](https://github.com/allenai/open-instruct) (Tülu) | `objective/kl{0..3}_avg`: four KL estimators on rollouts. No pretraining mix. | No | adjacent |
| [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | `mismatch_kl/*` and an optional `ref_kl`. SFT logs `val/loss` and `val/perplexity`. | No | adjacent |
| [slime](https://github.com/THUDM/slime) | `kl`, `low_var_kl` and `opd_reverse_kl` on rollouts. | No | adjacent |
| [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) (src-checked) | See the bullets below this table. | Not today, but one config change away | **close in mechanism, on task data** |
| transformers `Trainer` | `eval_dataset` can be a dict, and its metrics are prefixed with each key. This is the natural slot for a `general_text` eval loss, but it gives absolute loss only. | No | adjacent |

tinker-cookbook in more detail:

- [`SamplerNLLEvaluator`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/supervised/nll_evaluator.py) reports `{name}/nll` and `{name}/bpb` on a fixed set of datums. It is attached automatically to the SFT and DPO test split.
- [`rl/metrics.py`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/metrics.py) has `kl_sample_train_*` and `kl_pre_post`.
- Pointing the evaluator at a general corpus would be trivial, but nothing ships that way. It has no base delta and no bootstrap.

## 3. Evaluation, monitoring and model-diffing tools

**lm-evaluation-harness `uncheatable_eval`** ([task](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/uncheatable_eval), [PR #3442](https://github.com/EleutherAI/lm-evaluation-harness/pull/3442), merged 2026-08-28) (src-checked).
- **Measures:** `bits_per_byte`, `byte_perplexity` and `word_perplexity`.
- **Data:** a monthly post-cutoff snapshot, [`Jellyfish042/UncheatableEval-2026-07`](https://huggingface.co/datasets/Jellyfish042/UncheatableEval-2026-07), covering Wikipedia, arXiv, BBC News, GitHub, bioRxiv and AO3.
- **Scope:** the upstream [README](https://github.com/Jellyfish042/uncheatable_eval) says it is "designed to evaluate **base models** only".
- **Differs from ours:**
  - an absolute score for one model;
  - no pairing with a base, no CI;
  - no claim about forgetting;
  - a much larger corpus than our 100 KB.
- **Overlap: close.** It uses the same type of corpus and the same metric for a different job. It is also the obvious drop-in second fresh corpus for us.

Other tools:

| Tool | What it measures | Differs from ours | Overlap |
|---|---|---|---|
| lm-eval perplexity tasks ([paloma](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/paloma), `wikitext`, `pile_10k`, `c4`, `lambada`) | Absolute bpb for one model. `scripts/model_comparator.py` runs a z-test on accuracy between the HF and vLLM backends of one model. `scripts/regression.py` compares code branches. | No compare-to-base mode. No mention of "forgetting", "alignment tax" or "retention" anywhere in the repo. | adjacent |
| [OpenCompass](https://github.com/open-compass/opencompass) `llm_compression` (`BPCEvaluator`) | Bits per character on Common Crawl, Python and arXiv-math (from Huang et al.). | Absolute and aimed at base models. Validated across models, not as a delta between checkpoints. | adjacent |
| HELM and LightEval language-modelling scenarios | `bits_per_byte` on the_pile, twitter_aae and ice (HELM), and on wikitext, the_pile and twitterAAE (LightEval). | Absolute. No retention view. | adjacent |
| MLflow `MetricThreshold` ([validation.py](https://github.com/mlflow/mlflow/blob/master/mlflow/models/evaluation/validation.py)) | Gates a candidate against a baseline model on any metric (`min_absolute_change`, `min_relative_change`). Its built-in perplexity scorer is GPT-2's perplexity on the model's outputs, which measures fluency. | The gate machinery exists, but the signal does not. | adjacent |
| [diffing-toolkit](https://github.com/science-of-finetuning/diffing-toolkit) (88 stars, pushed 2026-10-06) | `diffing/method=kl`: per-token KL between base and fine-tuned on chosen datasets, alongside crosscoder, SAE and activation-lens methods. | Its goal is to explain what changed. No scalar gate, not validated against general ability. | adjacent |
| [mergekit-evolve](https://github.com/arcee-ai/mergekit/blob/main/docs/evolve.md) | Uses lm-eval tasks as the fitness of a merge. Perplexity is allowed with a negative weight. | Absolute, not paired. | adjacent |
| [Compression-Lens](https://huggingface.co/spaces/Jellyfish042/Compression-Lens) (HF Space) | Byte-level loss comparison of Qwen3 and RWKV7 on pasted text. | A demo of two model families, not a base/post pair. It showed a runtime error when checked. | adjacent |
| [flex-mmlu fine-tune-vs-base](https://huggingface.co/spaces/joshause/flex-mmlu-fine-tune-vs-base-model-evaluator) (HF Space); [LLM Diff](https://github.com/novachrono09/Lllmdiff) (hackathon) | MMLU accuracy of a fine-tune vs its base; a "behavioral fingerprint distance" over six behaviour probes. | Benchmark- or generation-based. | adjacent |
| Monitoring products: [Fiddler](https://docs.fiddler.ai/glossary/product-concepts/model-drift), [Evidently](https://docs-old.evidentlyai.com/reference/data-drift-algorithm), Arize Phoenix, WhyLabs LangKit, Galileo, Deepchecks, Arthur, Distributional, Opik, W&B Weave, Neptune | Fiddler defines model drift as the input–output relationship changing over time. Evidently compares column distributions between a reference and a current dataset. Phoenix measures embedding drift against reference data. LangKit computes text metrics on prompts and responses; WhyLabs is discontinued and the repo was last pushed in Nov 2024. Weave uses scorers the user defines. Neptune's standalone service reportedly ended in 2026-03 (not fetched). | All cover data, prediction or app-output drift. **None measures weight drift from fine-tuning.** | unrelated |
| Leaderboards and hubs | Open LLM Leaderboard v1 is archived and its comparator space errors. HF Hub `base_model` metadata ([model cards](https://huggingface.co/docs/hub/model-cards)) only drives the model tree. | No comparison with the base. | unrelated |

## 4. Open-source repos on forgetting and alignment tax

**dlm-sway, probe `external_perplexity`** ([repo](https://github.com/tenseleyFlow/sway), [PyPI `dlm-sway`](https://pypi.org/project/dlm-sway/)) (src-checked).
- **Status:** v0.1.0 released 2026-04-24, last push 2026-05-10, 0 stars.
- **What it measures:** `mean_delta_nats`, the per-token log-prob change from base to fine-tuned, on a bundled public-domain English prose corpus.
  - It reports a `bootstrap_ci` and a z-score against a null built from random-init adapters, which drives a pass/fail assertion.
  - It ships as a CLI and a pytest gate, as one of about 15 "differential testing" probes.
  - Its docstring frames the question as whether the adapter raised perplexity on natural English prose in general.
- **Differs from ours:**
  - built for LoRA or QLoRA adapters trained on one document, not for post-training stages;
  - the prose is likely seen in pretraining (not fresh);
  - reports nats per token, not bpb;
  - the null is a random adapter, not a base-vs-base noise floor;
  - not validated against benchmarks;
  - never run across recipes.
- **Overlap: close.** This is the closest packaged tool.

**nano-llm-posttraining** ([repo](https://github.com/pochenai/nano-llm-posttraining)) (src-checked).
- **Status:** 29 stars, created 2026-07-21, last push 2026-08-04. A non-fork copy exists at [Akgithub2028/Catastrophic-Forgetting-in-LLM-Post-Training](https://github.com/Akgithub2028/Catastrophic-Forgetting-in-LLM-Post-Training).
- **What it is:** a tutorial using SmolLM2-135M on IFEval. Accuracy on the old task did not move, so the README switches to "the perplexity (PPL) of held-out general text", measured with lm-eval word perplexity on WikiText and Pile.

  | Method | WikiText | Pile |
  |---|---|---|
  | In-distribution SFT | +11% | +21% |
  | Off-distribution SFT | +10% | +19% |
  | GRPO | +0.3% | +0.4% |

  It also measures KL on task inputs, reproducing RL's Razor.
- **Differs from ours:** corpora that were in pretraining, word perplexity instead of bpb, no CIs, a 135M model only, a tutorial rather than a tool, and no gating claim.
- **Overlap: close.** An independent small-scale replication of our SFT ≫ GRPO ordering.

Other repos:

| Repo | What it measures | Differs | Overlap |
|---|---|---|---|
| [INK-USC/lm-forgetting-prediction-code](https://github.com/INK-USC/lm-forgetting-prediction-code) (Jin & Ren, [2406.14026](https://arxiv.org/abs/2406.14026), ICML 2024) | Log-perplexity increase on upstream Dolma examples after fine-tuning OLMo-7B. | The goal is to predict which individual examples are forgotten, so they can be replayed. Pretraining data, not fresh text. No checkpoint score. | close in metric, different purpose |
| [tspthomas/slm_stability_cl](https://github.com/tspthomas/slm_stability_cl) ([2606.27634](https://arxiv.org/abs/2606.27634), v2 2026-09-12) | Per checkpoint: KL to base (adapters disabled), entropy and top-token margin on a fixed set of mixed task prompts. | Task prompts rather than general text. Built for LoRA personalization of small models. No gating. | adjacent |
| [hardikmeisheri/AlignmentTax](https://github.com/hardikmeisheri/AlignmentTax) | A base → SFT → DPO → instruct ladder scored on diversity, sycophancy, PopQA long-tail, ECE and lm-eval benchmarks, via `score_model.py --baseline`. | Generation- and benchmark-based, with no likelihood on text. | adjacent |
| [Mari-kn/finetune-regressions](https://github.com/Mari-kn/finetune-regressions) | A deterministic, generation-based CI deploy gate: over-refusal, distinct-n, verbatim memorization. | Same "gate without a judge" spirit, different signal. | adjacent |
| [TRACE](https://github.com/BeyonderXX/TRACE), [ContinualLM](https://github.com/UIC-Liu-Lab/ContinualLM), [spurious-forgetting](https://github.com/zzz47zzz/spurious-forgetting) | TRACE: benchmark deltas (last push 2024-01). ContinualLM: RoBERTa-era continual pretraining. spurious-forgetting: code for the ICLR 2025 result that much forgetting is recoverable. | All need task data. | adjacent |
| [mammoth](https://github.com/aimagelab/mammoth); survey lists ([Wang-ML-Lab](https://github.com/Wang-ML-Lab/llm-continual-learning-survey), [zzz47zzz](https://github.com/zzz47zzz/awesome-lifelong-learning-methods-for-llm)) | mammoth is continual learning for vision. The other two are paper lists. | | unrelated |

## 5. Code releases of the papers in `prior-art.md`

**Answer: none ships a reusable meter.** Each is either experiment scripts or a benchmark harness.

| Paper | Code | What ships | Overlap (as a tool) |
|---|---|---|---|
| RL's Razor (2509.04259) | No official LLM code. [rl_razor_rebuttal](https://github.com/idanshen/rl_razor_rebuttal) contains a README and PDFs. [SakanaAI/rl-razor-mnist](https://github.com/SakanaAI/rl-razor-mnist) is a third-party MNIST-only replication. The same authors' [Self-Distillation](https://github.com/idanshen/Self-Distillation) repo measures forgetting with lm-eval benchmarks. | No meter | adjacent |
| Yang 2026, vocabulary bias (alphaXiv) | None found | n/a. It measures the same quantity on DCLM. | n/a |
| Marek et al. (2605.26097) | [martin-marek/forgetting](https://github.com/martin-marek/forgetting) (26 stars), JAX research code. Its `prev_ntp` metric is next-token loss on the original pretraining distribution of the authors' own small models. | Not packaged | close in metric, unrelated as a tool |
| Harmon et al. (2510.17776) | [post-forget/post-forget](https://github.com/post-forget/post-forget) (0 stars) plus an [HF dataset](https://huggingface.co/datasets/post-forget/post-forget). Shell scripts on LightEval count per-sample 1→0 and 0→1 transitions, with chance adjustment. | Needs benchmarks and reports gain as well as loss. **This is the outcome measure for exp04.** | adjacent |
| Retaining by Doing (2510.18874) | [princeton-pli/retaining-by-doing](https://github.com/princeton-pli/retaining-by-doing) (46 stars). SFT, Self-SFT and RL scripts, plus eval data for IFEval, MMLU and others. | Benchmark-based | adjacent |
| Kalajdzievski (2401.05605) | None found | n/a | n/a |
| Oyama et al. (2502.16173); Kishino et al. ([2505.15353](https://arxiv.org/abs/2505.15353), ACL 2026 Findings) | [shimo-lab/modelmap](https://github.com/shimo-lab/modelmap) (23 stars, pushed 2026-10-07). Ships 10k Pile chunks, log-likelihood matrices for 1,018 models, and a notebook. Its [`kl-scale/`](https://github.com/shimo-lab/modelmap/tree/main/kl-scale) folder estimates KL between models from log-likelihood vectors on a fixed text set, including fine-tuning lineages and 8-bit quantization. | No pip package or CLI. Gives a symmetric distance or map position, not a signed one-sided loss, and has no gating use. | **close** |
| Information Capacity (2511.08066) | [TeleAI-AI-Flow/InformationCapacity](https://github.com/TeleAI-AI-Flow/InformationCapacity) (3 stars), `calc_ic.py` | Scores one model at a time. The base-vs-post comparison exists only as Fig. 4. | adjacent |
| Huang et al. (2404.09937) | [hkust-nlp/llm-compression-intelligence](https://github.com/hkust-nlp/llm-compression-intelligence) (last push 2024-09), [data](https://huggingface.co/datasets/hkust-nlp/llm-compression) | Bits per character on cc, python and arxiv_math, one model per run. Fine-tuned models are excluded by design. | adjacent |

## 6. Compression-as-eval and quantization-KL tools

These tools are the closest structural analogue: a reference model and a candidate model scored on the same fixed general corpus, summarised as one divergence number. All of them are used for quantization.

**llama.cpp `llama-perplexity --kl-divergence-base / --kl-divergence`** ([README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)) (src-checked).
- **How it works:** it saves the reference model's logits on a corpus (Wikitext-2 by convention). It then reports, for a candidate model:
  - KLD;
  - Δppl and the ppl ratio;
  - Δp statistics;
  - the same-top-token rate.
- **Uncertainty:** Gaussian error propagation.
- **Works on fine-tunes:** any model with the same vocabulary is accepted, so a fine-tune can stand in for a quant today.
- **Differs from ours:** built and documented for quantization; the logit files run to tens of GiB; no bootstrap; never validated as a forgetting signal. The same README notes that fine-tunes usually raise perplexity.
- **Overlap: close.** The structurally nearest mainstream tool.

**exllamav3 [`eval/model_diff.py`](https://github.com/turboderp-org/exllamav3/blob/master/eval/model_diff.py) and [`eval/qbench.py`](https://github.com/turboderp-org/exllamav3/blob/master/eval/qbench.py)** (src-checked).
- `model_diff.py` takes any two models A and B on wiki2, the only dataset implemented. It reports per-token KL in both directions, ppl, quantiles, and KLD bucketed by the reference's confidence.
- `qbench.py` adds a self-noise floor, which plays the role of our bootstrap null.
- Used for quantization. **Overlap: close.**

**Marin/Levanter `perplexity_gap`** ([doc](https://github.com/marin-community/marin/blob/main/docs/references/perplexity-gap-analysis.md), [code](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/evaluation/perplexity_gap.py), [PR #4962](https://github.com/marin-community/marin/pull/4962), merged 2026-04-21) (src-checked).
- **How it works:** scores two models on raw-text slices independently of tokenizer, by projecting token losses back onto bytes.
- **Output:** `gap_bpb` per slice and per document, with lists of the worst documents. Default data is Paloma plus Uncheatable Eval.
- **Differs from ours:**
  - built to answer where Marin is worse than another pretrained model;
  - runs as TPU/GCS jobs;
  - gives no uncertainty estimate.
- **Overlap: close in mechanism.**

Others:

| Tool | Notes | Overlap |
|---|---|---|
| [mlx-quant-fidelity](https://pypi.org/project/mlx-quant-fidelity/) (PyPI 0.10.0) | KL, top-token flip rate and Δppl for MLX quantization. | close as a structural analogue, quantization only |
| [ppllm](https://pypi.org/project/ppllm/) (PyPI) | A perplexity and surprisal library. | adjacent |
| Uncheatable Eval upstream ([paper 2609.27510](https://arxiv.org/abs/2609.27510)) | Base models only. Its crawlers are reusable. | adjacent |
| LMCompress, ts_zip | LMCompress has no official code. ts_zip is a closed binary that uses a base RWKV model. | unrelated |

PyPI name probes all returned 404 or unrelated packages: `forgetting`, `llm-forgetting`, `catastrophic-forgetting`, `llm-drift`, `model-drift`, `drift-meter`, `bits-per-byte`, `perplexity-compare`, `kld`, `kl-divergence`, `model-diff`, `uncheatable-eval`, `lmcompress`, `retention-eval`.

## 7. Recent work (Sept–Oct 2026) and industry practice

**Answer: no Sept–Oct 2026 paper, blog, LessWrong post or thread was found that proposes a fresh-general-text forgetting meter, a drift score, or a pretraining-distance gate for post-training.**

| Item | Date | What | Overlap |
|---|---|---|---|
| Intruder Threshold, [2607.23711](https://arxiv.org/abs/2607.23711) | rev. 2026-09-12 | A spectral law predicts the LoRA scale at which WikiText-2 perplexity starts to degrade. Perplexity is the outcome there, not a proposed tool. | adjacent |
| "When Perplexity Lies", [2603.26556](https://arxiv.org/abs/2603.26556) | rev. 2026-09-17 | In distillation, log-likelihood scoring underestimates teacher–student gaps and can reverse the ranking of design choices. **A caveat for any likelihood-based ranking gate; address it in exp04.** | adjacent |
| BALIGN, [2608.24192](https://arxiv.org/abs/2608.24192) | 2026-08-25 | Selects preference data to reduce alignment tax. Measured with benchmarks. | adjacent |
| ACLArena [2609.23989](https://arxiv.org/abs/2609.23989), SafeEvo [2610.09600](https://arxiv.org/abs/2610.09600), DeShortcut-Align [2609.34896](https://arxiv.org/abs/2609.34896), CapTrack [2603.06610](https://arxiv.org/abs/2603.06610) | 2026 | Forgetting or alignment tax measured on tasks or benchmarks. | adjacent |
| [2605.18607](https://arxiv.org/abs/2605.18607) | May 2026 | Token statistics on expert-written solutions rank models across post-training recipes (Spearman 0.81). This is the gain-side complement to our loss-side meter. | adjacent |
| "Quagmires in SFT-RL", [2510.01624](https://arxiv.org/abs/2510.01624) | 2025 | Held-out loss on in-domain reasoning examples predicts the result after RL. Checkpoint selection by held-out loss, but in-domain and on the gain side. | adjacent |
| LessWrong, ["Why does off-model SFT degrade capabilities?"](https://www.lesswrong.com/posts/9toz5YHYZsHpzy4ce/why-does-off-model-sft-degrade-capabilities) | May 2026 | The damage is shallow and recoverable with a small amount of on-model SFT. Relevant to interpreting our +16% for distillation. | adjacent |

Industry practice:

- **OpenAI, InstructGPT ([2203.02155](https://arxiv.org/abs/2203.02155)), Appendix C (PDF-checked).** To initialise RLHF, they pick the SFT learning rate "with low losses on both the demonstration and the pretraining validation datasets".
  - This is the only public statement found of held-out pretraining loss used as a selection criterion in a post-training sweep.
  - Limits: the validation set is in-distribution, it covers one hyperparameter, the text is not fresh, and the choice was not validated.
  - **Overlap: close, as a practice.**
- **Moonshot, Kimi K2 ([2507.20534](https://arxiv.org/abs/2507.20534)).** Adds an auxiliary PTX loss during RL to prevent forgetting. Mitigation, not measurement. **adjacent.**
- **Hugging Face, [SmolLM3](https://huggingface.co/blog/smollm3).** A benchmark caught a RULER regression, which they fixed by merging with a mid-training checkpoint. **adjacent.**
- **NVIDIA, Nemotron-Cascade.** "Regression repair" by on-policy distillation from the best intermediate checkpoint. Seen only in a [secondary blog](https://jxzhangjhu.github.io/blog/2026/how-frontier-labs-train-llms/); the primary source was not verified. Triggered by benchmarks. **adjacent.**
- **[Thinking Machines blog](https://thinkingmachines.ai/blog/).** Nothing beyond the on-policy distillation post already covered in `prior-art.md`.
- **Not found:** any public statement that Anthropic, Google DeepMind, Qwen, DeepSeek, Meta (Llama), Ai2 (Tülu, OLMo) or Databricks track held-out pretraining loss or KL to base as a post-training gate. This rests on targeted searches; not every tech report was read in full.

---

## (a) Closest existing tool

**dlm-sway's `external_perplexity` probe is the closest existing tool.** It is the only packaged tool that computes our quantity the way we do and can gate on it:

- it measures the per-token log-prob change of a fine-tuned model against its base;
- on a fixed corpus of general English prose;
- with a bootstrap CI;
- with a z-score against a null and a pass/fail assertion;
- runnable as a CLI or a pytest gate.

Where it falls short of ours:

- It targets LoRA adapters trained on one document, not post-training stages.
- Its public-domain prose is likely in the pretraining data.
- Its null is a random adapter, not a base noise floor.
- It reports nats per token.
- It has never been run across SFT, DPO, RL or distillation, or checked against benchmark forgetting.
- It has 0 stars and no push since May 2026.

For reach, the nearest mainstream tool is llama.cpp's `--kl-divergence` mode. Anyone can point it at a fine-tune today and get KL and Δppl against the base on Wikitext-2, but it is built and documented only for quantization. Two other items are closest in a single respect:

- **Closest corpus:** lm-eval `uncheatable_eval`. It uses fresh text and reports bpb, but gives one absolute score per model and covers base models only.
- **Closest evidence:** the SFT-vs-GRPO table in nano-llm-posttraining.

## (b) The 5 closest

| # | Tool | What it shares with ours | What it lacks | Overlap |
|---|---|---|---|---|
| 1 | [dlm-sway `external_perplexity`](https://github.com/tenseleyFlow/sway) | Fine-tune vs base log-prob delta on general prose, bootstrap CI, pass/fail gate, packaged CLI | Fresh text, bpb, a base noise floor, post-training scope, validation against forgetting; it has no users | close |
| 2 | [llama.cpp `--kl-divergence`](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md) (with [exllamav3 `model_diff.py`](https://github.com/turboderp-org/exllamav3/blob/master/eval/model_diff.py)) | Reference vs candidate on a fixed general corpus (Wikitext-2), KL and Δppl, accepts any same-vocab model | Fresh text, a bootstrap, any post-training framing or validation; logit files are tens of GiB | close |
| 3 | [nano-llm-posttraining](https://github.com/pochenai/nano-llm-posttraining) | Held-out general-text perplexity vs base as a forgetting measure; same SFT ≫ GRPO ordering | A tool, CIs, fresh text, bpb, scale (135M only), gating, validation | close |
| 4 | [Marin `perplexity_gap`](https://github.com/marin-community/marin/blob/main/docs/references/perplexity-gap-analysis.md) | Paired, tokenizer-independent bpb gap between two models, per document; includes Uncheatable Eval | Uncertainty, the base→post use case, a gate; runs as TPU jobs | close |
| 5 | [lm-eval `uncheatable_eval`](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/uncheatable_eval) | Fresh post-cutoff general text scored in bpb, in the standard harness | Pairing with a base, a delta, a CI, any forgetting claim; base models only by design | close |

Runners-up:

- shimo-lab `modelmap/kl-scale`: KL between models on a fixed corpus, including fine-tuning lineages; symmetric, no gate.
- tinker-cookbook `SamplerNLLEvaluator`: bpb on fixed datums, but on task data.
- InstructGPT Appendix C: pretraining validation loss used for selection, as a practice.
- INK-USC forgetting prediction: log-ppl rise on Dolma examples.

None rates "same thing".

## (c) What remains unbuilt

- **Validation against forgetting.** No tool reports whether its score predicts benchmark forgetting. sway, llama.cpp KL, Marin's gap and tinker's bpb are all unvalidated, and the validated predictor (RL's Razor) uses task data. Still open: Δbpb vs Harmon-style chance-adjusted forgetting, across SFT, DPO, RLVR, GRPO and distillation checkpoints, with rank correlations and CIs (exp04).
- **Fresh, paired and with a CI, all at once.** Fresh corpora exist only as absolute scores (`uncheatable_eval`). The paired tools use Wikitext-2, Pile or public-domain prose. Only sway (bootstrap against a random-adapter null) and exllamav3 (self-noise floor) model noise at all, and neither runs a paired window bootstrap against the base on post-cutoff text.
- **One-sidedness and a gate policy.** Nobody states or tests "drift predicts loss, not gain", or calibrates a threshold on real post-training checkpoints. The gates that exist are safety gates (OpenAI, Azure), benchmark gates (Nova Forge, axolotl MMLU), or generic gates with no signal (MLflow `MetricThreshold`).
- **Separating calibration from drift.** No tool reports a temperature-adjusted Δbpb. A single temperature undoes up to 35% of the gap (exp02), so a raw-perplexity gate mixes overconfidence with forgetting.
- **Integration into training loops and platforms.** transformers' `eval_dataset` dict and tinker's `SamplerNLLEvaluator` are one-config slots for this. No library or platform ships a base-relative general-text eval, and all KL logging is on rollouts or the training batch.

## (d) Risk of being scooped in the next month

**Low for the validated cross-recipe predictor, moderate for a bare tool.** Every component is public (`uncheatable_eval` in lm-eval since 2026-08-28, and Yang 2026 measuring the same quantity on 2026-10-12), but no Sept–Oct 2026 work validates such a score against forgetting across recipes, which takes benchmark runs.

---

## Search log

How the search was run: three parallel searches (platforms and monitoring; libraries, paper code and compression tools; repos, recent work and industry practice), then link and source checks by the coordinator on 2026-10-15.

**Platforms and monitoring**

- Docs fetched for OpenAI, Azure Foundry, Vertex (SFT and RL metrics), Together, Fireworks (SFT, RFT, quantization), Bedrock, Nova Forge and HyperPod, Mistral Forge, Tinker, Lamini, Nebius, Anyscale, Modal, Cohere, Predibase, OpenPipe and NeMo Evaluator.
- Monitoring products checked: Fiddler, Evidently, Arize Phoenix, WhyLabs LangKit, Galileo, Deepchecks, Arthur, Distributional, Opik, W&B Weave, Neptune, MLflow.
- Leaderboards and hubs checked: HELM, OpenCompass, LightEval, Open LLM Leaderboard, HF Hub.
- An extended-mode search for "capability retention" or "forgetting score" fine-tuning platforms.

**Libraries**

- Shallow clones on 2026-10-14 of TRL, OpenRLHF, verl, axolotl, LLaMA-Factory, Unsloth, torchtune, NeMo-Aligner, NeMo-RL, open-instruct, prime-rl, slime and tinker-cookbook.
- Grepped for forgetting, retention, catastroph, KL metric keys, ptx/pretrain, perplexity/bpb and eval hooks.
- Checked `ptx_coef` across OpenRLHF tags v0.8.0 to v0.11.0.

**GitHub repo search** (`gh search repos`, including README-match variants):

- catastrophic forgetting llm
- alignment tax
- forgetting metric llm
- LLM forgetting
- continual learning llm
- model diff perplexity
- finetune regression
- capability regression
- forgetting benchmark
- model diffing
- crosscoder model diff
- llm checkpoint diff
- drift llm checkpoint
- perplexity forgetting fine-tuned

**GitHub code search** (`gh search code`):

- `kl_to_base`
- "forgetting" bits_per_byte sft
- "alignment tax" perplexity
- "held-out perplexity" forgetting
- `general_ppl`
- `pretrain_ppl` sft
- forgetting meter
- drift score
- uncheatable (in Marin)
- forgetting / alignment tax / retention / BPB terms inside lm-eval, HELM, OpenCompass, LightEval and MLflow

**Papers and recent work**

- arXiv HTML link extraction for the code releases of 2605.26097, 2401.05605, 2509.04259, 2511.08066, 2510.18874, 2404.09937, 2609.27510 and 2407.07723.
- arXiv listings sorted newest first for: forgetting + post-training, alignment tax, perplexity + forgetting + fine-tuning, "checkpoint selection" forgetting.
- Extended-mode web searches across HF Papers, alphaXiv, X, LessWrong and lab tech reports, for forgetting meter / drift score / held-out pretraining loss as a gate.

**Checks run for this note**

- HTTP status of every cited link (all 200).
- Source checks: sway `external_perplexity.py`; the nano-llm README table; lm-eval PR #3442 merge date and the Uncheatable README; tinker `nll_evaluator.py`; exllamav3 `model_diff.py`; the Marin doc and PR #4962 merge date; the llama.cpp perplexity README; the InstructGPT Appendix C sentence (PDF); the Nova Forge forgetting text; the modelmap `kl-scale` README.

**Not done or caveats**

- **PyPI:** its web search sits behind a JavaScript bot check, which was not bypassed. PyPI coverage is name probes plus `site:pypi.org` web searches.
- **GitHub:** code search hit its rate limit once; local clones were used instead.
- **Unverified items:**
  - Cohere's "change from default model" metric; the page is now a deprecation notice.
  - Neptune's shutdown; not fetched.
  - Nemotron-Cascade; secondary source only.
  - Lamini's comparison metrics; not specified in its docs.
- **X/Twitter** was searched only through web search, not natively.
- **Lab tech reports** were searched for the relevant practice, not read in full.
- **Note on link checks:** GitHub returns 200 after redirects, so a renamed org (e.g. THUDM/slime) still shows as resolving.

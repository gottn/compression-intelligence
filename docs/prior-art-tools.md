# Prior art: tools and products close to the drift meter

Compiled 2026-10-15 from three search sweeps (fine-tuning platforms and monitoring products; training libraries and paper code releases; open-source repos, Sept–Oct 2026 work, industry practice). Every link below was fetched by the sweep that reported it. Papers are covered separately in `prior-art.md`.

## Verdict

No platform, library, product or code release ships the combination we built: a fixed general-text corpus, a paired bits-per-byte delta against the base model, a bootstrap interval, and use as a post-training gate validated against benchmark forgetting. The quantity exists in scattered form; the practice exists informally; nothing is packaged or validated.

## Closest existing tool

`llama.cpp llama-perplexity --kl-divergence-base / --kl-divergence` is the closest structural analogue: it stores a reference model's logits on a corpus (Wikitext-2 by convention) and reports KL divergence, Δperplexity, top-token agreement and Δp percentiles for a second model with the same vocabulary. It was built to check quantized models against their originals; nothing stops it being pointed at a fine-tune, but it has no bootstrap, needs 11–37 GiB logit files, and has never been validated as a forgetting signal. The closest *purpose-built* item is the `external_perplexity` probe in `dlm-sway` (PyPI, April 2026, 0 stars): base-vs-adapter log-prob change on bundled prose with a bootstrap CI, for LoRA adapters only, nats per token, a random-adapter null, no benchmark validation.

## The five closest

| tool | what it does | differs from ours | overlap |
|---|---|---|---|
| llama.cpp `--kl-divergence` (ggml-org/llama.cpp, tools/perplexity) | KL, ΔPPL, Δp percentiles of model B vs reference A on a corpus | quantization use; huge logit files; no bootstrap; no forgetting validation | close |
| dlm-sway `external_perplexity` (github.com/tenseleyFlow/sway) | base→adapter log-prob change on prose, bootstrap CI, pytest gate | LoRA only; contaminated corpus; nats/token; random-adapter null; no benchmarks | close |
| Marin/Levanter `perplexity_gap` (marin-community/marin, PR #4962) | two-model bits-per-byte gap on raw text with per-document attribution; defaults to Paloma + Uncheatable Eval | built for comparing pretrained models; TPU/JAX; no uncertainty; no post-training use | close |
| Information Capacity code (TeleAI-AI-Flow/InformationCapacity, arXiv 2511.08066) | compression on a mixed text set, with a base-vs-post figure (post-training degrades plain-text prediction) | one figure, no paired statistics, no gating, not tied to forgetting | close |
| lm-evaluation-harness `uncheatable_eval` (PR #3442, 2026-08-28) | bits per byte on monthly post-cutoff snapshots (Wikipedia, arXiv, BBC, GitHub, bioRxiv, AO3) | absolute score, base models only, no base delta, no forgetting claim | close in corpus, different job |

Other structural analogues from quantization: exllamav3 `eval/model_diff.py` and `qbench.py` (per-token KL both directions, self-noise floor), `mlx-quant-fidelity` (KL, top-token flips, ΔPPL). Same mechanism, never used for post-training.

## What platforms and libraries actually ship

- **Fine-tuning platforms** (OpenAI, Azure Foundry, Vertex, Fireworks, Together, Bedrock, Nebius, Anyscale, Modal): training and validation loss on the user's data; post-training *safety* gates (OpenAI `refusals_v3`, Azure harm eval) that block deployment; no capability-retention or base-relative drift score. Amazon Nova Forge is the only one that names forgetting as a product problem and answers it with data mixing plus user-run MMLU/HumanEval/GSM8K comparisons. Cohere, Mistral fine-tuning, AutoTrain, Predibase, OpenPipe are deprecated or gone.
- **Training libraries** (TRL, OpenRLHF, verl, axolotl, LLaMA-Factory, unsloth, torchtune, NeMo-Aligner/RL, open-instruct, prime-rl, slime, tinker-cookbook): KL to a reference is logged only on rollouts or the training batch; eval hooks score the task's validation split or a benchmark (axolotl can run MMLU mid-training). OpenRLHF dropped its PPO-ptx pretraining mix in v0.10.4. tinker-cookbook's `SamplerNLLEvaluator` reports nll and bpb on a fixed datum set and is the easiest hook to point at a general corpus, but ships pointed at the task split with no base delta.
- **Monitoring products** (Arize, Fiddler, WhyLabs, Evidently, Deepchecks, Arthur, Galileo, Distributional, Opik, W&B Weave, MLflow): data, prediction, embedding or app-output drift; MLflow `MetricThreshold` can gate a candidate against a baseline on any metric, but its perplexity scorer is GPT-2 on outputs. None measure weight drift from fine-tuning.
- **Paper code**: RL's Razor has no official LLM code; Yang 2026 none; Marek et al. (`martin-marek/forgetting`) is JAX research code with a `prev_ntp` metric on the authors' own pretraining distribution; Harmon et al. (`post-forget/post-forget`) counts per-sample benchmark flips with LightEval; Jin & Ren 2024 (`INK-USC/lm-forgetting-prediction-code`) predicts per-example log-perplexity increase on Dolma for replay; `shimo-lab/modelmap` (`kl-scale/`) estimates symmetric KL between 1,018 models from log-likelihood vectors on fixed Pile chunks, including fine-tuning lineages.
- **Tutorial precedent**: `pochenai/nano-llm-posttraining` (July 2026) reproduces SFT +10–21% vs GRPO +0.3–0.4% in WikiText/Pile word perplexity on SmolLM2-135M, no intervals, no gating.
- **Industry practice**: InstructGPT appendix C chose the SFT learning rate for low loss on the pretraining validation set and added PPO-ptx; Kimi K2 uses an auxiliary PTX loss; SmolLM3 fixed a benchmark regression by weight-merging. No lab states that it tracks held-out pretraining loss or KL-to-base as a post-training gate.

## What remains unbuilt

- A packaged scorer: frozen reference corpus per tokenizer family, paired Δbpb vs base with a window bootstrap, seconds per checkpoint, CLI plus training-library callback.
- Validation of that number against benchmark forgetting across recipes (ours: threshold above ~15–20% drift, recipe-dependent below; exp04).
- Any decomposition of drift into behaviour and knowledge components (exp05, running).
- A content-weighted KL anchor during reasoning distillation (planned exp06).

## Risk of being scooped within a month

Low to moderate. The ingredients are public and cheap, Yang 2026 and Information Capacity already measure the base-vs-post loss rise, and "When Perplexity Lies" (2603.26556) is a ready-made reviewer objection; but every group measuring the quantity is measuring it for a different purpose, and no one has the decomposition or the staged RL table.

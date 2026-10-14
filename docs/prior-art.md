# Prior art: fresh-text bpb as a post-training drift and forgetting meter

Searched 2026-10-14. Every arXiv id below was checked against its arxiv.org abstract page on that date. The one exception is flagged.

## Thesis under check

Bits per byte (bpb) of a post-trained LLM on fresh, post-cutoff general text, minus its base model's bpb on the same text, is a cheap, task-agnostic measure of post-training drift. The thesis has four parts:

- **C1 (matched table).** Matched base→post-trained bpb across recipes, on fresh text.
- **C2 (predictor).** That drift predicts forgetting of general capability.
- **C3 (one-sided).** It does not predict gain on the target task.
- **C4 (anchor).** A KL-to-base anchor on fresh general text during distillation or RL recovers most of the compression tax and keeps the gain.

Overlap labels used below:

- **full:** the same claim, in the same setting.
- **partial:** the same mechanism or claim in a different setting, or the same setting with a different measure.
- **adjacent:** related and worth citing, but a different claim.

## Verdict table

| Component | Rating | Closest paper |
|---|---|---|
| C1 matched bpb table across recipes on fresh text | **partially published** | Yang 2026, "Is Forgetting in Post-Trained LMs Just a Shift in Vocabulary Bias?" (alphaXiv, 2026-10-12). It covers 14 base→derived pairs on DCLM web text, but only instruct and math/code continued pretraining: no RLVR or reasoning distillation, the text is not post-cutoff, and there is no benchmark link. |
| C2 general-text drift predicts benchmark forgetting | **partially published** | Kalajdzievski 2024 (arXiv 2401.05605) uses cross-entropy to the base model's predictions on WikiText-103 as its forgetting metric, with no correlation to benchmarks. RL's Razor (2509.04259) is the validated predictor, but it measures KL on *new-task* data (R²=0.71 for LLMs). |
| C3 one-sidedness (drift predicts loss, not gain) | **not found** (implicit only) | RL's Razor (2509.04259): the same new-task accuracy is reached at very different KL, and forgetting follows KL. |
| C4 fresh-text KL anchor during reasoning distillation or RL | **partially published** | Marek et al. 2026 (2605.26097) apply a KL penalty on self-generated or pretraining text during finetuning, which almost eliminates forgetting. Their settings are Verilog SFT and math continued pretraining, not R1-style distillation or RLVR, and they report no bpb tax. |

---

## Q1. Matched base→post-trained bpb/perplexity tables across recipes

**Answer.** Partial tables exist for instruct/SFT/RLHF pairs and for continued pretraining. **None found** that covers SFT, DPO, RLHF, RLVR *and* reasoning distillation with per-model numbers on held-out post-cutoff general text. For RLVR, the only perplexity numbers found are on task text (MATH-500).

| Paper | Finding | Overlap |
|---|---|---|
| Yang, "Is Forgetting in Post-Trained Language Models Just a Shift in Vocabulary Bias?", preprint 2026-10-12. Available only on [alphaXiv](https://www.alphaxiv.org/abs/2610.language-model-forgetting-vocabulary-bias): **the arXiv id was not resolvable on 2026-10-14** (too new; listing pages were rate-limited). | 14 base→derived pairs (OLMo-2, Qwen2.5, SmolLM2; 1B–7B) for instruction tuning and math/code continued pretraining. Loss increases on 1M held-out DCLM tokens, e.g. 0.354 nats/token for OLMo-2-7B→Instruct and 0.796 for Qwen2.5-1.5B→Math. The pairs include Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct, the same as ours. No benchmarks, no RL or distillation, and the text is not post-cutoff. | **partial (closest)** |
| Kuribayashi, Oseki, Baldwin, "Psychometric Predictive Power of LLMs", Findings NAACL 2024, 2311.07484 | 6 matched pairs (Llama-2 7/13/70B vs chat, Falcon 7/40B vs instruct, davinci-002 vs text-davinci-003). Instruction tuning raises perplexity on reading-time corpora, e.g. Llama-2-7B goes from 76.4 to 153.5. | partial |
| Yuan, Shao, Li, "Information Capacity", 2511.08066 | Fig. 4 shows 5 base/post families (Qwen3, Qwen2.5, Llama-3, InternLM2.5, Seed-OSS) on non-fresh corpora (FinePDFs, FineWeb-Edu). The only number in the text is Seed-OSS-36B information capacity, 0.17633 for base and 0.15995 for instruct. They state that "advanced RL" degrades compression more. There is no per-recipe table. | partial (confirmed: figure only) |
| Li, Guo, Guerin, Lin, "Evaluating LLMs for Generalization and Robustness via Data Compression", 2402.00861 | Compression on post-cutoff text (2017–2023) for 14 mostly base models. Notes that CodeLlama (domain-trained Llama-2) generalizes worse than base Llama-2. | adjacent (fresh text; one domain pair) |
| Tan, Li, Shen, "Uncheatable Eval", 2609.27510 | Fresh-text compression for 80 models. **Base models only by design**: "compression rate does not capture the instruction-following abilities acquired through post-training". | adjacent; leaves C1 open |
| Oyama et al., "Mapping 1,000+ LMs via the Log-Likelihood Vector", ACL 2025, 2502.16173; follow-up 2603.18593 | Log-likelihood vectors on Pile texts for 1,018 models (base and fine-tuned). Squared distance ≈ KL between models. Ridge regression on the vectors predicts benchmark scores (r=0.88–0.95). There is no base→derived drift analysis. | adjacent (closest *method* for cheap model-to-model KL) |
| Yan, Li, Chen, Li et al., "Spurious Rewards Paradox", ICML 2026, 2601.11061 | Spurious RLVR raises full-text perplexity on MATH-500 for Qwen2.5-Math, LLaMA and OLMo. Answer-token perplexity falls only for Qwen2.5-Math. | adjacent (RLVR perplexity, but on task text) |
| Harmon, Hochlehnert, Bethge, Prabhu, "Mapping Post-Training Forgetting in LMs at Scale", ICLR 2026, 2510.17776 | About 30 base→post pairs (domain continued pretraining, instruct, reasoning SFT/RL incl. R1-Distill, QwQ). Uses a chance-adjusted, sample-wise benchmark forgetting metric. No loss or KL proxy. Reasoning training from base shows low-to-moderate forgetting and backward transfer. | adjacent; this is the natural **outcome** to correlate against |
| llama.cpp `tools/perplexity/README.md` (community) | One line, with no table: "finetunes typically result in a higher perplexity value even though the human-rated quality of outputs increases." | adjacent (folk knowledge) |

## Q2. General-text loss, compression or KL-to-base as a *predictor* of benchmark forgetting

**Answer.** **None found** that validates held-out general-text (non-task) loss or KL-to-base against downstream benchmark forgetting with a correlation coefficient across post-training recipes.

The validated predictors all measure KL on the **task** distribution:

- RL's Razor: R²=0.96 on the toy task, R²=0.71 for LLMs.
- Retaining by Doing: Pearson r=0.52.

Papers that do use general-text loss as the forgetting measure treat it as the outcome itself, or check it only qualitatively against ARC.

| Paper | Finding | Overlap |
|---|---|---|
| Kalajdzievski, "Scaling Laws for Forgetting When Fine-Tuning LLMs", 2401.05605 | Forgetting metric = cross-entropy to the base model's predictions on **WikiText-103**, i.e. general text that is not fresh. It is argued to be a continuous surrogate and shown next to ARC accuracy per checkpoint, with **no correlation coefficient**. LoRA on Llama-2-7B-chat. Forgetting is a power law in steps and parameters, and *linearly coupled* to fine-tuning loss. | **partial (closest for general text)** |
| Shenfeld, Pari, Agrawal, "RL's Razor", ICLR 2026, 2509.04259 | Forward KL(π0‖π) **on new-task inputs** predicts forgetting across SFT and RL (quadratic fit R²=0.96 toy, 0.71 LLM). Alternatives (weight change, activation change, reverse KL, TV) were tested only on the MNIST toy. KL on general text was not tested. | **partial (closest validated predictor)** |
| Chen, Razin, Narasimhan, Chen, "Retaining by Doing", ICML 2026, 2510.18874 | KL from the initial policy, estimated on 100 eval examples, vs drop on non-target tasks: Pearson **r=0.52** across Llama/Qwen, SFT/Self-SFT/GRPO. Weak for SFT vs Self-SFT. | partial |
| Huan et al., "Does Math Reasoning Improve General LLM Capabilities?", ICML 2026, 2507.00432 | Over 20 reasoning models. KL and token-rank shift vs backbone, measured on benchmark prompts. SFT drifts and forgets, RL preserves. Qualitative, **no r**. | partial |
| Béthune et al., "Scaling Laws for Forgetting during Finetuning with Pretraining Data Injection", ICML 2025, 2502.06042 | Forgetting = pretraining validation loss after domain finetuning of base models. A scaling law predicts it. ARC-easy is checked qualitatively, with no correlation. | partial (outcome, not predictor) |
| Marek et al., "Forgetting in LMs: Capacity, Optimization, and Self-Generated Replay", 2605.26097 | Tracks pretraining loss and MMLU/CSQA/ARC-C. No correlation analysis. | adjacent for C2 (see Q4) |
| Gan et al., "GFT", 2604.14258 | Uses average KL to base on the *training data* as "a proxy for distributional drift, and hence forgetting", citing it as established. | adjacent (KL-as-proxy is folk practice) |
| Zibakhsh Shabgahi et al., "Beyond Perplexity: A Lightweight Benchmark for Knowledge Retention in SFT", 2601.03505 | Argues validation perplexity "confounds stylistic mimicry with genuine factual internalization". | **counter-evidence to address** |
| Zheng et al., "Spurious Forgetting", ICLR 2025, 2501.13453; Palit et al., "When Forgetting is not Catastrophic", 2610.08718 | Much benchmark "forgetting" is a reversible loss of task alignment or access, not knowledge loss. | counter-evidence (forgetting outcome is noisy) |
| Sanati et al., "Forgetting is Everywhere", 2511.04666 | Task-agnostic definition: forgetting as loss of self-consistency in the predictive distribution, with a "propensity to forget" measure. | adjacent (theory) |

## Q3. One-sidedness (drift predicts loss, not gain)

**Answer.** **None found** as an explicit, tested claim. It is implicit in two places:

- **RL's Razor:** RL and SFT reach equal new-task accuracy at very different KL, and forgetting tracks KL, not accuracy.
- **Retaining by Doing:** comparable gain with less forgetting.

Some papers report the *opposite* coupling:

- **Kalajdzievski 2024:** an "inverse linear relationship" between fine-tuning performance and forgetting.
- **Lin et al., "Mitigating the Alignment Tax of RLHF"** (EMNLP 2024, 2309.06256): higher RLHF reward comes with a higher alignment tax, along a Pareto front.

Two adjacent decoupling results:

- **Scholz et al., "Fine-Tuning LLMs for Translation: General Forgetting Mitigation Does Not Preserve MT-Specific Instruction Following"** (2609.28395): retention on general benchmarks does not imply retention of the task-specific behaviour.
- **Ruan et al., "Unveiling Over-Memorization in Finetuning LLMs for Reasoning Tasks"** (2508.04117): high test perplexity coexists with high test accuracy in reasoning SFT.

## Q4. Anchor or regularizer beyond PPO-ptx and RLHF KL, during distillation or RLVR

**Answer.** Anchors on general or replay data during SFT and continued pretraining are **published**, with measured trade-offs. During RLVR there is replay-based work. **None found** for R1-style long-CoT distillation with a KL-to-base anchor on general text that measures both the general-text compression tax and the reasoning gain.

| Paper | Finding | Overlap |
|---|---|---|
| Marek, Cho, Qiu, Chunara, Izmailov, Wilson, 2605.26097 | KL penalty on **self-generated** text or on OLMo pretraining data during finetuning. Settings: math continued pretraining of base models, and Verilog SFT of Llama-3.2-1B-Instruct. "KL regularization almost entirely eliminates forgetting". KL beats next-token-prediction replay. Fails when the model is near capacity (overtrained). | **partial (closest)** |
| Riemer, Miehling, Liu, Bouneffouf, Campbell, "Approximate Regularized Replay", 2512.22337 | KL to the initial model plus next-token-prediction replay from an open corpus similar to pretraining, during SFT of Qwen instruct models (1.5–14B). Forgetting measured on a 6-task lm-eval average. Small overhead "virtually eliminate[s]" forgetting. | partial |
| Béthune et al. 2025, 2502.06042 | Injecting 1% pretraining data during domain finetuning prevents pretraining-loss forgetting, with a scaling law. | partial (replay, not KL; not distillation) |
| Phan et al., "RECAP", 2510.21978 | Replay of general data plus dynamic objective reweighting during **RLVR** (Qwen2.5-VL 3B/7B). | partial (RLVR, replay, VL) |
| Luo et al., "RL Forgets! Towards Continual Policy Optimization", 2607.04364 | Argues standard RL KL is on current-task data while forgetting is drift on prior distributions. Proposes a prior-task behavioural KL. | partial (same diagnosis, prior-task data instead of fresh general text) |
| Wang et al., "CoKL", 2608.01743 | KL restricted to correctness-conditioned responses in RL, for retention. | adjacent |
| Zhu et al., "Anchored SFT (ASFT)", ICLR 2026, 2509.23753 | DFT plus KL to base **on training data**, for math, medical and code SFT. Explicitly does *not* measure forgetting (their Appendix B). | adjacent |
| Zhu et al., "Proximal SFT", ICLR 2026, 2508.17784 | PPO-style clipped SFT objective. Better out-of-domain results, no entropy collapse. | adjacent |
| Wang et al., "Anchored Learning", 2605.04468 | Moving anchor between the current model and a frozen reference. Forgetting reduced from about 53% to under 5% on iGSM/MedCalc. | adjacent |
| Yang et al., "SDFT", ACL 2024, 2402.13669; Shenfeld et al., "Self-Distillation Enables Continual Learning", 2601.19897 | Self-distilled or on-policy targets reduce forgetting, measured on a 6-benchmark average. | adjacent (data-side anchor) |
| Lin & Han, "SPOT", 2603.01683 | Proximal on-policy reasoning distillation. Argues the KL-constrained reward "plays a critical role in retaining knowledge" (Qwen3-8B). | adjacent (reasoning distillation, task-data anchor) |
| Thinking Machines, "On-Policy Distillation" blog, 2025-10-27 | Qwen3-8B IF-eval falls from 85% to 45% after document midtraining (79% with 30% chat mix). On-policy distillation from the original model restores 83%. | adjacent (recovery after the fact) |
| Lin et al., "Mitigating the Alignment Tax of RLHF", EMNLP 2024, 2309.06256 | Weight averaging between pre- and post-RLHF models gives the best alignment-forgetting Pareto front. | baseline to beat |
| Biderman et al., "LoRA Learns Less and Forgets Less", TMLR 2024, 2405.09673 | LoRA forgets less but learns less (code and math, instruction tuning and continued pretraining). | baseline to beat |
| Scholz et al., 2609.28395 | Compares KL-to-base (computed on SFT data), Tülu replay, token masking and other mitigations. General retention ≠ task-specific retention. | adjacent |

## Q5. Compression or bpb of instruct and reasoning models specifically

**Answer.** **None found** that reports bpb of R1-style distilled or RLVR reasoning models on general text, beyond Information Capacity's Fig. 4. That figure includes Qwen3 (hybrid thinking) and Seed-OSS-Instruct (RL); the authors say advanced RL degrades compression more.

The compression-as-intelligence literature excludes post-trained models by design:

- **Huang et al., COLM 2024 (2404.09937):** "fine-tuned models are no longer general-purpose compressors". They suggest relating base-model compression to the fine-tuned model's benchmarks as future work.
- **Uncheatable Eval (2609.27510):** base models only.
- **Delétang et al., "Language Modeling Is Compression", ICLR 2024 (2309.10668):** base and Chinchilla models.
- **ts_zip:** uses a base RWKV.

Not checked: which models LMCompress (2407.07723) uses. No Hutter-prize-style discussion of chat models as compressors was found.

## Q6. Calibration or temperature as the explanation for post-training perplexity increases

**Answer.** **Partially answered:**

- Temperature fixes RLHF *calibration* on multiple-choice questions (Kadavath).
- For general-text *loss*, temperature plus a vocabulary bias recovers only up to 36% of the increase on overconfident instruct models, and about 1% for math continued pretraining (Yang 2026).
- So most of the loss increase is not a calibration artefact.

**None found** on whether RLVR entropy collapse raises held-out general-text perplexity. The entropy papers measure entropy on policy rollouts.

| Paper | Finding |
|---|---|
| Kadavath et al., "LMs (Mostly) Know What They Know", 2207.05221 | RLHF policies are badly calibrated, and T=2.5 "largely fixes" it on MMLU, TruthfulQA and True/False. |
| OpenAI, "GPT-4 Technical Report", 2303.08774 | Post-training hurts MMLU calibration relative to the pretrained model. |
| Zhu et al., "On the Calibration of LLMs and Alignment", Findings EMNLP 2023, 2311.13240 | Calibration degrades during alignment; LoRA mitigates the degradation. |
| Yang 2026 (alphaXiv, see Q1) | Bias alone recovers under 10% of the web-text loss increase. Bias plus temperature recovers up to 36% (OLMo-2-7B-Instruct 35.9%; Qwen2.5-1.5B-Instruct 5.8%; SmolLM2-1.7B-Instruct 3.6%; Qwen2.5-Math about 1%). |
| Luo et al., "Dual-Align", 2601.04277 | Splits calibration loss into "confidence drift" (fixable by temperature) and "process drift" (layer-wise divergence). Uses the pretrained model as the unsupervised calibration reference. |
| Han et al., "A Deafening Silence", 2610.09835 | Forgetting concentrates in output embeddings of tokens absent from the new data. Raising Adam ε on the output layer cuts forgetting by 39–68%. A mechanism for broad, vocabulary-level drift. |
| Xie et al., "Adaptive Temperature Scaling", 2409.19817; Leng et al., "Taming Overconfidence in LLMs", 2410.09724 | Post-RLHF overconfidence is input-dependent, so a single global T is insufficient. |
| Cui et al., "Entropy Mechanism of RL for Reasoning LMs", 2505.22617 | RLVR entropy collapse, measured on rollouts, not held-out text. |
| Du et al., "How Post-Training Reshapes LLMs", COLM 2025, 2504.02904 | Knowledge storage is unchanged by post-training. Confidence differences are not explained by entropy neurons. |
| Information Capacity, 2511.08066 | Temperature ablation (T = 0.7, 1, 1.3) on **base** models only; T=1 is best. |
| Lin et al., URIAL, ICLR 2024, 2312.01552 | Base vs aligned token distribution shift is concentrated on stylistic tokens (measured on alignment-style responses). |

## Known context: confirmed and extended

- **Huang et al. 2404.09937 (COLM 2024).** Base models only (31 models), r≈−0.95. Fine-tuned models were excluded on purpose.
- **Information Capacity 2511.08066.** Confirmed figure-only (Fig. 4), on non-fresh corpora, with one numeric Seed-OSS comparison. Also asserts that base-model pretraining loss "still provides a reliable precursor" of the instruct model's downstream performance.
- **RL's Razor 2509.04259 (ICLR 2026).** KL on new-task inputs. LLM fit R²=0.71. Alternative predictors were tested on the toy task only.
- **InstructGPT 2203.02155.** PPO-ptx. Raising the KL coefficient alone did not recover the regressions; the pretraining-mix term did.
- **Morris et al., "How much do language models memorize?"** The correct id is **2505.24832**. The id 2505.24142 is an unrelated physics paper.
- **Delétang et al. 2309.10668 (ICLR 2024).** Confirmed.

## Risks this search surfaced for the research thread

- **The forgetting outcome may not move where bpb moves.** Harmon et al. (ICLR 2026) find reasoning SFT/RL from base gives only low-to-moderate benchmark forgetting, plus backward transfer. Our largest Δbpb (+15.8% for R1-Distill) may therefore pair with small forgetting, which threatens the |r|>0.8 gate.
- **Our R1 pair's base is already drifted.** Qwen2.5-Math-1.5B is itself heavily drifted from Qwen2.5-1.5B (0.796 nats/token on DCLM per Yang 2026). The R1 pair should also be scored against the general Qwen2.5-1.5B base.
- **Part of Δbpb is calibration.** Up to about 36% for instruct pairs. Any predictor claim must use temperature-optimized bpb, or show that the residual still predicts forgetting.
- **Reviewers will raise the perplexity objection.** "Perplexity ≠ retention" (2601.03505) and spurious-forgetting results (2501.13453, 2610.08718) will be cited against a perplexity-based meter.

## Novelty verdict

| Component | Rating | Single closest paper |
|---|---|---|
| C1 matched table | partially published | Yang 2026, vocabulary-bias paper (alphaXiv; 14 pairs, DCLM, instruct and continued pretraining only) |
| C2 fresh-text drift predicts forgetting | partially published | Kalajdzievski 2024, 2401.05605 (WikiText cross-entropy to base as forgetting metric; no r). The validated predictor is RL's Razor, on task data. |
| C3 one-sidedness | not found | RL's Razor, 2509.04259 (implicit only) |
| C4 fresh-text anchor during distillation/RL | partially published | Marek et al. 2026, 2605.26097 (KL on self-generated or pretraining text; not reasoning distillation or RLVR) |

Net: no single paper states the thesis. The open, defensible core is C2+C3: a *task-agnostic*, *post-cutoff* general-text drift score, validated against benchmark forgetting **across recipes including RLVR and reasoning distillation**, that is uncorrelated with gain. C1 alone is not enough for a paper. C4 is an incremental method unless it beats the existing anchors.

## What would make it publishable

1. **Head-to-head predictor study.** Use one base with at least 4 recipes (SFT-chat, long-CoT distillation, DPO, GRPO) × at least 5 checkpoints, plus public pairs (Qwen3, R1-Distill, OLMo-2/3 SFT→DPO→RLVR, Tülu 3). Report the rank correlation, with CIs, between Δbpb and benchmark forgetting. Use Harmon et al.'s chance-adjusted forgetting as the outcome. Compare against:
   - KL on task data (RL's Razor);
   - cross-entropy to base on WikiText (Kalajdzievski);
   - pretraining-loss proxies.

   The claim must be "as predictive at lower cost, and task-agnostic".
2. **Calibration control.** Report raw bpb, temperature-optimized bpb, and bias-plus-temperature bpb (per Yang 2026). Show which residual predicts forgetting. This pre-empts the obvious "it's just overconfidence" objection.
3. **Show that "fresh" matters.** Run the same analysis on pre-cutoff text (WikiText, DCLM) and on post-cutoff text. If the post-cutoff version predicts better or more robustly (less memorization confound), the "fresh" qualifier earns its place; otherwise drop it.
4. **Formal one-sidedness test.** Show Δbpb vs forgetting r≫0 and Δbpb vs gain r≈0, with an equivalence test or partial correlation. Include a manipulation where drift is changed at fixed gain (anchor on/off) and gain at fixed drift.
5. **Anchor on the Pareto frontier.** Run a fresh-text KL anchor during R1-style distillation and GRPO. Plot reasoning gain, general forgetting and Δbpb together, against the strongest existing baselines:
   - self-generated-text KL (Marek);
   - KL plus open-corpus replay (Riemer);
   - 1% pretraining injection (Béthune);
   - task-data KL (ASFT);
   - LoRA;
   - weight averaging (Lin et al.).

   Novelty requires the fresh-text anchor to dominate at least one of them at equal compute.

## Search log

**Queries run (web search, standard and extended modes):**

- alignment tax perplexity
- instruction tuning perplexity on held-out text
- base vs instruct vs reasoning bpb
- reasoning distillation forgetting
- KL to base on general text predicts forgetting
- held-out pretraining loss as forgetting proxy
- scaling laws for forgetting
- RL's Razor follow-ups
- continual learning surveys
- post-training drift metric
- KL anchor in long-CoT distillation
- pretraining replay in RLVR
- PSFT, ASFT, SDFT
- KL on general unlabeled text
- compression of chat models (ts_zip, LMCompress, Uncheatable Eval)
- "compression tax"
- post-cutoff loss of instruct and reasoning models
- temperature recovering perplexity
- calibration after RLHF
- entropy collapse and held-out perplexity

**arXiv listings.** The cs.CL past-week listing was scanned for October 2026 forgetting papers.

**Not done:**

- Google Scholar, Semantic Scholar and OpenReview were not queried directly; coverage comes from the web-search engine and arXiv pages.
- The arXiv API and the cs.LG listing were rate-limited, so the vocabulary-bias paper's arXiv id is unresolved.
- Dropped as low-credibility or irrelevant: 2601.18699 (claims analysis of 109B–1.5T models; correlation claim not in the abstract) and 2504.07158 (withdrawn).

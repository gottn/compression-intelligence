# Research thread: fresh-text compression as a control signal for post-training

Drafted 2026-10-14 from exp01 results. Compute: Mac M2 Pro (scoring, analysis), AAU A100 cluster (training, full evals; needs VPN).

## Claim under test

Bits per byte (bpb) on fresh post-cutoff text measures how far post-training moved a model from its pretraining distribution (exp01). If that drift predicts lost general capability, it is a forgetting meter that costs ~10 GPU-seconds instead of ~2–3 GPU-hours per checkpoint.

## Why it matters (cost)

A post-training sweep of 20 configs × 5 checkpoints = 100 full evals ≈ 200–300 GPU-hours on a 1.5B model. The same 100 bpb scores ≈ 20 minutes. Valid only if the cheap number predicts the expensive one.

## Chain with gates

1. **Drift predicts forgetting.** Fine-tune Qwen2.5-1.5B (or OLMo-2-1B, public data) with 4–6 recipes: SFT general chat, SFT long-CoT (OpenR1-Math subset), DPO, GRPO on math. Save 5 checkpoints each. Score bpb drift (Mac, exp01 pipeline). Run full evals once (AAU, lm-eval-harness: MMLU, ARC, HellaSwag, GSM8K, HumanEval).
   Gate: |r| > 0.8 between drift and loss on general benchmarks (MMLU, ARC, HellaSwag) across checkpoints.
2. **One-sided meter.** Check drift predicts what is lost, not what is gained (GSM8K/MATH gains should be uncorrelated with drift). A one-sided meter is what recipe design needs.
3. **Anchor loss buys back the tax.** Add KL-to-base on general fresh text during long-CoT distillation.
   Gate: keep ≥ 90% of MATH gain, cut bpb tax from ~16% to < 5%.
4. **Shocking branch.** Any RL recipe that *lowers* fresh-text bpb below base means RL teaches general language modelling. Would reverse current assumptions.

## First AAU run

- Base: Qwen/Qwen2.5-1.5B. Recipes: SFT-chat (e.g. smoltalk subset), SFT-CoT (OpenR1-Math ~20k), DPO (ultrafeedback subset), GRPO-math (gsm8k/MATH train, verifiable reward).
- 5 checkpoints per recipe → 20 checkpoints. Training ≈ 15 GPU-hours. Evals ≈ 40–60 GPU-hours.
- Copy checkpoints to Mac (or score on AAU with score_bpb.py). Corpora: exp01 data/ (wiki2026, arxiv2026, gutenberg2026, enwik8_ctrl).
- Output: per-checkpoint table (recipe, step, bpb per corpus, Δbpb vs base, benchmark scores), scatter of Δbpb vs ΔMMLU and vs ΔGSM8K.

## Alternatives considered

- **50 KB benchmark:** smallest corpus/token subset whose bpb predicts benchmarks during pretraining. Cheaper monitoring; novelty thin (labs watch held-out loss) unless token-subset effect is large.
- **Compression-gain data selection for fine-tuning:** PreSelect-style at fine-tune scale. Published Entropy Law gain only 0.1–0.2 MT-bench points; expect modest effect.

## Open items before running

- Verify HF ids for staged RL checkpoints (OLMo-2-0425-1B SFT/DPO/Instruct; hkust-nlp SimpleRL-Zoo 0.5B/1.5B) — using them would skip step 1's training for the RL arm.
- Decide eval subset sizes to keep eval under 60 GPU-hours.

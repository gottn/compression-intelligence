# Independent review: "The Code Length of Post-Trained Language Models on Fresh Text"

Reviewed 2026-10-16 against `paper/main.tex` and `paper/main.pdf` (15 pages, about 8,000 words) at commit 3c4916b.
Every number below was recomputed with `.venv/bin/python` from the committed CSVs, per-token `.npy` arrays and
lm-eval `summary.json` files. The scripts are in the reviewer's scratchpad, outside the repository. The test suite
was run with caches disabled (96 passed, ruff clean). `git status` was clean before and after. Nothing in the
repository was modified except this file.

---

## 1. Numerical integrity

### 1.1 Headline numbers recomputed from the raw files

| # | Paper claim | Recomputed | Source | Status |
|---|---|---|---|---|
| 1 | OLMo-2 SFT Δfresh +6.99 [6.61, 7.40] | +6.993 [6.60, 7.40] (own bootstrap seed) | exp03 `tokens/*.npy`, `windows.csv` | ok |
| 2 | DPO +14.10; DPO step +6.65; RLVR step +0.49 [0.44, 0.55] | 14.102; 6.645; 0.493 [0.434, 0.554] | same | ok |
| 3 | SimpleRL +0.18 / +0.12; "8.7 and 7.8 times less"; RLVR "fourteen times" | 0.184 / 0.117; 8.74 / 7.78; 14.2 | same | ok |
| 4 | Math +23.84, R1 +42.95, R1 vs Math +15.81 | 23.843, 42.953, 15.805 | same | ok |
| 5 | 15 of 15 comparisons have every per-corpus CI above 0 | 15 / 15 (`f1_counts`) | `make_figures.f1_data` | ok |
| 6 | Qwen2.5-1.5B codes wiki in 73,191 bits (0.719 bpb); Instruct 73,674 bits (+0.66%) | 73,191.2; 0.71867; 73,673.8; +0.659% | exp01 `bpb.csv` | ok |
| 7 | Qwen3 chat-template drift +37.0% | +37.04% | exp01 `bpb.csv` | ok |
| 8 | Temperature: recovered 15 / 6 / 35 / 19%; residual +0.9 / +3.6 / +10.4 / +12.7% | 15.0 / 6.1 / 35.3 / 18.9; 0.91 / 3.57 / 10.38 / 12.73 | exp02 `grid/*.npy` | ok |
| 9 | Gate 1 r = 0.887, ρ = 0.671; without the math line 0.32 / 0.43; LOO 0.69–0.93; enwik8 r = 0.949, ρ = 0.678 | 0.8873, 0.6713; 0.3167 / 0.4303; 0.693–0.925; 0.9487 / 0.6783 | exp04 `scores.csv` (forgetting rebuilt from per-task accuracies) | ok |
| 10 | IFEval chat gain r = 0.89, ρ = 0.88 (n = 8) | 0.8918, 0.8810 | same | ok |
| 11 | Per-checkpoint forgetting; floor about 2.1% (1.9–2.5%) | max abs difference 5.6e-4 pp; floors 1.92–2.52, median 2.11 | same, from per-task SEs | ok |
| 12 | exp05: r(Δ_K, Δ_S) = 0.925; contrast vs forgetting 0.92 / 0.81 / gain 0.17; R−F 0.82 / 0.74 / −0.07; H2 0.805 / 0.844 | 0.9246; 0.9189 / 0.8056 / 0.1647; 0.818 / 0.736 / −0.066; 0.805 / 0.844 | **recomputed from raw per-token bits and `classes/*.npz`**, not from `decomposition.csv` | ok |
| 13 | exp06 forgetting per checkpoint (Table 6) | max abs difference 4.9e-4 pp from the six accuracies | exp06 `checkpoints.csv` | ok |
| 14 | H1 0.43; H2 −0.41; r(K−S, Δ) 0.61 over marks / 0.80 over 12; all-12 H1 0.29; Δ vs forgetting −0.04 | 0.4310; −0.4045; 0.6088 / 0.7996; 0.2884; −0.0383 | same | ok |
| 15 | H3 ranges 5.04 and 5.83 pp against a floor of 2.22 | 5.04, 5.83; floors 2.21–2.22 | same | ok |
| 16 | Epoch jumps: tulu +0.51, +0.91, +0.73 (4.06 → 5.70); smoltalk +1.88 (6.19 → 8.07) | identical | `trajectories.csv` | ok |
| 17 | Drift "a median of 19 s on one GPU", "67 times" cheaper | **19 s is the median over 18 rows that mix T4 (10–33 s) and M2 Pro MPS (60–64 s)**. The T4-only median is 15.0 s. "67×" is the median of per-model ratios across both hardware types. | exp03/exp01 `bpb.csv`, exp04 `summary.md` | misdescribed (see 1.2) |
| 18 | "enwik8 drift is 1.3 to 3.0 times the fresh drift, the one exception being R1-Distill against its parent (0.6)" | **RLVR against its DPO parent is a second exception: 0.356 / 0.493 = 0.72** | exp03 tokens | wrong |
| 19 | "mathcot drifts smoothly"; "Both marks of tulu were reached just after a pass boundary" | mathcot jumps +0.46 pp in 20 updates at pass 3.02 (5.61 → 6.06, loss 0.454 → 0.390), and **that jump created its 6% mark**. smoltalk's 3% mark (update 920, pass 1.02) follows a +0.27 pp jump. 4 of the 6 marks fall within 0.05 passes after a boundary. Fig. 4a shows the mathcot step. | `trajectories.csv` | wrong |
| 20 | "one temperature removes at most a third" (abstract) | 35.3% | exp02 | wording |
| 21 | "Within each recipe the contrast grows with the dose (r = 0.80 … over the 12)" | 0.80 is the pooled r, not a within-recipe one. Within each recipe the sign is positive. | exp06 | wording |
| 22 | Rare − frequent order "by a margin of 0.07 pp" | 0.064 | exp06 | rounding |

**Verdict on integrity.** No reported number is fabricated or mis-transcribed beyond rounding. The tables in
`main.tex` are typed by hand. `make_figures.py` checks against constants copied from RESULTS.md, not against
`main.tex`. I checked Tables 3–6 against the CSVs by hand and found them correct. One rounding slip is in
`exp06_matched_drift/RESULTS.md`, not in the paper: ARC-e 0.70749 is printed as 0.708. The errors are in the
prose (items 17–19).

### 1.2 The cost ratio compares the wrong things

The forgetting outcome that the drift is supposed to replace needs only the `loglik` group. Across the 18
exp04 checkpoints that group has a median of **167 s** (92–273 s, from `exp04_benchmarks/results/lm_eval/*/summary.json`).
gsm8k and IFEval take about 85% of the 18.2-minute "benchmark pass", and the paper itself says drift does not
predict either of them. Against the loglik group the measured ratio is about **11×** (167 s on an A10 against 15 s on a T4).
In the exp06 setting it is about **31–39×** (a drift score takes about 5 s on the A40; loglik takes 157–195 s on the
A10). Drift is still cheap. But the "screen a sweep at a small fraction of its evaluation cost" motivation in the
introduction is overstated by roughly 2–6×.

### 1.3 Figures

I ran `f1_data` to `f4_data` from `paper/figures/make_figures.py` with `PYTHONDONTWRITEBYTECODE=1`. All internal
checks pass and the plotted values equal the CSVs. The caption problems:

- **Fig. 1.** The Qwen2.5-Instruct pipelines end in online GRPO, yet they are coloured "SFT, DPO or instruct",
  while OLMo-2-Instruct is coloured "last step pure RL". The colour rule is applied inconsistently.
- **Fig. 2.** The caption calls the 30% and 70% lines "pre-registered". That cannot be verified (see 2.7).
- **Fig. 4a.** The plot shows the mathcot jump at pass 3.0, which contradicts the text (item 19).

---

## 2. Statistical validity

### 2.1 n = 12, two leverage points, nested checkpoints

- **The checkpoints are not independent.** OLMo-2 SFT → DPO → RLVR is one chain, and R1-Distill descends from
  Qwen2.5-Math. The effective sample is closer to six families than to 12 checkpoints.
- **A family-cluster bootstrap of Gate 1 gives r in [−0.64, 0.99].**
- **The family-demeaned correlation is 0.97,** and it comes entirely from the Qwen2.5-1.5B family (Math and R1).
- **Under exchangeability the association is not nothing.** A one-sided permutation test gives p = 0.001 for r and
  p ≈ 0.01 for ρ (20,000 permutations). The honest summary is "one lineage carries it".
- **The percentile bootstrap at n = 12 is poorly calibrated.** The sentence "The width of the interval is honest"
  should go, and the analysis should be at family level.

### 2.2 The leverage point depends on the document-start convention

The repository's own `\n\n` ablation (`exp01_posttraining_bpc/results/ablation_docstart.csv`) changes the
leverage point a lot:

- Qwen2.5-Math against Qwen2.5-1.5B falls from **+23.84% to +16.03%**, essentially Qwen3's value (16.2%).
- R1 against Math rises from **+15.81% to +20.90%**.
- Qwen2.5-Math puts 10.1% of its extra bits in the first 16 tokens of each window. OLMo-2 and the exp06 runs put
  −2.1% to −0.2% there, so they show no window-start artefact. That check passes.

Two claims are affected. Section 4.1's split, "continued pretraining … 23.84%, and the distillation … added
15.81%", **reverses** under the other convention. So does exp03's "more than half of R1's distance comes from the
math continued pretraining". Gate 1 survives: r = 0.835 with `\n\n` where it is available. The paper reports
only the signs under the alternative convention.

### 2.3 Window bootstrap against document bootstrap

A document-cluster bootstrap barely changes the intervals:

| Comparison | Window bootstrap | Document bootstrap |
|---|---|---|
| OLMo-2 SFT, Gutenberg (4 books) | [6.52, 7.87] | [6.79, 7.61] |
| Qwen2.5-Math, Wikipedia | [30.53, 38.16] | [29.94, 38.72] |

So the within-corpus intervals are fine. The larger uncertainty is the choice of corpus, which the fresh mean
treats as fixed. Qwen2.5-Math drifts +5.7 / +33.9 / +31.9% on the three corpora. The intervals in Table 3 are
conditional on these three 100 KB corpora, and the text should say so.

### 2.4 The contrast tests in exp06 have no stated uncertainty, and H4 turns on noise

I computed window-bootstrap 95% intervals for K−S at the marks from `train*/<recipe>/tokens` and exp05's olmo2 classes.

| Mark | tulu | smoltalk | mathcot |
|---|---|---|---|
| 6% | 3.37 [2.83, 3.90] | 2.38 [1.48, 3.28] | 3.41 [2.78, 4.10] |
| 3% | 2.47 [2.01, 2.93] | 1.55 [1.01, 2.09] | 2.86 [2.41, 3.33] |

Paired differences (same windows):

| Mark | mathcot − tulu | mathcot − smoltalk |
|---|---|---|
| 6% | 0.04 [−0.59, 0.72] | 1.04 [0.13, 1.97] |
| 3% | 0.39 [−0.11, 0.91] | 1.31 [0.78, 1.85] |

**The contrast does separate smoltalk from the other two. It cannot resolve mathcot from tulu.**

- **Update-to-update jitter is large.** The SD of the 20-update change in K−S is 0.20–0.28 pp.
- **mathcot's 6% value is a local minimum.** The checkpointed 3.41 is the lowest value in its ±60-update window
  (4.07, 3.71, 3.85, **3.41**, 3.96, 3.66, 3.55; median 3.71).
- **Smoothing flips H4.** With a 7-row median the H4 contrast order holds: mathcot 3.71 > tulu 3.37 > smoltalk 2.43.
- **So the fail rule's trigger is fragile.** The pre-registered "fail rule triggers" needs H1 < 0.6 *and* a violated
  H4 order, so it rests on a 0.04 pp margin. H1 = 0.43 fails either way, but with n = 6 in three recipe clusters,
  CI [−0.79, 1.00].
- **The honest verdict is "not supported, and the design could not have supported it".** "The lead fails" is
  stronger than the data.

### 2.5 H1 and H2 could not both pass

- **The outcomes are almost perfectly anti-correlated.** In exp06, r(forgetting, IFEval gain) = **−0.93** over all
  12 checkpoints and over the 6 marks. A *perfect* forgetting meter would have r ≈ −0.93 with the gain and fail H2.
- **The registered expectations already implied this.** They predicted forgetting ordered mathcot > smoltalk > tulu
  and IFEval gain ordered tulu > smoltalk > mathcot. Together that implies the anti-correlation, so expectation E1
  ("H1 at least soft *and* H2 pass") contradicted the expected values registered beside it.
- **The paper presents the H2 failure as evidence against the contrast.** It is mostly a property of the recipe set.

### 2.6 Multiple comparisons, and post-hoc readings presented as headlines

- **Many tests, no correction.** `exp04/results/correlations.csv`, `exp05/.../correlations.csv` and
  `exp06/.../correlations.csv` hold well over 150 correlations.
- **Selected readings carry the narrative.** They are the IFEval-chat r (the same gain over all 12 is r = −0.03),
  the 15% threshold, the exp05 contrast lead and the Wikipedia profile.
- **Both abstract claims about benchmarks are post hoc.**
  - "It tracks the instruction-following gain" uses the chat-template addendum. It was committed in c295d52
    (00:04:44) **two minutes after** the main result in 669d2bc (00:02:57).
  - "Above 15% it was always accompanied by forgetting" was, by exp04 RESULTS.md's own wording, "added after an
    interim look".
  - The appendix discloses both. The abstract states them as findings.

### 2.7 Pre-registration audit (git)

| Experiment | Pre-registration commit | Run or result | Text unchanged? |
|---|---|---|---|
| exp01 | none | 2d9c318 (with results) | n/a: no rule exists |
| exp02 | **none before the run**. The rule first appears in `run_all.sh` in 762d4da, the results commit (2026-10-14 20:05). | same commit | unverifiable |
| exp03 | 9562f71, 20:21 | job started 20:23:21 (`slurm-1080281.out`); c94d9a5 at 20:46 | identical (diffed) |
| exp04 | abc08d9 21:01, f33fa18 21:02, 9f22ff0 22:00 | array 22:20–23:58; 669d2bc 00:02 | identical to 9f22ff0 (diffed) |
| exp05 | 1eb436a 14:46 (with the analysis code and class files) | e689078 15:01 | identical (diffed) |
| exp06 | 9cca20f 15:24; deviations becc72a 16:36, 0454f34 20:39, ec2e6f6 08:22 | c0944b1 08:28 | identical, including the deviation blocks (diffed) |

- **No pass/fail threshold was edited after the fact.** For exp03–exp06 the registration is real and unchanged.
- **The introduction's sentence is false for two experiments.** "Each experiment was pre-registered: the rules and
  expected values were committed before the runs" fails for exp01 (no rules) and exp02 (no commit predates the
  results). The appendix table lists exp02's rule as committed before the run.
- **exp02's outcome was foreseeable.** `docs/research-thread.md` (c140ced, 19:31) already recorded Yang's "at most
  36%" result before exp02 ran.
- **The softening is in the narrative, not in the rules.** The pre-registered gsm8k gain (r = +0.38) is replaced in
  the abstract by the post-hoc IFEval-chat gain.
- **One omission runs the other way.** exp06's H2 over all 12 passes (r = −0.20) and is not reported.

### 2.8 The 15% threshold is a fit

- **Any cut in the gap passes.** Every cut between 14.66% (OLMo-2 RLVR, no forgetting) and 16.18% (Qwen3,
  forgetting 4.7%) gives "3 of 3 above". The threshold has no margin.
- **Sensitivity is poor across all 24 checkpoints.** Eight forget beyond the floor: Math, R1, Qwen3, SmolLM2 and
  mathcot ×4. A 15% cut flags 3 of the 8. The discussion's 20% cut flags 2, both from one lineage.
- **The abstract and summary use 15%; Discussion rule 3 uses 20%.**

### 2.9 Noise floor

The formula is computed correctly; I reproduced every value. It has four problems:

1. **It treats base and checkpoint as independent samples, though they are scored on the same items.** That makes
   it conservative for "forgot" calls and *liberal* for "forgot nothing" calls. Rule 2 ("SFT and DPO were safe")
   rests on the second kind. Paired tests are impossible because `log_samples` was off, so per-item outcomes were
   never saved.
2. **It covers item sampling only, not seed-to-seed variance,** which at 1B is probably of the same order.
3. **H3 compares a range of three checkpoints with a two-checkpoint floor.** The 95th percentile of the range of
   three iid normals is about 1.19 times that floor, so the right bar is about 2.6 pp. H3 still passes.
4. **The subsets are first-N, not random** (HellaSwag's first 2,000 validation documents).

---

## 3. Confounds and design

1. **The "fresh" corpora are mostly not fresh writing.**
   - Only arXiv is unambiguously post-cutoff.
   - wiki2026 has 6 event pages from 2026, about 25 KB of 102 KB. The other 11 are new pages about old subjects:
     a 1971 UN resolution, a historical monument, a 1980s video game and others, some probably translations.
   - gutenberg2026 is 4 books written between the 1900s and the 1920s (`gutenberg2026.prov.json`), so its windows
     are clustered in 4 documents.
   - The abstract's "text written after its training cutoff" is false for about two-thirds of the bytes.
   - The paper's own enwik8 result (r = 0.95) shows that freshness does not matter for the forgetting outcome.
     The title and abstract emphasis on "fresh" is unjustified.
2. **Document start and tokenizer.** See 2.2. R1-Distill is scored after its BOS token and its parent Qwen2.5-Math
   after `<|endoftext|>`, so the R1-vs-Math step mixes two conventions.
3. **The log-likelihood benchmarks are run without chat templates.** exp04's own pre-registration says this "can
   look like forgetting on log-likelihood tasks". The paper's Limitations mention prompt format only for the gain.
4. **"Forgetting" in the decisive cases is a single benchmark family.** This is the most important design finding
   of this review.
   - **mathcot loses ARC only.** Without ARC-e and ARC-c, its forgetting is +0.37, −0.04, +1.45 and +1.50%.
     HellaSwag, PIQA and WinoGrande stay within ±1.5 points, and MMLU moves −0.8 to −1.9 points.
   - **SmolLM2-Instruct loses ARC only:** −0.03% without ARC.
   - **So two of the three non-math-line forgetting cases are entirely ARC.** That includes the case on which the
     whole recipe conclusion rests.
   - **Length normalisation amplifies the ARC drop.** For mathcot at update 1500, ARC-e acc falls 0.726 → 0.662
     (−6.4 points) while acc_norm falls 0.735 → 0.641 (−9.4). For SmolLM2-Instruct, acc falls −8.6 and acc_norm
     −10.6.
   - **No control separates format from knowledge.** A model trained only on chat-wrapped `<think>` answers and
     scored on raw "Question: … Answer:" cloze could lose ARC through answer-format change. Nothing tests this:
     no MC-letter ARC, no chat-template log-likelihood, no few-shot ARC, no brief re-alignment test in the style of
     the spurious-forgetting work.
5. **"Forgot nothing" is an average of opposite-signed effects.**
   - **MMLU rises while ARC-e falls.** tulu and smoltalk raise 0-shot MMLU, a letter-choice format, by +3.6 to +5.7
     points; OLMo-2-SFT by +5.9. Meanwhile ARC-e falls by 0.5 to 2.8 points.
   - **Without MMLU, the "no forgetting" recipes forget a little:** tulu forgets up to +1.4%, smoltalk up to +0.8%
     and OLMo-2-SFT +1.9%.
   - **The general score mixes the two formats.** It combines cloze tasks that SFT hurts with a multiple-choice
     format that SFT teaches.
6. **The 256-token generation cap was known to be broken and was kept.** exp04 had already shown it truncates R1
   (15.7%), yet exp06 used it for an R1-style recipe. The forgetting recipe's target gain is therefore unmeasured by
   design. H2 tests the IFEval gain of a recipe never trained for IFEval.
7. **"Recipe" bundles several variables.**
   - Loss-token share is 0.94 for mathcot against 0.70 and 0.74. By update 1500 mathcot has had **69.4M**
     supervised tokens against 54.5M (tulu) and 55.3M (smoltalk), 27% more.
   - Distinct examples are 30.5k against 79k and 60k.
   - mathcot is the 24% of scanned OpenR1 answers that fit in 2,048 tokens, a length and difficulty selection.
   - Content, supervision density, example count and selection are all confounded, so "the recipe alone decides"
     cannot be separated from them.
8. **Optimizer state differs between the compared checkpoints.**
   - Stage 2 restarts with a fresh AdamW and re-warms the learning rate to 2e-5.
   - The marks sit at a learning rate of 1–2e-5 (unannealed). The stage-1 finals sit at learning rate 0.
   - The stage-2 finals are also unannealed: the wall-time stop came at about 1,970 of 3,000 scheduled updates.
9. **The marks were selected on the same noisy measurement.** A checkpoint is saved at the first crossing, which
   selects upward excursions; 4 of the 6 marks follow pass boundaries. "Matched dose" largely means "matched
   repetition event".
10. **Packing fell short of plan, unreported.** 1,500 updates gave 1.63–1.72 passes, not the pre-registered about
    2.2, because 21–25% of positions are padding. This is not listed as a deviation.
11. **One base, one seed, 1B parameters.** OLMo-2-0425-1B was mid-trained on Dolmino, which contains instruction and
    math data, so none of the recipes acts on a "pure" pretrained model.
12. **Only three RL checkpoints, with unmatched compute.** "Pure RL moved it by 0.1–0.5%" compares checkpoints, not
    units of training. The SimpleRL-Zoo and RLVR runs are far smaller than the SFT and DPO stages, so "ordered by
    the kind of training" confounds method with dose.
13. **Qwen2.5-Math is continued pretraining, a new base.** The two leverage points are a continued-pretraining
    lineage, and domain continued pretraining is already known to forget (Harmon et al., whom the paper cites, say
    so). Calling them "post-trained" inflates the post-training claim.

---

## 4. Claims against evidence

### 4.1 Stronger than the data support

**Abstract**

1. **"text written after its training cutoff".** False for Gutenberg and for most of Wikipedia (§3.1).
2. **"Every post-training stage lengthened the code".** This is 15 comparisons on 12 checkpoints, with no run-to-run
   variance, stated as a law.
3. **"One temperature removes at most a third".** It removes 35.3%, on 4 pairs only, and never on the OLMo-2 chain.
4. **"The increase measures how much post-training was applied".** In exp06 the drift also jumps 0.27–1.88 pp at
   pass boundaries with no benchmark change, so it measures repetition and memorisation of the SFT set too.
5. **"It tracks the instruction-following gain".** This holds only across 8 public chat checkpoints, on a post-hoc
   measure. **Under recipe control it does not hold:**
   - r(Δ, IFEval gain) = 0.13 over all 12 checkpoints and −0.04 over the 6 marks;
   - tulu's gain falls 15.5 → 12.0 while its drift rises from 3.3% to 7.1%.
6. **"Above 15% it was always accompanied by forgetting".** The threshold is post hoc, with no margin, and rests on 3
   checkpoints, two of them nested (§2.8).
7. **"The recipe alone decides whether a model forgets".** Three recipes, one seed and one base; the "forgetting" is
   ARC-only; and the recipe bundle is confounded (§3.4, §3.7).

**Introduction**

8. **"The rest is a change in what the model predicts, not in how confident it is".** One scalar temperature cannot
   establish this; the body admits as much. Yang, whom the paper cites, also shows that a calibrator recovers 29%
   even from random weight corruption, so the recovered share is not diagnostic.
9. **"The drift is one-dimensional across token classes".** r = 0.925 depends on the math line: 0.76 without it,
   0.71 over the chat-tuned checkpoints. exp05's E1 predicted the trigger *because of* that leverage, so the rule
   was uninformative by construction.
10. **"A step … set by the recipe at the first checkpoint and constant thereafter".**
    - The first mathcot checkpoint evaluated is update 1500: 1.63 passes, at the end of a full cosine schedule.
      Nothing is known about any earlier point.
    - "Constant" means "within a ±2.2 pp floor".
    - STATE.md's "early step" is unsupported.
11. **"Each experiment was pre-registered".** False for exp01 and exp02 (§2.7).
12. **"19 s … 67 times as much".** Mixed hardware, and the wrong comparator (§1.2).

**Discussion: the three practical rules**

13. **Rule 1, "Evaluate the first checkpoint of a new recipe".** Not actionable.
    - "First checkpoint" is undefined. Here it means 1.6 epochs.
    - No checkpoint before update 1500 was evaluated for mathcot, so the data cannot say how early to look.
    - Practitioners already evaluate checkpoints.
14. **Rule 2, "Conversational SFT and DPO were safe to large doses".**
    - The evidence is two SFT datasets on one base plus the OLMo-2 chain.
    - It is contradicted inside the paper by SmolLM2-Instruct (3.8%).
    - It rests on MMLU format gains offsetting ARC losses (§3.5), and on an unpaired floor that favours "no
      difference" (§2.9).
15. **Rule 3, "Treat a drift above 20% as an alarm".**
    - Two checkpoints from one continued-pretraining lineage.
    - Inconsistent with the 15% used in the abstract and summary.
    - Nobody needs a meter to learn that domain continued pretraining forgets.
    - No practitioner could set a threshold from this.
16. **"Our measurements support the exclusion [of Huang et al.] … after post-training the code length … measures a
    distance … and not a capability".** This is untested, and the data partly contradict it (§4.2, item 1).

### 4.2 Weaker than the data support (undersold)

1. **The absolute compression-capability relation survives post-training for log-likelihood benchmarks.** Using
   the committed files, absolute fresh-text bpb against the general score gives:
   - r = −0.97 (ρ = −0.94) over the 6 bases;
   - r = **−0.81** (ρ = −0.71) over the 12 post-trained checkpoints;
   - r = −0.65 over those 12 without the math line.

   This is the test Huang et al.'s exclusion actually concerns, and the paper never runs it. It is an interesting
   partial *survival* of the compression-intelligence relation, with the obvious caveat that both sides are
   likelihood-based.
2. **The across-family association is statistically non-trivial** (permutation p = 0.001 for r), even though one
   lineage carries it.
3. **The pipeline is robust, and the paper undersells it.**
   - The 16 Mac/T4 rows agree to 5e-5 bpb.
   - The document bootstrap equals the window bootstrap.
   - OLMo-2 and the exp06 runs show no window-start artefact.
   - Tables 3–6 recompute exactly from raw arrays.
4. **The repetition finding is under-developed.** Drift jumps within 20 updates of every pass boundary, together
   with drops in training loss, while benchmarks stay flat. That suggests a cheap detector of SFT-set
   memorisation, which may be the most useful property of the meter. No checkpoint was evaluated on either side
   of a jump to confirm it.
5. **Saturation.** In exp06 both forgetting and the IFEval gain are flat from the first evaluated checkpoint
   (tulu IFEval 14.0 → 12.0, mathcot forgetting 4.24 → 4.41), while the drift triples. That is direct evidence that
   most of the drift is orthogonal to both outcomes at this scale. It is the cleanest form of the paper's main
   negative result.
6. **The OLMo-2 DPO step is a clean counterexample to "distance from base on general text = forgetting".** It
   doubles the drift (+6.65%), forgets nothing on the cloze-only score (0.00%) and gains 12 IFEval points.

---

## 5. Prior work

**Verified against arXiv or alphaXiv on 2026-10-16.** All seven are attributed correctly in substance.

- **Huang et al. 2404.09937.** 31 LLMs, 12 benchmarks, r = −0.93 overall and about −0.95 within domains; the
  exclusion rationale ("Fine-tuned models are no longer general-purpose compressors…") is reported correctly.
- **Shenfeld et al. 2509.04259.** KL to the base "evaluated on the new task". Correct.
- **Harmon et al. 2510.17776.** "low-to-moderate forgetting" after RL or SFT from base models. Correct. The same
  abstract reports that domain-continual pretraining causes moderate forgetting, which is the right citation for
  the Qwen2.5-Math point and is not used.
- **Yuan et al. 2511.08066.** Section 4.3.1 and Fig. 4 say post-training degrades compression and attribute
  Qwen3's larger degradation to multi-stage RL. The paraphrase is fair. The paper should state the stronger point
  its own Table 2 implies: Qwen3-1.7B was *distilled*, not RL-trained.
- **Marek et al. 2605.26097.** KL or NTP replay on self-generated data. Correct.
- **Kalajdzievski 2401.05605.** Cross-entropy to the base's predictions on WikiText-103. Correct. It is
  essentially the paper's quantity, so the abstract's "A method is described" overstates novelty.
- **Yang (alphaXiv).** The page exists and gives "up to 36%". It also reports 1.2–1.3% recovery for math models,
  against this paper's 19% for R1 against Math. The difference is not discussed.

**Missing work a reviewer in this area would expect.** Several of these are already in the repository's own
`docs/prior-art.md`, which even warns that "Reviewers will raise the perplexity objection".

- **Spurious or format forgetting.** Zheng et al., "Spurious Forgetting in Continual Learning of Language Models"
  (ICLR 2025, 2501.13453). Kotha et al. 2023 (implicit-inference account; conjugate prompting). Palit et al.
  (listed in `docs/prior-art.md`). These are central to §3.4.
- **Multiple-choice format sensitivity.** OLMES (Gu et al. 2024) on cloze against MCF and the emergence of MMLU
  MCF; work on first-token versus text answers in instruction-tuned models. These are central to §3.5.
- **Alignment tax.** Askell et al. 2021, Bai et al. 2022, Lin et al. "Mitigating the Alignment Tax of RLHF"
  (2309.06256). InstructGPT is cited only for its learning-rate choice.
- **Catastrophic forgetting in LLMs and continued pretraining.** Luo et al. 2023 (2308.08747), Biderman et al.,
  "LoRA Learns Less and Forgets Less" (2405.09673, math continued pretraining and instruction tuning), Ibrahim et
  al. 2024.
- **Token-level shift after alignment.** URIAL (Lin et al., ICLR 2024, 2312.01552) finds base→aligned shifts
  concentrated on stylistic tokens. exp05's "knowledge tokens pay first in 9 of 12" needs to be set against it.
- **DPO likelihood displacement** (Razin et al. 2024; Pal et al. 2024). This bears directly on the OLMo-2 DPO step.
- **KL as the dose of RL** (Gao et al. 2023, reward over-optimisation).
- **Calibration after RLHF** (Kadavath et al. 2022; the GPT-4 report), for exp02.
- **Fresh-text compression and contamination.** Uncheatable Eval (listed in `docs/prior-art.md`, base models
  only by design), and contamination of enwik8 and other memorised benchmarks (e.g. Magar & Schwartz 2022).
- **Multi-epoch fine-tuning and the loss staircase at epoch boundaries**, for §4.6.
- **Decorative citations.** Morris et al. (3.6 bits per parameter) is cited for the epoch jumps but supports
  nothing measured here. Kostelec & Guo (hybrid-model distillation) is tangential.

---

## 6. Reproducibility

**What works**

- Every experiment has a `run_all.sh` with its steps in the header.
- The corpora are committed with provenance JSON (URL and date per document).
- 289 per-token `.npy` arrays and the lm-eval summaries are committed.
- The SFT subsets are regenerable: dataset revisions are pinned and the seed is fixed (`logs/setup/data/*.prov.json`).
- The test suite passes and the figure script reproduces the plotted numbers.

**What does not**

1. **The repository URL in Appendix B returns HTTP 404.** `https://github.com/gottn/compression-intelligence` and
   the GitHub API both 404, so the repository is private or absent. An outside reader cannot rerun anything.
2. **The 18 Hugging Face model revisions are not pinned** (`cilab/stages.py`; the scorers load `main`).
3. **No per-sample lm-eval outputs** (`log_samples: false`). Truncation, paired tests and item-level forgetting in
   the style of Harmon et al. cannot be checked.
4. **The exp06 checkpoints are not released.**
5. **exp03–exp06 need the AAU Slurm cluster.** They depend on `infra/slurm/*.sbatch`, the NGC 25.09 container and
   `ssh aau`, and the logs carry `/home/id.aau.dk/...` run paths. A reader needs to port them.
6. **exp01 and exp02 ran on MPS.** exp03 shows the T4 agrees to 5e-5 bpb, which is fine.
7. **exp01's model-card capability numbers are partly extracted by a sub-agent and unverified.** The paper does not
   use them, which is correct.

---

## 7. Writing

- **Length.** 15 pages and about 8,000 words, twice a typical workshop limit.
  - exp01 and exp02 fit in one paragraph and one table.
  - The token-class section could move to an appendix.
  - The Definitions section spends a displayed equation and the phrase "It is easily seen" on the fact that entropy
    cancels in a difference.
- **Register.** "Code length", "lengthened the code", "The answers are the following" and "something of the order
  of" sound like Shannon. They turn hedged empirical statements into laws, and the caveats disappear behind them:
  - "Every post-training stage lengthened the code";
  - "The increase measures how much post-training was applied";
  - "What the run shows is that the recipe sets forgetting as a step";
  - "The repeated examples are being memorised", an interpretation stated as fact;
  - "The width of the interval is honest".

  ML readers will look for "NLL", "cross-entropy" and "perplexity". Use them, or at least gloss them.
- **Internal inconsistencies.**
  - 15% against 20%.
  - "At most a third" against 35%.
  - Section 4.6's account of which marks followed pass boundaries contradicts Fig. 4a.
  - "Pre-registered" is used for exp02.
- **Agent-workflow leakage.** Appendix A says "not the 11 of the brief". The RESULTS files speak of "the
  coordinator". Remove these.
- **Title and abstract.** The title stresses "fresh", which the results show to be irrelevant. The abstract opens by
  claiming a method that Kalajdzievski (2024) already used.
- **Strength.** Appendix A's table of rules against outcomes is the best part of the paper and should be kept.

---

## (a) Summary

The engineering is unusually careful. Every number I recomputed from raw arrays matches, the pre-registrations of
exp03–exp06 are genuine and unedited, and the paper honestly reports that its central hypothesis, a one-sided
forgetting meter, failed. The scientific claims in the abstract and discussion go further than the evidence.

- **The headline benchmark findings are post hoc.** The IFEval tracking and the 15% threshold were added after the
  data were seen. The IFEval tracking disappears under the paper's own recipe control. The threshold has no margin
  and rests on one continued-pretraining lineage, whose drift moves by 8 pp with the document-start token.
- **The controlled experiment cannot carry its conclusion as stated.**
  - The forgetting it attributes to "the recipe alone" is entirely an ARC drop: 0.0–1.5% forgetting without ARC.
  - The "no forgetting" of the other recipes is MMLU format gains cancelling ARC losses.
  - The contrast test compares values 0.04 pp apart, inside a ±0.6 pp interval, so its fail-rule trigger rests on
    that margin.
  - H1 and H2 could not both pass, given outcome correlations the authors themselves predicted.
- **What the paper can claim.** A clean negative result: base-relative held-out NLL grows with SFT, DPO and
  repetition, without tracking forgetting or the gain at ordinary doses. Add the undersold finding that absolute bpb
  still ranks post-trained models on log-likelihood benchmarks.

## (b) Major issues (must fix)

1. **Rewrite the abstract and introduction to match the evidence.**
   - Drop "written after its training cutoff", "tracks the instruction-following gain" (it fails under control),
     "always" and "alone".
   - Label the 15% threshold and the IFEval-chat measure as post hoc in the abstract.
   - Pick one threshold.
2. **Correct the pre-registration claim.** exp01 had no rules, and exp02's rule is not verifiable in git.
3. **Control format in the forgetting measure.** Report forgetting without ARC and without MMLU next to the
   six-task mean. Add a format control for mathcot and SmolLM2's ARC loss: MC-letter ARC, ARC with the chat
   template, few-shot ARC, or a brief re-alignment test. Without it, "forgetting" in exp06 may be a format effect.
4. **Fix the statistics.**
   - Family-level or lineage-level analysis (or a permutation test) for exp04.
   - Window-bootstrap intervals, paired across recipes, on every contrast in exp06.
   - Report the smoothing sensitivity of H4 and the fail rule.
   - Report the H1/H2 incompatibility (r(forgetting, gain) = −0.93).
   - State the document-start sensitivity of Qwen2.5-Math (+23.8 → +16.0) and the reversal of the
     CPT-versus-distillation split.
5. **Re-measure the gain with a long generation cap** (≥ 2,048 tokens) for mathcot and R1, or withdraw every gain
   statement about them, including the reading of H2.
6. **Drop or rebuild the "practical rules".** They need an operational definition of "first checkpoint", evidence
   from early checkpoints, and more than two points behind any threshold. As written, a practitioner cannot act on
   them.
7. **Correct the cost claim.** Use one hardware type and compare against the log-likelihood group only, about 11–35×.
8. **Make the repository public, or fix the URL,** and pin the model revisions.
9. **Add the missing prior work** on spurious and format forgetting, alignment tax, URIAL, and MC-format
   sensitivity, and position the measure as Kalajdzievski's quantity evaluated, not a new method.

## (c) Minor issues

- RLVR is a second exception to the enwik8 ratio of 1.3–3.0 (0.72).
- "mathcot drifts smoothly" is false, and 4 of the 6 marks follow pass boundaries, not only tulu's.
- "Within each recipe … r = 0.80" is a pooled r.
- The Fig. 1 colour rule is inconsistent: Qwen2.5-Instruct also ends in GRPO.
- The H3 floor should be a range-of-three floor (about 2.6 pp). H3 still passes.
- Report the unannealed learning rate at the marks and at the stage-2 finals, and the shortfall in packing
  efficiency (1.6–1.7 passes against the planned 2.2).
- Report the 27% higher supervised-token count of mathcot.
- The noise floor is unpaired; say that it favours "no difference" verdicts.
- Report exp06's H2 over all 12 (−0.20, a pass) alongside H1 over all 12.
- exp05's lead: report that it falls to 0.55 without Qwen3, and that its scale-free form (K−S divided by total
  drift) gives r = 0.24 over 12 and 0.45 without the math line.
- Discuss the difference from Yang's 1.2–1.3% recovery for math models.
- Remove "the brief" from Appendix A.
- Rename "drift", which collides with distribution-shift usage, or define it once in ML terms.
- Cut the paper to workshop length.

## (d) Scores

- **Soundness: 5/10.** The measurements are exact and the pre-registration is real for four experiments, but the
  conclusions rest on post-hoc readings, one lineage of leverage points, an ARC-only forgetting signal and a contrast
  test decided by sub-noise margins.
- **Contribution: 4/10.** A careful negative result: base-relative held-out NLL, already used by Kalajdzievski, does
  not meter forgetting at ordinary doses. It comes at small scale (≤ 3B, one controlled base, one seed), and its
  most interesting findings (absolute bpb still ranking post-trained models, drift as an SFT-memorisation detector)
  are undeveloped.
- **Clarity: 6/10.** The tables and pre-registration appendix are exemplary, but the paper is twice workshop length,
  its abstract contradicts its own body in three places, and the formal register hides the caveats.

## (e) Recommendation

- **As a workshop paper (COLM / ICLR evaluation workshop): major revision.** In its current form it is a weak reject
  at a selective venue, because the abstract's claims exceed the evidence. Reframed as a pre-registered negative
  result, with the format control and family-level statistics added, it would be a reasonable accept.
- **As a portfolio piece for a post-training team: minor revision before sending.** The pipeline, the
  pre-registration discipline, the reproducibility and the willingness to report failure are well above typical
  portfolio work. A post-training reviewer will quickly find the ARC-only forgetting, the MMLU format offset, the
  256-token cap kept after it was known to fail, and the thin "practical rules", so fix those first: major issues 1,
  3, 6 and 8 at minimum. Be ready to explain which design decisions were the author's, since the acknowledgement
  says agents implemented and ran the experiments.

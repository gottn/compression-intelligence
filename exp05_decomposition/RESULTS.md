# exp05: does post-training drift sit on knowledge tokens or on style tokens?

**Status: pre-registered, no results yet.** Everything below was written on 2026-10-15, before any per-class bits were summed and before any correlation was computed. The outcome sections are added above this text after the run; the definitions and rules are not edited.

## Question

exp04 found that fresh-text Δbpb tracks both forgetting (ρ = 0.67, n = 12) and the IFEval chat gain (r = 0.89, n = 8). One number cannot be a forgetting meter if it also measures what was gained. Does the drift split into two channels?
- A **knowledge** channel: bits on names, numbers and rare content words, which should rise when a recipe overwrites what the model knows (math continued pretraining, R1 distillation, SmolLM2's SmolTalk SFT).
- A **behaviour** channel: bits on function words, punctuation and whitespace, which should rise when a recipe changes how the model writes (chat SFT, DPO, RLVR).

The test: knowledge drift should predict forgetting, and style drift should predict the instruction-following gain.

## Data (no new scoring; Mac only)

- **Per-token bits, RAW condition** (−log2 p of every text token, input = [doc-start, *window], 1024-token windows):
  - exp03 `results/tokens/`: 14 models × 4 corpora (AAU T4, fp16).
  - exp01 `results/tokens/`, `raw` rows only: SmolLM2-1.7B, SmolLM2-1.7B-Instruct, Qwen3-1.7B-Base, Qwen3-1.7B (Mac MPS, fp16).
  - 18 models × 4 corpora = 72 (model, corpus) arrays.
- **Corpora:** exp01's dated snapshots, whole files as scored: wiki2026, arxiv2026, gutenberg2026 (fresh) and enwik8_ctrl.
- **Outcomes:** exp04 `results/scores.csv`, definitions unchanged.
- **Checkpoints:** the 12 post-trained rows of `cilab/stages.py`, each against its family base. Qwen2.5-Math-1.5B and R1-Distill are compared with plain Qwen2.5-1.5B, as in exp04.

## Alignment (fixed now)

- **Token ids and character offsets.** The fast tokenizer is called on exactly the scored text: `tok(text, add_special_tokens=False, return_offsets_mapping=True)`.
- **The run raises unless:**
  - the ids equal `cilab.text.encode(tok, text)`;
  - the number of ids equals the length of the saved bits array, for every one of the 72 (model, corpus) pairs;
  - the windows from `cilab.text.make_windows(tok, ids, text, 1024)` cover those ids, and for exp03 models their lengths equal exp03's `windows.csv`;
  - for exp03 models, the ids' SHA-1 equals `ids_sha1` in exp03's `bpb.csv`, and for exp01 models, the token count equals exp01's `bpb.csv`;
  - each model's own tokenizer gives the same ids as the first model of its tokenizer group (`cilab.stages.tokenizer_groups`: olmo2, qwen2.5, qwen3, smollm2);
  - the offsets tile the text: the first span starts at 0, starts never decrease, no gap between a token and the next, the last span ends at the end of the text;
  - every token whose own decode is a whole string (no U+FFFD) decodes to exactly the text under its offsets.
- Classes are computed once per tokenizer group per corpus.
- **POS tags:** spaCy 3.8 `en_core_web_sm` 3.8.0, tagger and attribute ruler only (parser, NER and lemmatizer disabled), run once per corpus on the whole text.
- **Token → word:** each token takes the Universal POS tag (`pos_`) and the lowercased form (`lower_`) of the spaCy token covering the token's **first non-whitespace character**. A token made only of whitespace has no word.

## Token classes (primary definition)

| class | members |
|---|---|
| **KNOWLEDGE** | PROPN, NUM; NOUN, ADJ, VERB whose lowercased word is **not** among the 2,000 most frequent English words |
| **STYLE** | DET, ADP, PRON, CCONJ, SCONJ, AUX, PART, INTJ, PUNCT, SPACE, SYM, and whitespace-only tokens |
| **MIDDLE** | everything else: NOUN, ADJ, VERB among the 2,000 most frequent words; ADV; X |

- **Frequency list:** wordfreq 3.1.1, `top_n_list("en", 2000)` (Robyn Speer, 2022, rspeer/wordfreq v3.0, Zenodo, doi:10.5281/zenodo.7199437). Surface forms, not lemmas.
- All three classes are reported. The hypotheses are about KNOWLEDGE against STYLE.

**Robustness variant (secondary).** A frequency-only split per corpus and tokenizer group:
- each token occurrence gets f = the count of its token id in that corpus;
- **RARE** if f is below the median of f over all token occurrences of the corpus, else **FREQUENT**.
- RARE stands in for KNOWLEDGE and FREQUENT for STYLE.

## Quantities

All per post-trained checkpoint m, its family base b, corpus k and class c, from the RAW per-token bits:
- **drift_c** = 100 (Σ_c bits_m − Σ_c bits_b) / Σ_c bits_b, in %. This is the relative change of the class's code length.
- **share_c** = 100 (Σ_c bits_m − Σ_c bits_b) / (Σ_all bits_m − Σ_all bits_b), in % of the total Δbits. The three classes sum to 100. This shows where the drift sits.
- **Proportional null:** base_frac_c = 100 Σ_c bits_b / Σ_all bits_b. If post-training scaled every token's bits by one factor, share_c would equal base_frac_c. It is reported next to share_c.
- **Fresh value:** the unweighted mean over wiki2026, arxiv2026 and gutenberg2026, the same convention as Δbpb_fresh in exp03 and exp04.
- **CIs:**
  - paired window bootstrap, 5,000 resamples, seed 0 + corpus index (exp03's seeds, so the same resamples as exp03);
  - the fresh mean resamples windows within each corpus (`cilab.stats.stratified_mean_ci`);
  - the windows are the 1024-token `make_windows` windows.
- **Memorisation axis:** excess = Δbpb_enwik8 − Δbpb_fresh, in percentage points, over all tokens.
- **Reproduction check:** drift over all tokens must reproduce exp04's `dbpb_fresh` and `dbpb_enwik8` within 1e-3 percentage points. The run raises otherwise.

**Outcomes** (exp04 `scores.csv`):
- **forgetting** = 100 (general_base − general_post) / general_base, in % of the base;
- **IFEval chat gain** = `gain_ifeval_chat`, in points (the post-trained model with its chat template, minus its base, prompt-level strict);
- **MMLU loss** = 100 (mmlu_base − mmlu_post) / mmlu_base, in % of the base (relative, like forgetting). MMLU loss in points is reported as a reading.

**Statistics** follow exp04's conventions:
- Pearson r and Spearman ρ over checkpoints;
- 95% percentile CIs from 10,000 checkpoint-bootstrap resamples (seed 0; resamples with a constant column are dropped);
- leave-one-out range;
- "without the math line" = without Qwen2.5-Math-1.5B and R1-Distill (n = 10);
- "chat-tuned" = stages sft, dpo, rlvr, instruct (n = 8).

## Pre-registered tests

| id | statistic | passes if |
|---|---|---|
| **H1** knowledge channel | Spearman ρ(drift_KNOWLEDGE fresh, forgetting), n = 12; and Pearson r of the same pair without the math line, n = 10 | ρ ≥ ρ(Δbpb_fresh, forgetting) as recomputed here (exp04: 0.671) **and** r ≥ 0.6 without the math line (the total gave 0.32) |
| **H2** behaviour channel | Pearson r over the 8 chat-tuned checkpoints: drift_STYLE fresh vs IFEval chat gain, and drift_KNOWLEDGE fresh vs IFEval chat gain | r_style ≥ 0.8 **and** r_knowledge < 0.5 |
| **H3** case contrast | share_KNOWLEDGE fresh with its 95% CI, SmolLM2-1.7B-Instruct vs OLMo-2-0425-1B-DPO | the lower CI bound for SmolLM2-Instruct lies above the upper CI bound for OLMo-2-DPO |
| **H4** memorisation axis | Spearman ρ over the 12 | ρ(excess, MMLU loss) ≥ ρ(Δbpb_fresh, MMLU loss) |
| **Fail: one-dimensional drift** | Pearson r(drift_KNOWLEDGE fresh, drift_STYLE fresh) over the 12 | r > 0.9 |

**What "fail" means.** If knowledge and style drift correlate above 0.9 across the 12 checkpoints, all channels move together and drift is one-dimensional at this scale. In that case the decomposition adds no channel, whatever H1 and H2 say: a pass on either would be the total drift seen through a subset of tokens. If the fail rule does not trigger:
- H1 alone supports a knowledge channel;
- H2 alone supports a behaviour channel;
- both together support the two-channel reading.

**Secondary readings** (pre-registered here, not gates):
- **Contrast:** drift_KNOWLEDGE − drift_STYLE, in percentage points, against forgetting (n = 12 and n = 10) and against the IFEval chat gain (n = 8). A channel-specific signal survives one-dimensionality only if the contrast predicts.
- **Frequency variant:** H1 and H2 repeated with RARE in place of KNOWLEDGE and FREQUENT in place of STYLE, and the one-dimensional rule on RARE vs FREQUENT.
- **Full table:** drift_c and share_c for every class, checkpoint and corpus, plus the proportional null.
- **Other forms:** MMLU loss in points; Spearman for every test that is stated in Pearson, and the reverse.

**Expected** (stated now, scored after):
- **E1.** The fail rule triggers (r > 0.9), because Qwen2.5-Math and R1-Distill drift far on every class and dominate any Pearson over 12.
- **E2.** drift_KNOWLEDGE > drift_STYLE on the fresh mean for Qwen2.5-Math, R1-Distill and SmolLM2-Instruct, and drift_STYLE ≥ drift_KNOWLEDGE for OLMo-2 DPO.

## Confounds (stated before the run)

- **Tagger errors.** en_core_web_sm is a small model. arXiv abstracts carry LaTeX, Gutenberg has archaic spelling and enwik8 has wiki markup; POS tags there will be noisier than on news text.
- **First-character rule.** A BPE token that spans a word boundary (`",\n\n"`, `").`") takes the class of its first non-space character. Every continuation piece of a long rare word counts as KNOWLEDGE, so rare words weigh by their token count.
- **Tokenizers split differently.** Qwen tokenizers split numbers into single digits, so NUM carries more tokens in the Qwen families than in OLMo-2.
- **Surface forms.** The top-2,000 list is not lemmatised, so inflections of common verbs outside the list (for example "explained") land in KNOWLEDGE.
- **Relative vs absolute.** Style tokens are cheap (few bits each), so a small absolute change is a large relative drift_c. share_c is the absolute view; both are reported.
- **Small and heterogeneous n.** The 12 checkpoints come from different labs, recipes and families. The two math-line points carry leverage in any Pearson r.
- **Outcome noise.** The outcomes are exp04's single-run benchmark scores. 8 of 12 forgetting values are inside their noise floor.

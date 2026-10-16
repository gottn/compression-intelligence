# Derived numbers (written by paper/derived.py; not printed in any RESULTS.md)

## Cost: log-likelihood group (A10) against one drift score

- loglik_median_s: 167
- loglik_min_s: 92.5
- loglik_max_s: 272.7
- drift_t4_median_s: 14.95
- drift_t4_min_s: 10.4
- drift_t4_max_s: 32.9
- ratio_exp04: 11.17
- drift_a40_median_s: 5.2
- loglik6_min_s: 156.8
- loglik6_max_s: 194.7
- ratio_exp06_lo: 30.15
- ratio_exp06_hi: 37.44

## Absolute fresh-text bpb against the general score (exp03/exp01 bpb, exp04 scores)

- r_post: -0.8086
- rho_post: -0.7133
- r_post_no_math: -0.6524
- r_base: -0.9687
- rho_base: -0.9429
- n_post: 12
- n_base: 6

## Gate 1 at family level (exp04)

- r: 0.8873
- cluster_lo: -0.6358
- cluster_hi: 0.9906
- cluster_valid: 9999
- r_demeaned: 0.9696
- n_families: 6
- perm_p_r: 0.00105
- perm_p_rho: 0.0099

## Blank-line document start (exp01 ablation_docstart.csv)

- math_vs_base_nl: 16.03
- r1_vs_base_nl: 40.21
- r1_vs_math_nl: 20.9
- n_substituted: 5
- gate_r_nl: 0.8354
- gate_rho_nl: 0.6643

## exp05 contrast divided by total drift

- r_sf_12: 0.2406
- r_sf_10: 0.4526

## exp06 correlations, supervision, learning rate

- r_forget_gain_12: -0.9353
- r_forget_gain_marks: -0.9268
- r_drift_gain_12: 0.1328
- r_drift_gain_marks: -0.04386
- loss_tokens_M_tulu: 54.53
- padding_tulu: 0.2114
- loss_tokens_M_smoltalk: 55.26
- padding_smoltalk: 0.2382
- loss_tokens_M_mathcot: 69.36
- padding_mathcot: 0.2533
- lr_marks_min: 1e-05
- lr_marks_max: 2e-05
- lr_stage2_final_min: 1e-05
- marks_within_0.06_passes_after_boundary: 4

## Range-of-three floor

- factor: 1.196
- floor_pp: 2.646

## enwik8 / fresh ratio (exp03)

| model                                     | kind      |   ratio |
|:------------------------------------------|:----------|--------:|
| Qwen/Qwen2.5-0.5B-Instruct                | vs_base   |   1.714 |
| Qwen/Qwen2.5-1.5B-Instruct                | vs_base   |   1.735 |
| Qwen/Qwen2.5-3B-Instruct                  | vs_base   |   2.965 |
| Qwen/Qwen2.5-Math-1.5B                    | vs_base   |   2.463 |
| allenai/OLMo-2-0425-1B-DPO                | vs_base   |   1.364 |
| allenai/OLMo-2-0425-1B-DPO                | vs_parent |   1.36  |
| allenai/OLMo-2-0425-1B-Instruct           | vs_base   |   1.341 |
| allenai/OLMo-2-0425-1B-Instruct           | vs_parent |   0.722 |
| allenai/OLMo-2-0425-1B-SFT                | vs_base   |   1.337 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | vs_base   |   1.73  |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | vs_parent |   0.621 |
| hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo      | vs_base   |   2.181 |
| hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo      | vs_base   |   2.41  |

## exp06 contrasts with 95% window-bootstrap intervals

| recipe   |   step | marks   | contrast   |   point |   csv |    lo |    hi |
|:---------|-------:|:--------|:-----------|--------:|------:|------:|------:|
| tulu     |   1500 | final   | ks         |   1.69  | 1.69  | 1.325 | 2.039 |
| tulu     |   1500 | final   | rf         |   1.851 | 1.851 | 1.583 | 2.122 |
| tulu     |   1760 | 3       | ks         |   2.47  | 2.47  | 2     | 2.93  |
| tulu     |   1760 | 3       | rf         |   2.259 | 2.259 | 1.913 | 2.607 |
| tulu     |   2660 | 6       | ks         |   3.37  | 3.37  | 2.847 | 3.881 |
| tulu     |   2660 | 6       | rf         |   2.363 | 2.363 | 1.94  | 2.773 |
| tulu     |   3489 | final   | ks         |   4.349 | 4.349 | 3.758 | 4.922 |
| tulu     |   3489 | final   | rf         |   2.94  | 2.94  | 2.506 | 3.358 |
| smoltalk |    920 | 3       | ks         |   1.549 | 1.549 | 1.037 | 2.079 |
| smoltalk |    920 | 3       | rf         |   2.589 | 2.589 | 2.204 | 2.979 |
| smoltalk |   1500 | final   | ks         |   2.173 | 2.173 | 1.647 | 2.72  |
| smoltalk |   1500 | final   | rf         |   3.059 | 3.059 | 2.667 | 3.443 |
| smoltalk |   2500 | 6       | ks         |   2.376 | 2.376 | 1.518 | 3.272 |
| smoltalk |   2500 | 6       | rf         |   2.871 | 2.871 | 2.109 | 3.623 |
| smoltalk |   3487 | final   | ks         |   3.41  | 3.41  | 2.437 | 4.411 |
| smoltalk |   3487 | final   | rf         |   3.294 | 3.294 | 2.459 | 4.102 |
| mathcot  |   1500 | final   | ks         |   2.097 | 2.097 | 1.667 | 2.533 |
| mathcot  |   1500 | final   | rf         |   2.278 | 2.278 | 1.947 | 2.616 |
| mathcot  |   1660 | 3       | ks         |   2.861 | 2.861 | 2.418 | 3.324 |
| mathcot  |   1660 | 3       | rf         |   2.931 | 2.931 | 2.593 | 3.283 |
| mathcot  |   2780 | 6       | ks         |   3.414 | 3.414 | 2.772 | 4.087 |
| mathcot  |   2780 | 6       | rf         |   2.935 | 2.935 | 2.513 | 3.374 |
| mathcot  |   3464 | final   | ks         |   4.478 | 4.478 | 3.786 | 5.21  |
| mathcot  |   3464 | final   | rf         |   3.687 | 3.687 | 3.203 | 4.174 |

## exp06 paired K − S differences at the marks

|   mark | pair               |   point |     lo |    hi |
|-------:|:-------------------|--------:|-------:|------:|
|      3 | mathcot - tulu     |   0.392 | -0.096 | 0.908 |
|      3 | mathcot - smoltalk |   1.313 |  0.788 | 1.856 |
|      3 | tulu - smoltalk    |   0.921 |  0.511 | 1.329 |
|      6 | mathcot - tulu     |   0.044 | -0.584 | 0.74  |
|      6 | mathcot - smoltalk |   1.038 |  0.136 | 1.978 |
|      6 | tulu - smoltalk    |   0.994 |  0.303 | 1.683 |

## exp06 K − S at the marks, raw and 7-row median

| recipe   |   mark |   raw |   median7 |
|:---------|-------:|------:|----------:|
| tulu     |      3 | 2.47  |     2.47  |
| tulu     |      6 | 3.37  |     3.37  |
| smoltalk |      3 | 1.549 |     1.709 |
| smoltalk |      6 | 2.376 |     2.433 |
| mathcot  |      3 | 2.861 |     2.861 |
| mathcot  |      6 | 3.414 |     3.706 |

## exp06 pass-boundary jumps

| recipe   |   pass |   step |   before |   jump20 |   jump40 |   loss_before |   loss_after |
|:---------|-------:|-------:|---------:|---------:|---------:|--------------:|-------------:|
| mathcot  |      1 |    900 |    2.183 |    0.007 |    0.128 |         0.55  |        0.492 |
| mathcot  |      2 |   1820 |    3.718 |    0.062 |    0.298 |         0.493 |        0.471 |
| mathcot  |      3 |   2740 |    5.564 |    0.043 |    0.5   |         0.464 |        0.39  |
| smoltalk |      1 |    900 |    2.748 |    0.271 |    0.45  |         0.879 |        0.762 |
| smoltalk |      2 |   1800 |    4.416 |    0.237 |    0.54  |         0.812 |        0.735 |
| smoltalk |      3 |   2700 |    6.192 |    1.878 |    1.879 |         0.726 |        0.588 |
| tulu     |      1 |    860 |    1.604 |    0.149 |    0.311 |         0.84  |        0.765 |
| tulu     |      2 |   1740 |    2.763 |    0.513 |    0.592 |         0.72  |        0.717 |
| tulu     |      3 |   2600 |    4.061 |    0.907 |    1.635 |         0.728 |        0.563 |
| tulu     |      4 |   3480 |    7.044 |    0.066 |    0.066 |         0.588 |        0.536 |

## exp04 per task (points; forgetting in % of base)

| model                                     |   forget_no_arc |   forget_no_mmlu |   d_arc_easy |   d_arc_challenge |   d_mmlu |
|:------------------------------------------|----------------:|-----------------:|-------------:|------------------:|---------:|
| allenai/OLMo-2-0425-1B-SFT                |           -2.03 |             1.88 |        -3.58 |             -1.28 |     5.88 |
| allenai/OLMo-2-0425-1B-DPO                |           -2.91 |             0    |        -2.65 |              0.68 |     4.96 |
| allenai/OLMo-2-0425-1B-Instruct           |           -2.57 |             0.27 |        -2.57 |             -0.43 |     3.95 |
| Qwen/Qwen2.5-0.5B-Instruct                |            0.51 |            -1.06 |         0.55 |              1.28 |    -2.15 |
| hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo      |            0.56 |            -0.34 |         0.13 |              1.62 |    -0.44 |
| Qwen/Qwen2.5-1.5B-Instruct                |           -0.36 |            -1.97 |         4.38 |              1.36 |     0.48 |
| hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo      |            0.34 |            -1.86 |         4.21 |              2.56 |     0.04 |
| Qwen/Qwen2.5-Math-1.5B                    |           15.62 |            11.26 |        -7.32 |             -4.44 |   -16.89 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |           21.34 |            18.46 |       -15.49 |            -10.41 |   -23.25 |
| Qwen/Qwen2.5-3B-Instruct                  |           -0.49 |            -0.7  |        -0.13 |              0.51 |    -0.57 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct       |           -0.03 |             4.26 |       -10.56 |             -3.75 |    -0.35 |
| Qwen/Qwen3-1.7B                           |            6.41 |             3.74 |         1.31 |             -2.05 |    -5.88 |

## exp06 per task

|                    |   forget_no_arc |   forget_no_mmlu |   d_arc_easy |   d_arc_challenge |   d_mmlu |
|:-------------------|----------------:|-----------------:|-------------:|------------------:|---------:|
| ('tulu', 1500)     |           -2.01 |            -0.5  |        -0.5  |              0.85 |     3.55 |
| ('tulu', 1760)     |           -2.05 |             1.43 |        -2.78 |             -1.02 |     5.57 |
| ('tulu', 2660)     |           -2.68 |            -0.15 |        -0.93 |              0.68 |     5.66 |
| ('tulu', 3489)     |           -2    |             0.61 |        -2.1  |              0.26 |     4.82 |
| ('smoltalk', 920)  |           -1.94 |             0.3  |        -1.35 |              0.17 |     4.39 |
| ('smoltalk', 1500) |           -1.81 |             0.63 |        -1.81 |              0.43 |     4.91 |
| ('smoltalk', 2500) |           -1.93 |             0.43 |        -1.6  |              0.17 |     4.52 |
| ('smoltalk', 3487) |           -1.77 |             0.76 |        -2.82 |              0.94 |     4.69 |
| ('mathcot', 1500)  |            0.37 |             4.38 |        -9.43 |             -4.69 |    -1.19 |
| ('mathcot', 1660)  |           -0.04 |             4.31 |        -9.68 |             -4.78 |    -0.79 |
| ('mathcot', 2780)  |            1.45 |             3.98 |        -7.28 |             -3.75 |    -1.93 |
| ('mathcot', 3464)  |            1.5  |             4.42 |        -7.24 |             -4.78 |    -1.67 |

# Position audit: bits/token by position within window

bpt = mean bits per token. 'pos128+' = tokens with >= 128 tokens of in-window context; 'pos0' = first text token of each window (conditioned only on the doc-start token or chat prefix).

| pair | corpus | base raw pos128+ | post raw pos128+ | Δ% raw pos128+ | post chat pos128+ | Δ% post chat vs base raw pos128+ | base raw pos0 | post raw pos0 | post chat pos0 |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5 | arxiv2026 | 3.857 | 3.893 | +0.9% | 3.925 | +1.8% | 22.78 | 20.40 | 32.11 |
| qwen2.5 | enwik8_ctrl | 2.119 | 2.154 | +1.7% | 2.182 | +3.0% | 16.84 | 15.79 | 24.79 |
| qwen2.5 | gutenberg2026 | 3.535 | 3.579 | +1.2% | 3.615 | +2.2% | 21.72 | 19.25 | 27.92 |
| qwen2.5 | wiki2026 | 3.130 | 3.154 | +0.8% | 3.174 | +1.4% | 20.40 | 18.42 | 28.75 |
| qwen2.5-math | arxiv2026 | 4.009 | 4.955 | +23.6% | 4.603 | +14.8% | 27.03 | 20.14 | 44.93 |
| qwen2.5-math | enwik8_ctrl | 3.286 | 3.738 | +13.8% | 3.410 | +3.8% | 20.16 | 16.39 | 44.19 |
| qwen2.5-math | gutenberg2026 | 4.593 | 5.321 | +15.8% | 4.989 | +8.6% | 25.79 | 17.25 | 42.76 |
| qwen2.5-math | wiki2026 | 4.148 | 4.774 | +15.1% | 4.511 | +8.7% | 23.69 | 18.95 | 46.58 |
| qwen3 | arxiv2026 | 3.783 | 4.454 | +17.7% | 4.965 | +31.2% | 19.83 | 19.32 | 41.75 |
| qwen3 | enwik8_ctrl | 2.120 | 2.680 | +26.4% | 3.085 | +45.5% | 15.91 | 14.63 | 40.14 |
| qwen3 | gutenberg2026 | 3.576 | 4.156 | +16.2% | 4.771 | +33.4% | 19.49 | 17.56 | 41.20 |
| qwen3 | wiki2026 | 3.167 | 3.698 | +16.8% | 4.304 | +35.9% | 18.42 | 17.22 | 43.05 |
| smollm2 | arxiv2026 | 3.932 | 4.074 | +3.6% | 4.066 | +3.4% | 26.46 | 21.48 | 20.44 |
| smollm2 | enwik8_ctrl | 2.272 | 2.564 | +12.8% | 1.974 | -13.1% | 23.71 | 18.91 | 17.56 |
| smollm2 | gutenberg2026 | 3.184 | 3.310 | +4.0% | 3.319 | +4.2% | 26.17 | 21.16 | 18.62 |
| smollm2 | wiki2026 | 3.014 | 3.127 | +3.7% | 3.125 | +3.7% | 26.23 | 21.39 | 20.24 |

# Concentration of the gap (RAW, token-aligned within each pair)

Δ = post − base bits on the same token. 'top1% share' = fraction of the total Δ contributed by the 1% of tokens with the largest Δ; 'Δ>10 share' = fraction from tokens where post pays >10 extra bits.

| pair | corpus | Δ total bits | Δ % | tokens where post worse | median Δ (bits) | top1% share | Δ>10 share |
|---|---|---|---|---|---|---|---|
| qwen2.5 | arxiv2026 | +698 | +0.8% | 50% | +0.000 | 0.56 | 0.02 |
| qwen2.5 | enwik8_ctrl | +1095 | +1.6% | 45% | -0.000 | 0.63 | 0.00 |
| qwen2.5 | wiki2026 | +483 | +0.7% | 50% | -0.000 | 0.78 | 0.00 |
| qwen2.5 | gutenberg2026 | +1115 | +1.2% | 56% | +0.009 | 0.39 | 0.00 |
| qwen3 | arxiv2026 | +13654 | +16.9% | 49% | -0.000 | 0.14 | 0.04 |
| qwen3 | enwik8_ctrl | +17317 | +24.9% | 35% | -0.004 | 0.20 | 0.13 |
| qwen3 | wiki2026 | +11877 | +16.1% | 43% | -0.005 | 0.17 | 0.06 |
| qwen3 | gutenberg2026 | +14446 | +15.6% | 49% | -0.001 | 0.15 | 0.03 |
| smollm2 | arxiv2026 | +2936 | +3.4% | 52% | +0.004 | 0.30 | 0.00 |
| smollm2 | enwik8_ctrl | +10364 | +12.5% | 50% | +0.000 | 0.48 | 0.40 |
| smollm2 | wiki2026 | +2670 | +3.6% | 49% | -0.000 | 0.38 | 0.00 |
| smollm2 | gutenberg2026 | +3296 | +3.8% | 55% | +0.007 | 0.30 | 0.01 |
| qwen2.5-math | arxiv2026 | +18191 | +20.9% | 63% | +0.212 | 0.11 | 0.03 |
| qwen2.5-math | enwik8_ctrl | +10827 | +9.8% | 49% | -0.000 | 0.31 | 0.20 |
| qwen2.5-math | wiki2026 | +12531 | +12.8% | 53% | +0.015 | 0.19 | 0.11 |
| qwen2.5-math | gutenberg2026 | +16584 | +13.8% | 59% | +0.208 | 0.14 | 0.03 |

# Uncertainty: paired bootstrap over windows (5000 resamples, seed 0)

Resamples whole windows (the same windows for base and post), so the CI reflects text-sample variation within each corpus, not variation across corpora or seeds.

| pair | corpus | condition (post vs base) | n windows | Δ% | 95% CI |
|---|---|---|---|---|---|
| qwen2.5 | arxiv2026 | raw vs raw | 21 | +0.8% | [+0.7%, +1.0%] |
| qwen2.5 | arxiv2026 | chat vs raw | 21 | +2.4% | [+2.2%, +2.6%] |
| qwen2.5 | arxiv2026 | chat vs chat | 21 | +1.6% | [+1.3%, +1.8%] |
| qwen2.5 | enwik8_ctrl | raw vs raw | 30 | +1.6% | [+1.3%, +1.9%] |
| qwen2.5 | enwik8_ctrl | chat vs raw | 30 | +4.4% | [+3.8%, +5.1%] |
| qwen2.5 | enwik8_ctrl | chat vs chat | 30 | +2.8% | [+2.3%, +3.3%] |
| qwen2.5 | wiki2026 | raw vs raw | 22 | +0.7% | [+0.5%, +0.8%] |
| qwen2.5 | wiki2026 | chat vs raw | 22 | +2.2% | [+2.0%, +2.5%] |
| qwen2.5 | wiki2026 | chat vs chat | 22 | +1.4% | [+1.1%, +1.6%] |
| qwen2.5 | gutenberg2026 | raw vs raw | 25 | +1.2% | [+1.0%, +1.4%] |
| qwen2.5 | gutenberg2026 | chat vs raw | 25 | +3.3% | [+3.0%, +3.7%] |
| qwen2.5 | gutenberg2026 | chat vs chat | 25 | +2.3% | [+1.9%, +2.7%] |
| qwen3 | arxiv2026 | raw vs raw | 21 | +16.9% | [+15.8%, +18.0%] |
| qwen3 | arxiv2026 | chat vs raw | 21 | +36.3% | [+34.1%, +38.7%] |
| qwen3 | arxiv2026 | chat vs chat | 21 | +34.8% | [+32.6%, +37.2%] |
| qwen3 | enwik8_ctrl | raw vs raw | 30 | +24.9% | [+22.2%, +27.9%] |
| qwen3 | enwik8_ctrl | chat vs raw | 30 | +51.2% | [+47.4%, +55.7%] |
| qwen3 | enwik8_ctrl | chat vs chat | 30 | +48.3% | [+44.8%, +52.3%] |
| qwen3 | wiki2026 | raw vs raw | 22 | +16.1% | [+15.2%, +17.0%] |
| qwen3 | wiki2026 | chat vs raw | 22 | +41.2% | [+38.6%, +44.0%] |
| qwen3 | wiki2026 | chat vs chat | 22 | +39.8% | [+37.2%, +42.7%] |
| qwen3 | gutenberg2026 | raw vs raw | 25 | +15.6% | [+14.9%, +16.4%] |
| qwen3 | gutenberg2026 | chat vs raw | 25 | +38.1% | [+35.2%, +41.3%] |
| qwen3 | gutenberg2026 | chat vs chat | 25 | +36.5% | [+33.8%, +39.6%] |
| smollm2 | arxiv2026 | raw vs raw | 22 | +3.4% | [+3.1%, +3.6%] |
| smollm2 | arxiv2026 | chat vs raw | 22 | +3.2% | [+2.3%, +3.8%] |
| smollm2 | enwik8_ctrl | raw vs raw | 34 | +12.5% | [+6.2%, +20.5%] |
| smollm2 | enwik8_ctrl | chat vs raw | 34 | -11.1% | [-31.4%, +9.2%] |
| smollm2 | wiki2026 | raw vs raw | 23 | +3.6% | [+2.9%, +4.2%] |
| smollm2 | wiki2026 | chat vs raw | 23 | +2.5% | [-0.7%, +4.6%] |
| smollm2 | gutenberg2026 | raw vs raw | 26 | +3.8% | [+3.1%, +4.3%] |
| smollm2 | gutenberg2026 | chat vs raw | 26 | +4.2% | [+3.1%, +5.1%] |
| qwen2.5-math | arxiv2026 | raw vs raw | 21 | +20.9% | [+18.4%, +23.2%] |
| qwen2.5-math | arxiv2026 | chat vs raw | 21 | +13.3% | [+12.0%, +14.5%] |
| qwen2.5-math | arxiv2026 | chat vs chat | 21 | +19.0% | [+17.9%, +20.1%] |
| qwen2.5-math | enwik8_ctrl | raw vs raw | 30 | +9.8% | [+5.1%, +14.3%] |
| qwen2.5-math | enwik8_ctrl | chat vs raw | 30 | +1.2% | [-3.5%, +5.6%] |
| qwen2.5-math | enwik8_ctrl | chat vs chat | 30 | +22.8% | [+20.8%, +25.1%] |
| qwen2.5-math | wiki2026 | raw vs raw | 22 | +12.8% | [+9.7%, +15.8%] |
| qwen2.5-math | wiki2026 | chat vs raw | 22 | +7.3% | [+4.6%, +9.5%] |
| qwen2.5-math | wiki2026 | chat vs chat | 22 | +14.9% | [+13.6%, +16.2%] |
| qwen2.5-math | gutenberg2026 | raw vs raw | 25 | +13.8% | [+11.8%, +15.8%] |
| qwen2.5-math | gutenberg2026 | chat vs raw | 25 | +7.1% | [+5.9%, +8.3%] |
| qwen2.5-math | gutenberg2026 | chat vs chat | 25 | +14.1% | [+13.3%, +15.0%] |

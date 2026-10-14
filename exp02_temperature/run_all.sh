#!/usr/bin/env bash
# exp02: does a single temperature recover the post-training compression gap on fresh text?
#
# For the 8 exp01 models x 4 corpora (RAW, same windows and doc-start token as exp01): one fp16 forward
# pass per window, bits at T in {0.50, 0.55, ..., 1.60}; T* fit on even windows (grid argmin +
# 3-point parabola in log T), evaluated on odd windows. Then tables, bootstrap CIs and the figure.
#
# Rule, stated before the main run (2026-10-14):
#   recovered = (bpb_post(1) - bpb_post(T*)) / (bpb_post(1) - bpb_base(1)), odd windows.
#   Headline per pair = mean of the per-corpus recovered fraction over the 3 fresh corpora.
#   If both big-penalty pairs (Qwen3, Qwen2.5-Math -> R1-Distill) recover >= 70%, the gap is mostly a
#   confidence shift (supports "calibration", weakens "forgetting"); if both recover <= 30%, it is
#   mostly a change in beliefs; otherwise mixed (reported per pair).
# Budget rule: if any model takes > 8 min per corpus, rerun everything with --max-bytes 60000.
#
# Resumable: `score_temperature.py run` skips (model, corpus) rows already in results/runs.csv.
# To start over: rm -rf results logs. Needs the exp01 models in the HF cache and exp01's
# results/tokens/*.npy for the T = 1 sanity check. ~30 min on an M2 Pro 16 GB.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
mkdir -p logs results
$PY score_temperature.py run --models Qwen/Qwen3-1.7B --corpora wiki2026 > logs/smoke.log 2>&1  # sanity first
$PY score_temperature.py run                  >> logs/run.log 2>&1   # 8 models x 4 corpora, one process each
# the 4 runs whose exp01 reference rows came from exp01's superseded .half() path: re-score with today's code
$PY score_temperature.py recheck --models Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct \
    --corpora arxiv2026 enwik8_ctrl > logs/recheck.log 2>&1
$PY analyze_temperature.py                    >  logs/analyze.log 2>&1  # tables, CIs, figure
echo "done: results/{runs.csv,summary.md,fig_temperature.png}"

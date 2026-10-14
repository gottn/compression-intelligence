#!/usr/bin/env bash
# exp03: how far does each post-training stage move a model from its pretraining distribution?
#
# Δbpb on fresh 2026 text of every stage versus its family base (and versus its parent), RAW condition,
# exp01's corpora, windows and doc-start rule. Stage table: cilab/stages.py (14 models, 4 families:
# OLMo-2-0425-1B base/SFT/DPO/RLVR; Qwen2.5-0.5B, -1.5B, -3B with Instruct, SimpleRL-Zoo (pure GRPO from
# the base), Qwen2.5-Math and R1-Distill). All 14 models are scored in ONE AAU job (one GPU, fp16) so every
# number comes from the same machine. Analysis and figure run on the Mac.
#
# Pre-registered expectations: RESULTS.md, section "Expectations", committed before the job ran.
#
# Steps
#   bash run_all.sh preflight   Mac: Qwen2.5-0.5B on enwik8_ctrl only (+ tokenizer identity) -> logs/
#   (AAU) commit, push, then on the front end:
#       ssh aau 'cd ~/projects/compression-intelligence && infra/cluster/submit.sh exp03'
#       resubmit after preemption: RESUME_FROM=~/runs/ci/exp03/<run>/results infra/cluster/submit.sh exp03
#   (Mac) rsync -az aau:~/runs/ci/exp03/<run>/results/ exp03_staged_drift/results/
#         cp ~/runs/ci/exp03/<run>/slurm-<id>.out -> logs/   (via rsync)
#   bash run_all.sh analyze     Mac: results/summary.md, results/stage_deltas.csv, results/fig_stages.png
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p exp03_staged_drift/logs
case "${1:-analyze}" in
  preflight)
    $PY exp03_staged_drift/score_stages.py tokcheck --results exp03_staged_drift/logs/preflight \
        > exp03_staged_drift/logs/preflight_tokcheck.log 2>&1
    $PY exp03_staged_drift/score_stages.py run --models Qwen/Qwen2.5-0.5B --corpora enwik8_ctrl \
        --results exp03_staged_drift/logs/preflight > exp03_staged_drift/logs/preflight_run.log 2>&1 ;;
  analyze)
    $PY exp03_staged_drift/analyze_stages.py > exp03_staged_drift/logs/analyze.log 2>&1
    echo "done: exp03_staged_drift/results/{summary.md,stage_deltas.csv,fig_stages.png}" ;;
  *) echo "usage: run_all.sh [preflight|analyze]"; exit 1 ;;
esac

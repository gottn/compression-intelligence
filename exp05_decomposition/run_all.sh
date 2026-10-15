#!/usr/bin/env bash
# exp05: does post-training drift sit on knowledge tokens or on style tokens?
#
# Splits the per-token bits already on disk (exp03: 14 models × 4 corpora, AAU T4; exp01: the SmolLM2 and Qwen3
# pairs, RAW rows, Mac) into token classes and tests which class's drift predicts exp04's forgetting and which its
# IFEval chat gain. Classes: spaCy en_core_web_sm POS of the word under each token's first non-space character;
# KNOWLEDGE = PROPN, NUM and rare NOUN/ADJ/VERB (outside wordfreq's top 2,000), STYLE = function words,
# punctuation and whitespace, MIDDLE = the rest. No new scoring; Mac only, CPU, about half a minute per step.
#
# Pre-registered definitions and tests: RESULTS.md, committed before any class-level drift was computed.
#
# Steps
#   VIRTUAL_ENV=.venv uv pip install -r exp05_decomposition/requirements.txt   (once)
#   bash run_all.sh align     classes per tokenizer group and corpus + alignment checks on all 72 (model, corpus)
#                             pairs -> results/{classes/, alignment.json, class_counts.csv, class_examples.md}
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=exp05_decomposition
mkdir -p $OUT/logs $OUT/results
export TOKENIZERS_PARALLELISM=false
case "${1:-align}" in
  align)
    $PY $OUT/analyze_decomposition.py align > $OUT/logs/align.log 2>&1
    echo "done: $OUT/results/{classes/,alignment.json,class_counts.csv,class_examples.md}" ;;
  *) echo "usage: run_all.sh [align]"; exit 1 ;;
esac

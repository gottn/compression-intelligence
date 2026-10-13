#!/usr/bin/env bash
# Full pipeline for exp01 (base vs post-trained bpb on fresh 2026 text).
# Resumable: `score_bpb.py run` skips (model, condition, corpus) rows already in results/bpb.csv.
# To start over: rm -rf results logs. Corpora in data/ are a dated snapshot: keep them (re-collecting
# later gives different text). ~1 h on an M2 Pro 16 GB after the ~27 GB model download.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
if [ ! -x "$PY" ]; then
  (cd .. && uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r exp01_posttraining_bpc/requirements.txt)
fi
mkdir -p logs results
$PY download_models.py                    >> logs/download.out 2>&1   # HF cache, one model at a time
[ -f data/wiki2026.txt ] || $PY collect_corpora.py >> logs/collect.log 2>&1  # Wikipedia API is throttled: ~10 min
$PY score_bpb.py tokcheck                 >  logs/tokcheck.log 2>&1   # tokenizer identity per pair
$PY score_bpb.py sanity                   >  logs/sanity.log 2>&1    # GPT-2 control + R1 doc-start ablation
$PY score_bpb.py run                      >> logs/run_main.log 2>&1  # main table, one process per model
$PY score_bpb.py docstart                 >  logs/docstart.log 2>&1  # RAW with "\n\n" doc-start (robustness)
# targeted float32 re-scoring of windows that showed extreme surprisals (appends to results/fp32_audit.jsonl)
$PY score_bpb.py audit --models HuggingFaceTB/SmolLM2-1.7B --corpora enwik8_ctrl --condition raw --windows 0 15 27 3 >> logs/audit.log 2>&1
$PY score_bpb.py audit --models HuggingFaceTB/SmolLM2-1.7B --corpora enwik8_ctrl --condition raw --windows 15 --device cpu >> logs/audit.log 2>&1
$PY score_bpb.py audit --models Qwen/Qwen3-1.7B --corpora enwik8_ctrl --condition chat --windows 28 5 >> logs/audit.log 2>&1
$PY score_bpb.py audit --models Qwen/Qwen3-1.7B --corpora wiki2026 --condition raw --windows 3 10 >> logs/audit.log 2>&1
$PY analyze_positions.py                  >  logs/positions.log 2>&1  # position / concentration / bootstrap
$PY plot_results.py                       >  logs/plot.log 2>&1       # figure + summary.md + sanity.md
echo "done: results/{bpb.csv,summary.md,sanity.md,positions.md,fig_bpb_pairs.png}"

#!/usr/bin/env bash
# exp06: at matched drift, does the knowledge − style contrast predict forgetting and ignore the gain?
#
# One base (OLMo-2-0425-1B), three SFT recipes that differ only in content (tulu: Tülu 3 SFT mix; smoltalk: SmolTalk
# "all"; mathcot: OpenR1-Math "all" R1 answers), token-matched subsets (45M template tokens each, OLMo-2-Instruct
# chat template, loss on assistant tokens, ≤ 2,048 tokens, packed), identical training (AdamW 2e-5 cosine, 32 × 2,048
# tokens per update, 1,500 updates or 5 h). Every 20 updates the fp32 weights are copied to bf16 and score exp01's
# corpora (RAW, exp03 mechanics); checkpoints are saved where Δbpb_fresh first reaches 3%, 6% and 10%, and at the
# stop. Each checkpoint and the base (with the Instruct template) get exp04's loglik + gsm8k_chat + ifeval_chat.
#
# Pre-registered definitions, tests and expectations: RESULTS.md, committed in 9cca20f before any training.
#
# Steps (AAU, after the commit is pushed; from the Mac; R = ~/projects/compression-intelligence):
#   ssh -o BatchMode=yes aau "cd R && infra/cluster/submit.sh exp06_setup"                      CPU, data (~30-60 min)
#   ssh -o BatchMode=yes aau "cd R && PREP=<setup run>/data JOB_ARGS='--max-steps 50' \
#       infra/cluster/submit.sh exp06_train --array=0 --time=00:40:00"                         dry run, tulu 50 updates
#   ssh -o BatchMode=yes aau "cd R && PREP=<setup run>/data infra/cluster/submit.sh exp06_train --array=0-2"
#   ssh -o BatchMode=yes aau "cd R && JOB_ARGS='--model-path allenai/OLMo-2-0425-1B --tokenizer \
#       allenai/OLMo-2-0425-1B-Instruct --label exp06__base' infra/cluster/submit.sh exp06_eval --array=0"
#   ssh -o BatchMode=yes aau "cd R && CKPT_LIST=<train run>/checkpoints.txt \
#       infra/cluster/submit.sh exp06_eval --array=0-<n-1>%4"
#   (Mac) SETUP=<setup run> TRAIN=<train run> EVAL=<eval run> BASE_EVAL=<base eval run> bash run_all.sh fetch
#   bash run_all.sh analyze     Mac: results/{checkpoints.csv, trajectories.csv, correlations.csv, tests.csv,
#                               summary.md, fig_matched.png}
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=exp06_matched_drift
mkdir -p $OUT/logs $OUT/results
case "${1:-analyze}" in
  fetch)
    if [ -n "${SETUP:-}" ]; then  # provenance and logs only; the token arrays stay on the cluster (re-made by prep)
      rsync -az --include='data/' --include='*.prov.json' --include='summary.json' --include='slurm-*.out' \
            --include='commit.txt' --include='pip_freeze.txt' --exclude='*' "aau:$SETUP/" $OUT/logs/setup/
    fi
    if [ -n "${TRAIN:-}" ]; then  # drift logs, train logs, per-token bits; never the checkpoint weights
      rsync -az --exclude='ckpt/' "aau:$TRAIN/" $OUT/results/train/
      rsync -az --include='*/' --include='exp06_checkpoint.json' --exclude='*' "aau:$TRAIN/" $OUT/results/train/
    fi
    for v in EVAL BASE_EVAL; do
      run="${!v:-}"
      [ -n "$run" ] || continue
      rsync -az --include='*/' --include='summary.json' --include='done.json' --include='results.json' \
            --exclude='*' "aau:$run/results/" $OUT/results/lm_eval/
      rsync -az --include='slurm-*.out' --include='commit.txt' --exclude='*' "aau:$run/" "$OUT/logs/$(echo "$v" | tr A-Z a-z)/"
    done ;;
  analyze)
    $PY $OUT/analyze_matched.py > $OUT/logs/analyze.log 2>&1
    echo "done: $OUT/results/{checkpoints.csv,trajectories.csv,correlations.csv,tests.csv,summary.md,fig_matched.png}" ;;
  *) echo "usage: run_all.sh [fetch|analyze]"; exit 1 ;;
esac

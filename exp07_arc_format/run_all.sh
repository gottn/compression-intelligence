#!/usr/bin/env bash
# exp07: is the ARC "forgetting" of exp04/exp06 (mathcot, SmolLM2-Instruct) knowledge loss or an answer-format effect,
# and is the chat recipes' "no forgetting" an MMLU letter-format gain cancelling an ARC loss?
#
# Every checkpoint (exp04's 18 public rows, plus exp06's tulu and mathcot stage 1 retrained here, whose checkpoints
# were deleted) gets the same ARC under three formats (B1 0-shot cloze = exp04's default, B2 25-shot, B3 0-shot with
# the chat template, the base scored with the checkpoint's own template), MMLU under letter choice and answer-text
# continuation (B4, same 2,280 questions), HellaSwag (B5, control) and, for the retrained ones, PIQA + WinoGrande (B0).
# lm-eval 0.4.13 on A10s with exp04's settings; per-item outcomes are kept so base and checkpoint can be paired.
#
# Pre-registered definitions, rules (K/F for mathcot and SmolLM2-Instruct; format/survives for the MMLU gain;
# reproduction of exp06 stage 1) and expectations: RESULTS.md, committed in 41728fd before any job.
#
# Steps (AAU, after the commit is pushed; from the Mac; R = ~/projects/compression-intelligence):
#   dry run, 2 public checkpoints x every group x 10 docs (A10):
#   ssh -o BatchMode=yes aau "cd R && CKPT_LIST=R/exp07_arc_format/checkpoints_public.tsv JOB_ARGS='--limit 10 --groups \
#       b1_arc_0shot,b4_mmlu_letter,b4_mmlu_cont,b5_hellaswag,b0_piqa_wino,b3_arc_0shot_chat,b2_arc_25shot' \
#       infra/cluster/submit.sh exp07_eval --array=4-5 --time=00:40:00"
#   data (CPU), training (2 A40), the public battery (indices 0-17, 2 at a time while training runs) and the battery
#   of the retrained checkpoints (indices 18-25, queued behind training) in one call:
#   ssh -o BatchMode=yes aau "cd R && S=\$(JOB_ARGS='--recipes tulu mathcot' infra/cluster/submit.sh exp06_setup) ...
#       PREP=<S>/data infra/cluster/submit.sh exp07_train --array=0-1 --dependency=afterok:<setup job>
#       cp exp07_arc_format/checkpoints_public.tsv <T>/checkpoints.txt
#       RUN_IN=<T> CKPT_LIST=<T>/checkpoints.txt infra/cluster/submit.sh exp07_eval --array=0-17%2
#       RUN_IN=<T> CKPT_LIST=<T>/checkpoints.txt infra/cluster/submit.sh exp07_eval --array=18-25%2 \
#           --dependency=afterany:<train array>"
#   (Mac) SETUP=<S> TRAIN=<T> DRY=<dry run> bash run_all.sh fetch
#   bash run_all.sh analyze     Mac, seconds: results/{scores_long.csv, format_table.csv, verdicts.csv,
#                               reproduction.csv, sanity.csv, summary.md, fig_format.png}
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=exp07_arc_format
mkdir -p $OUT/logs $OUT/results
lm_eval_files=(--include='*/' --include='summary.json' --include='done.json' --include='results.json'
               --include='items.json' --exclude='*')
case "${1:-analyze}" in
  fetch)
    if [ -n "${SETUP:-}" ]; then  # provenance and logs only; the token arrays stay on the cluster
      rsync -az --include='data/' --include='*.prov.json' --include='summary.json' --include='slurm-*.out' \
            --include='commit.txt' --include='pip_freeze.txt' --exclude='*' "aau:$SETUP/" $OUT/logs/setup/
    fi
    if [ -n "${DRY:-}" ]; then
      rsync -az --include='slurm-*.out' --include='commit*.txt' --exclude='*' "aau:$DRY/" $OUT/logs/dryrun/
      rsync -az --prune-empty-dirs "${lm_eval_files[@]}" "aau:$DRY/results/" $OUT/logs/dryrun/lm_eval/
    fi
    if [ -n "${TRAIN:-}" ]; then  # drift/train logs and checkpoint metadata, never weights; per-token bits of the
      dst=$OUT/results/train      # reference and checkpoint updates only
      rsync -az --exclude='ckpt/' --exclude='results/' --exclude='tokens/' --exclude='slurm-*.out' \
            --exclude='commit*.txt' "aau:$TRAIN/" "$dst/"
      rsync -az --exclude='results/' --include='*/' --include='exp06_checkpoint.json' --exclude='*' \
            --prune-empty-dirs "aau:$TRAIN/" "$dst/"
      rsync -az --include='slurm-*.out' --include='commit*.txt' --exclude='*' "aau:$TRAIN/" $OUT/logs/train_eval/
      for r in "$dst"/*/; do
        r=$(basename "$r")
        [ -f "$dst/$r/checkpoints.txt" ] || continue
        inc=("--include=step00000__*.npy")
        for st in $(cut -f4 "$dst/$r/checkpoints.txt"); do inc+=("--include=$(printf 'step%05d' "$st")__*.npy"); done
        mkdir -p "$dst/$r/tokens"
        rsync -az "${inc[@]}" --exclude='*' "aau:$TRAIN/$r/tokens/" "$dst/$r/tokens/"
      done
      rsync -az --prune-empty-dirs "${lm_eval_files[@]}" "aau:$TRAIN/results/" $OUT/results/lm_eval/
    fi ;;
  analyze)
    $PY $OUT/analyze_format.py > $OUT/logs/analyze.log 2>&1
    echo "done: $OUT/results/{scores_long.csv,format_table.csv,verdicts.csv,reproduction.csv,sanity.csv,summary.md,fig_format.png}" ;;
  *) echo "usage: run_all.sh [fetch|analyze]"; exit 1 ;;
esac

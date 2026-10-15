#!/usr/bin/env bash
# exp06: exp04's lm-eval driver on one exp06 checkpoint per array task (groups loglik, gsm8k_chat, ifeval_chat).
# Submit only via submit.sh. Checkpoints come from the train run's list (sorted, one line per checkpoint), read at
# run time, so the array can be queued behind the training array; it is sized for the most checkpoints possible
# (3 recipes x 4) and an index without a checkpoint exits 0. Results go into the train run directory (RUN_IN):
#   T=~/runs/ci/exp06_train/<run>; RUN_IN=$T CKPT_LIST=$T/checkpoints.txt \
#     infra/cluster/submit.sh exp06_eval --array=0-11%4 --dependency=afterany:<train array job id>
# The base, with the Instruct tokenizer's chat template (no list; the target comes in JOB_ARGS):
#   JOB_ARGS="--model-path allenai/OLMo-2-0425-1B --tokenizer allenai/OLMo-2-0425-1B-Instruct --label exp06__base" \
#     infra/cluster/submit.sh exp06_eval --array=0
# Each task writes $CI_OUT/results/<label>/ only; resumable after a requeue (groups with done.json are skipped).
# A10 pinned, as exp04, so every checkpoint and the base run the same fp16 kernels.
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
set -euo pipefail
IDX="${SLURM_ARRAY_TASK_ID:?submit as an array}"
EVAL_GROUPS="loglik,gsm8k_chat,ifeval_chat"
TARGET=""
if [ -n "${CKPT_LIST:-}" ]; then
  LINE="$( [ -f "$CKPT_LIST" ] && LC_ALL=C sort -u "$CKPT_LIST" | sed -n "$((IDX + 1))p" || true)"
  # the array is sized for the most checkpoints possible (3 recipes x 4); an index without one has nothing to do
  [ -n "$LINE" ] || { echo "no checkpoint at index $IDX in $CKPT_LIST: nothing to evaluate"; exit 0; }
  TARGET="--model-path $(printf '%s' "$LINE" | cut -f2) --label $(printf '%s' "$LINE" | cut -f1)"
fi
export HF_HOME="$HOME/hf_cache" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
CONTAINER="${CONTAINER:-/home/container/pytorch/pytorch_25.09.sif}"
VENV="$HOME/pydeps/ci-venv"
echo "commit $CI_SHA task $IDX (job $SLURM_JOB_ID, array $SLURM_ARRAY_JOB_ID) on $(hostname) at $(date -Is), gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader), target=$TARGET $JOB_ARGS, out=$CI_OUT"
mkdir -p "$CI_OUT/results"
# bash -c, not -l: the image's login profiles choke on CUDA version strings. No pipes to tail: stream the log.
srun singularity exec --nv "$CONTAINER" bash -c "
  set -euo pipefail
  cd $CI_ROOT
  $VENV/bin/python exp04_benchmarks/run_eval.py run $TARGET --groups $EVAL_GROUPS --out $CI_OUT/results $JOB_ARGS
"
echo "exp06_eval task $IDX done at $(date -Is)"

#!/usr/bin/env bash
# exp07: the ARC format battery (exp07_arc_format/run_battery.py: B1 0-shot ARC, B2 25-shot ARC, B3 ARC with a chat
# template, B4 MMLU letter and continuation, B5 HellaSwag, B0 PIQA+WinoGrande for retrained checkpoints) on one
# checkpoint per array task. Submit only via submit.sh. The checkpoint is line IDX+1 (file order, blank lines skipped)
# of CKPT_LIST, read at run time: the training run's checkpoints.txt holds the 18 public rows first, then the lines
# the training job appends, so the public part (indices 0-17) runs at once and the rest is queued behind training:
#   T=~/runs/ci/exp07_train/<run>; cp exp07_arc_format/checkpoints_public.tsv $T/checkpoints.txt
#   RUN_IN=$T CKPT_LIST=$T/checkpoints.txt infra/cluster/submit.sh exp07_eval --array=0-17%2
#   RUN_IN=$T CKPT_LIST=$T/checkpoints.txt infra/cluster/submit.sh exp07_eval --array=18-25%2 \
#       --dependency=afterany:<train array job id>
# The second array is sized for up to 8 retrained checkpoints (2 recipes x {mark, final} x a requeue); an index
# without a line exits 0. Dry run (new run directory, 2 public checkpoints, 10 docs per task):
#   CKPT_LIST=$PWD/exp07_arc_format/checkpoints_public.tsv JOB_ARGS="--limit 10" \
#       infra/cluster/submit.sh exp07_eval --array=4-5 --time=00:40:00
# Each task writes $CI_OUT/results/<label>/ only; resumable (groups with a done.json are skipped), so a timed-out
# index is simply resubmitted. A10 pinned, as exp04 and exp06, so every checkpoint runs the same fp16 kernels.
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
set -euo pipefail
IDX="${SLURM_ARRAY_TASK_ID:?submit as an array}"
: "${CKPT_LIST:?CKPT_LIST=<checkpoints list>}"
LINE="$( [ -f "$CKPT_LIST" ] && grep -v '^[[:space:]]*$' "$CKPT_LIST" | sed -n "$((IDX + 1))p" || true)"
[ -n "$LINE" ] || { echo "no checkpoint at index $IDX in $CKPT_LIST: nothing to evaluate"; exit 0; }
export HF_HOME="$HOME/hf_cache" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONHASHSEED=0
CONTAINER="${CONTAINER:-/home/container/pytorch/pytorch_25.09.sif}"
VENV="$HOME/pydeps/ci-venv"
echo "commit $CI_SHA task $IDX (job $SLURM_JOB_ID, array $SLURM_ARRAY_JOB_ID) on $(hostname) at $(date -Is), gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader), line=$(printf '%s' "$LINE" | cut -f1-2 | tr '\t' ' ') $JOB_ARGS, out=$CI_OUT"
mkdir -p "$CI_OUT/results"
# bash -c, not -l: the image's login profiles choke on CUDA version strings. No pipes to tail: stream the log.
srun singularity exec --nv "$CONTAINER" bash -c "
  set -euo pipefail
  cd $CI_ROOT
  $VENV/bin/python exp07_arc_format/run_battery.py run --list $CKPT_LIST --index $IDX --out $CI_OUT/results $JOB_ARGS
"
echo "exp07_eval task $IDX done at $(date -Is)"

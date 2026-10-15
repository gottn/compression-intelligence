#!/usr/bin/env bash
# exp06: full SFT of OLMo-2-0425-1B on one recipe (array index -> recipe: 0 tulu, 1 smoltalk, 2 mathcot), with the
# drift hook every 20 updates and bf16 checkpoints at the 3/6/10% Δbpb_fresh marks and at the stop
# (exp06_matched_drift/train_sft.py). Submit only via submit.sh, after exp06_setup:
#   PREP=~/runs/ci/exp06_setup/<run>/data JOB_ARGS="--max-steps 50" \
#     infra/cluster/submit.sh exp06_train --array=0 --time=00:40:00        dry run (first 50 updates of tulu)
#   PREP=~/runs/ci/exp06_setup/<run>/data infra/cluster/submit.sh exp06_train --array=0-2
# All tasks of one submission share $CI_OUT; each writes $CI_OUT/<recipe>/ and appends its checkpoints to
# $CI_OUT/checkpoints.txt (the list exp06_eval reads). A Slurm requeue restarts the recipe from scratch into
# $CI_OUT/<recipe>-attempt<k>/. No pip and no downloads here: base, tokenizers and data are already on disk.
# A40 pinned (48 GB: fp32 master weights + AdamW + bf16 scoring copy + 2 x 2048-token micro-batches); if no A40
# starts within 30 min, resubmit with --gres=gpu:l40s:1.
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
set -euo pipefail
IDX="${SLURM_ARRAY_TASK_ID:?submit as an array, e.g. --array=0-2}"
RECIPES=(tulu smoltalk mathcot)
RECIPE="${RECIPES[$IDX]}"
: "${PREP:?PREP=<exp06_setup run>/data}"
export HF_HOME="$HOME/hf_cache" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONTAINER="${CONTAINER:-/home/container/pytorch/pytorch_25.09.sif}"
VENV="$HOME/pydeps/ci-venv"
echo "commit $CI_SHA task $IDX=$RECIPE (job $SLURM_JOB_ID, array $SLURM_ARRAY_JOB_ID) on $(hostname) at $(date -Is), gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), prep=$PREP, out=$CI_OUT"
# bash -c, not -l: the image's login profiles choke on CUDA version strings. No pipes to tail: stream the log.
srun singularity exec --nv "$CONTAINER" bash -c "
  set -euo pipefail
  cd $CI_ROOT
  $VENV/bin/python exp06_matched_drift/train_sft.py selftest --device cuda
  $VENV/bin/python exp06_matched_drift/train_sft.py train --recipe $RECIPE --data $PREP --out $CI_OUT $JOB_ARGS
"
echo "exp06_train task $IDX ($RECIPE) done at $(date -Is)"

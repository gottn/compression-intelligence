#!/usr/bin/env bash
# exp07: retrain exp06's stage 1 for tulu and mathcot (array index -> recipe: 0 tulu, 1 mathcot) with exp06's trainer
# and stage-1 defaults (exp06_matched_drift/train_sft.py: 1,500 updates, AdamW 2e-5, cosine, seed 0, drift hook every
# 20 updates), saving the update-1,500 checkpoint and the 3% Δbpb_fresh mark if reached. Copied from
# exp06_train.sbatch; the differences are the recipes, --marks 3, --label-prefix exp07 and a 7.5 h training wall stop
# (exp06 stage 1 needed 6.5 h) inside the 8 h limit. Submit only via submit.sh, after exp06_setup has re-made the data
# (`JOB_ARGS="--recipes tulu mathcot" infra/cluster/submit.sh exp06_setup`, pinned revisions):
#   PREP=~/runs/ci/exp06_setup/<run>/data infra/cluster/submit.sh exp07_train --array=0-1 [--dependency=afterok:<setup>]
# Both tasks share $CI_OUT; each writes $CI_OUT/<recipe>/ and appends its checkpoints to $CI_OUT/checkpoints.txt,
# which also holds the 18 public rows (copied in from exp07_arc_format/checkpoints_public.tsv before training starts)
# and is the list exp07_eval reads. A requeue restarts the recipe from scratch into $CI_OUT/<recipe>-attempt<k>/.
# No pip and no downloads here: base, tokenizers and data are already on disk.
# A40 pinned, as exp06 (48 GB: fp32 master weights + AdamW + bf16 scoring copy + 2 x 2048-token micro-batches); if no
# A40 starts within 30 min, resubmit with --gres=gpu:l40s:1 (and say so in RESULTS.md: different kernels).
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
set -euo pipefail
IDX="${SLURM_ARRAY_TASK_ID:?submit as an array, e.g. --array=0-1}"
RECIPES=(tulu mathcot)
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
  $VENV/bin/python exp06_matched_drift/train_sft.py train --recipe $RECIPE --data $PREP --out $CI_OUT \
    --marks 3 --label-prefix exp07 --max-wall-s 27000 $JOB_ARGS
"
echo "exp07_train task $IDX ($RECIPE) done at $(date -Is)"

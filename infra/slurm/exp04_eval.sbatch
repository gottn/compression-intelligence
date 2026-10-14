#!/usr/bin/env bash
# exp04: lm-evaluation-harness on one checkpoint per array task (index -> model via cilab.stages.model_at).
# Submit only via submit.sh, after exp04_setup, as an array:
#   infra/cluster/submit.sh exp04_eval --array=4        dry run (Qwen/Qwen2.5-0.5B)
#   infra/cluster/submit.sh exp04_eval --array=0-17%4   all 18 checkpoints, at most 4 at a time
# All tasks of one submission share $CI_OUT; each writes $CI_OUT/results/<model slug>/ only. No pip here: the
# venv is built by exp04_setup. Datasets are read offline from the prefetched HF_HOME cache; the 4 models not in
# the cache from exp03 (SmolLM2-1.7B, -Instruct, Qwen3-1.7B-Base, Qwen3-1.7B) download inside their own task.
# A failed task group does not stop the others; the task exits non-zero at the end so sacct shows it.
# Resumable: a Slurm requeue (same $CI_OUT) skips groups that wrote done.json.
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --open-mode=append
set -euo pipefail
IDX="${SLURM_ARRAY_TASK_ID:?submit as an array, e.g. --array=0-17%4}"
export HF_HOME="$HOME/hf_cache" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
CONTAINER="${CONTAINER:-/home/container/pytorch/pytorch_25.09.sif}"
VENV="$HOME/pydeps/ci-venv"
echo "commit $CI_SHA task $IDX (job $SLURM_JOB_ID, array $SLURM_ARRAY_JOB_ID) on $(hostname) at $(date -Is), gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader), out=$CI_OUT"
mkdir -p "$CI_OUT/results"
# bash -c, not -l: the image's login profiles choke on CUDA version strings. No pipes to tail: stream the log.
srun singularity exec --nv "$CONTAINER" bash -c "
  set -euo pipefail
  cd $CI_ROOT
  $VENV/bin/python exp04_benchmarks/run_eval.py run --index $IDX --out $CI_OUT/results $JOB_ARGS
"
echo "exp04_eval task $IDX done at $(date -Is)"

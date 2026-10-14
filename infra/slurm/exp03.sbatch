#!/usr/bin/env bash
# exp03: fresh-text bpb of every post-training stage (cilab/stages.py: 14 models x 4 corpora, RAW, fp16),
# all on one GPU for same-machine consistency. Submit only via infra/cluster/submit.sh exp03.
# Models download into ~/hf_cache inside the job (~51 GB on first run). Resumable: rows already in
# results/bpb.csv are skipped, so a Slurm requeue (same CI_OUT) continues where it stopped. After a
# manual resubmission, seed the new run with the old results:
#   RESUME_FROM=~/runs/ci/exp03/<old run>/results infra/cluster/submit.sh exp03
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --open-mode=append
set -euo pipefail
export HF_HOME="$HOME/hf_cache" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
CONTAINER="${CONTAINER:-/home/container/pytorch/pytorch_25.09.sif}"
VENV="$HOME/pydeps/ci-venv"
echo "commit $CI_SHA on $(hostname) at $(date -Is), gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), out=$CI_OUT"
mkdir -p "$CI_OUT/data" "$CI_OUT/results"
cp "$CI_ROOT"/exp01_posttraining_bpc/data/*.txt "$CI_ROOT"/exp01_posttraining_bpc/data/*.prov.json "$CI_OUT/data/"
if [ -n "${RESUME_FROM:-}" ] && [ ! -e "$CI_OUT/results/bpb.csv" ]; then
  cp -r "$RESUME_FROM/." "$CI_OUT/results/"; echo "resumed from $RESUME_FROM"
fi
# bash -c, not -l: the image's login profiles choke on CUDA version strings. No pipes to tail: stream the log.
srun singularity exec --nv "$CONTAINER" bash -c "
  set -euo pipefail
  cd $CI_ROOT
  [ -x $VENV/bin/python ] || python -m venv --system-site-packages $VENV
  $VENV/bin/pip install -q -r infra/cluster/requirements-cluster.txt
  $VENV/bin/python -c 'import torch, transformers; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(\"transformers\", transformers.__version__)'
  $VENV/bin/python exp03_staged_drift/score_stages.py tokcheck --data $CI_OUT/data --results $CI_OUT/results || echo 'tokcheck: ids differ within a group, see tokcheck.json'
  $VENV/bin/python exp03_staged_drift/score_stages.py run --device cuda --data $CI_OUT/data --results $CI_OUT/results $JOB_ARGS
"
echo "exp03 done at $(date -Is)"

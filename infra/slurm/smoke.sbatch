#!/usr/bin/env bash
# Smoke test: one GPU, 30 minutes. Proves container + venv + model download + cilab scoring on a compute node.
# Submit only via infra/cluster/submit.sh smoke. v1 (job 1080110) timed out at 20 min: pip took 14 min on a T4
# node and the sanity command also downloaded a 3.5 GB model. v2 scores GPT-2 only; the venv is cached in ~/pydeps.
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
export HF_HOME="$HOME/hf_cache" PYTHONUNBUFFERED=1
CONTAINER="${CONTAINER:-/home/container/pytorch/pytorch_25.09.sif}"
VENV="$HOME/pydeps/ci-venv"
echo "commit $CI_SHA on $(hostname), gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), out=$CI_OUT"
cp -r "$CI_ROOT/exp01_posttraining_bpc/data" "$CI_OUT/data"   # inputs copied; outputs never enter the repo
# bash -c, not -l: the image's login profiles choke on CUDA version strings. No pipes to tail: stream the log.
srun singularity exec --nv "$CONTAINER" bash -c "
  set -euo pipefail
  cd $CI_ROOT
  [ -x $VENV/bin/python ] || python -m venv --system-site-packages $VENV
  $VENV/bin/pip install -q -r infra/cluster/requirements-cluster.txt
  $VENV/bin/python -c 'import torch, transformers; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(\"transformers\", transformers.__version__)'
  $VENV/bin/python -m cilab.smoke --data $CI_OUT/data --corpus enwik8_ctrl --device cuda --out $CI_OUT/smoke.json
"
echo "smoke done"

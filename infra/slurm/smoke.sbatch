#!/usr/bin/env bash
# Smoke test: one GPU, 20 minutes. Proves container + venv + model download + exp01 sanity run on a compute node.
# Submit only via infra/cluster/submit.sh smoke
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
export HF_HOME="$HOME/hf_cache" PYTHONUNBUFFERED=1
CONTAINER="${CONTAINER:-/home/container/pytorch/pytorch_25.09.sif}"
VENV="$HOME/pydeps/ci-venv"
echo "commit $CI_SHA on $(hostname), gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), out=$CI_OUT"
# Work on a copy so the repo checkout stays clean (outputs never go into the repo).
cp -r "$CI_ROOT/exp01_posttraining_bpc" "$CI_OUT/exp01"
srun singularity exec --nv "$CONTAINER" bash -lc "
  set -euo pipefail
  [ -x $VENV/bin/python ] || python -m venv --system-site-packages $VENV
  $VENV/bin/pip install -q -r $CI_ROOT/infra/cluster/requirements-cluster.txt
  $VENV/bin/python -c 'import torch, transformers; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(\"transformers\", transformers.__version__)'
  cd $CI_OUT/exp01 && mkdir -p results logs && $VENV/bin/python score_bpb.py sanity 2>&1 | tail -25
"
echo "smoke done"

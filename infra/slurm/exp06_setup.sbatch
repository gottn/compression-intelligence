#!/usr/bin/env bash
# exp06 setup, CPU only: check the venv has what exp06 needs (datasets, transformers; both came with exp04's
# lm-eval install, so nothing is pip-installed here), then download the three SFT datasets into HF_HOME and write
# the token-matched packed-ready subsets (exp06_matched_drift/prep_data.py) to $CI_OUT/data.
# Run alone, before exp06_train: infra/cluster/submit.sh exp06_setup
# The train jobs read the result through PREP=<this run>/data.
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
set -euo pipefail
export HF_HOME="$HOME/hf_cache" PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=true
export HF_DATASETS_DISABLE_PROGRESS_BARS=1 RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
CONTAINER="${CONTAINER:-/home/container/pytorch/pytorch_25.09.sif}"
VENV="$HOME/pydeps/ci-venv"
echo "commit $CI_SHA on $(hostname) at $(date -Is), out=$CI_OUT"
mkdir -p "$CI_OUT/data"
# bash -c, not -l: the image's login profiles choke on CUDA version strings. No pipes to tail: stream the log.
srun singularity exec "$CONTAINER" bash -c "
  set -euo pipefail
  cd $CI_ROOT
  $VENV/bin/pip freeze > $CI_OUT/pip_freeze.txt
  $VENV/bin/python -c 'import datasets, transformers, tokenizers, torch, numpy; print(\"datasets\", datasets.__version__, \"transformers\", transformers.__version__, \"tokenizers\", tokenizers.__version__, \"torch\", torch.__version__, \"numpy\", numpy.__version__)'
  $VENV/bin/python exp06_matched_drift/prep_data.py --out $CI_OUT/data $JOB_ARGS
"
echo "exp06_setup done at $(date -Is)"

#!/usr/bin/env bash
# exp04 setup, CPU only: install lm-eval into ~/pydeps/ci-venv and prefetch every task's dataset (HF_HOME)
# and ifeval's nltk data (~/nltk_data), so the GPU array tasks never pip-install or download datasets
# concurrently. Run alone, before exp04_eval: infra/cluster/submit.sh exp04_setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
export HF_HOME="$HOME/hf_cache" PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
CONTAINER="${CONTAINER:-/home/container/pytorch/pytorch_25.09.sif}"
VENV="$HOME/pydeps/ci-venv"
echo "commit $CI_SHA on $(hostname) at $(date -Is), out=$CI_OUT"
# bash -c, not -l: the image's login profiles choke on CUDA version strings. No pipes to tail: stream the log.
srun singularity exec "$CONTAINER" bash -c "
  set -euo pipefail
  cd $CI_ROOT
  [ -x $VENV/bin/python ] || python -m venv --system-site-packages $VENV
  $VENV/bin/pip install -q -r infra/cluster/requirements-exp04.txt
  $VENV/bin/pip freeze > $CI_OUT/pip_freeze.txt
  $VENV/bin/python -c 'import lm_eval, transformers, torch, datasets; print(\"lm_eval\", lm_eval.__version__, \"transformers\", transformers.__version__, \"torch\", torch.__version__, \"datasets\", datasets.__version__)'
  $VENV/bin/lm_eval --version || echo 'lm_eval CLI has no --version flag; version printed above'
  $VENV/bin/python exp04_benchmarks/run_eval.py prefetch
"
echo "exp04_setup done at $(date -Is)"

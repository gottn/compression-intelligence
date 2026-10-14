#!/usr/bin/env bash
# The only way jobs start on the cluster. Run ON the front end, from the repo checkout:
#   infra/cluster/submit.sh <job> [extra sbatch args]      e.g. infra/cluster/submit.sh smoke
#   JOB_ARGS="--models Qwen/Qwen2.5-1.5B" infra/cluster/submit.sh score
# Pulls fast-forward only, refuses a dirty checkout, records the commit, writes outputs OUTSIDE the repo.
set -euo pipefail
JOB="${1:?usage: submit.sh <job> [sbatch args]}"; shift || true
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SCRIPT="infra/slurm/$JOB.sbatch"
git pull --ff-only --quiet
[ -f "$SCRIPT" ] || { echo "no such job: $SCRIPT"; ls infra/slurm; exit 1; }
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "refusing to submit: cluster checkout has local modifications (develop on the Mac, pull here)"; git status --short; exit 1
fi
SHA="$(git rev-parse --short HEAD)"
OUT="$HOME/runs/ci/$JOB/$(date +%Y%m%d-%H%M%S)-$SHA"
mkdir -p "$OUT"
git log -1 --format='%H%n%s' > "$OUT/commit.txt"
export CI_OUT="$OUT" CI_SHA="$SHA" CI_ROOT="$ROOT" JOB_ARGS="${JOB_ARGS:-}"
JOBID=$(sbatch --parsable --job-name="ci-$JOB" --output="$OUT/slurm-%j.out" "$@" "$SCRIPT")
echo "submitted $JOB as job $JOBID @ $SHA"
echo "output:  $OUT"
echo "watch:   squeue -j $JOBID ; tail -f $OUT/slurm-$JOBID.out"

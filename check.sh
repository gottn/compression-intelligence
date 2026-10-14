#!/usr/bin/env bash
# Gate: lint, tests, hook installed. Run before every commit.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "no .venv; run: uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r exp01_posttraining_bpc/requirements.txt ruff pytest"; exit 1; }
$PY -m ruff --version >/dev/null 2>&1 || VIRTUAL_ENV=.venv uv pip install -q ruff pytest
$PY -m ruff check .
$PY -m pytest
[ "$(git config core.hooksPath)" = ".githooks" ] || { echo "run: git config core.hooksPath .githooks"; exit 1; }
echo "check.sh: ok"

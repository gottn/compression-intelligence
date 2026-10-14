#!/usr/bin/env bash
# One-time cluster setup. Run ON the AAU front end (file management only; no compute):
#   scp infra/cluster/setup.sh aau:~/ && ssh aau bash setup.sh
# 1. read-only deploy key for the private repo (add it on GitHub: gh repo deploy-key add),
# 2. clone to ~/projects/compression-intelligence,
# 3. run/cache directories. Python envs are built inside jobs, on compute nodes.
set -euo pipefail
REPO_SSH="git@github-ci:gottn/compression-intelligence.git"
DEST="$HOME/projects/compression-intelligence"
KEY="$HOME/.ssh/ci_deploy_key"
mkdir -p "$HOME/.ssh" "$HOME/projects" "$HOME/runs/ci" "$HOME/hf_cache" "$HOME/pydeps"
chmod 700 "$HOME/.ssh"
[ -f "$KEY" ] || ssh-keygen -t ed25519 -N "" -C "compression-intelligence-cluster-readonly" -f "$KEY" -q
grep -q "Host github-ci" "$HOME/.ssh/config" 2>/dev/null || cat >> "$HOME/.ssh/config" <<CFG

# compression-intelligence: read-only deploy key for gottn/compression-intelligence
Host github-ci
  HostName github.com
  User git
  IdentityFile $KEY
  IdentitiesOnly yes
CFG
chmod 600 "$HOME/.ssh/config"
echo "=== deploy key (add as READ-ONLY on GitHub) ==="; cat "$KEY.pub"
[ -d "$DEST/.git" ] || GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new" git clone -q "$REPO_SSH" "$DEST"
echo "ready: $DEST"

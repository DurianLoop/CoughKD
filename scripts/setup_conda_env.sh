#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-coughkd}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
  CONDA_BIN="$HOME/anaconda3/bin/conda"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  CONDA_BIN="$HOME/miniconda3/bin/conda"
else
  echo "conda not found. Install Miniconda/Anaconda or add conda to PATH." >&2
  exit 1
fi

if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_BIN" env update -n "$ENV_NAME" -f environment.yml --prune
else
  "$CONDA_BIN" env create -n "$ENV_NAME" -f environment.yml
fi

echo "Conda environment ready: $ENV_NAME"
echo "Validate with:"
echo "  $CONDA_BIN run -n $ENV_NAME scripts/validate_project.sh"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python python3.12 .venv
fi
uv pip sync --python .venv/bin/python requirements.lock
.venv/bin/python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'

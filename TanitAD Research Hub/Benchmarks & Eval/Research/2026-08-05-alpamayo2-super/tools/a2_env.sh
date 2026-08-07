#!/bin/bash
# Build the Alpamayo 2 Super env. The package pins python==3.12.* and the pod has
# 3.11, so uv provisions its own interpreter. flash-attn builds from source.
set -x
export HF_HOME=/workspace/hf-cache
export UV_CACHE_DIR=/workspace/uv-cache
export UV_PROJECT_ENVIRONMENT=/workspace/a2venv
export MPLCONFIGDIR=/tmp/a2_mpl
export MAX_JOBS=8
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/alpamayo2_repo
uv python install 3.12
uv sync --locked 2>&1 | tail -25
# bitsandbytes is NOT an Alpamayo dependency -- it is ours, for the 4-bit load
uv pip install bitsandbytes 2>&1 | tail -3
"$UV_PROJECT_ENVIRONMENT/bin/python" -c "
import torch, transformers, bitsandbytes as bnb, sys
print('python', sys.version.split()[0])
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('bitsandbytes', bnb.__version__)
try:
    import flash_attn; print('flash_attn', flash_attn.__version__)
except Exception as e: print('flash_attn MISSING:', type(e).__name__)
import alpamayo2_super; print('alpamayo2_super OK')
"
echo "A2_ENV_DONE"

#!/bin/bash
# Remote wrapper: run the Gate-1 decoder fine-tune (GPU).
set -uo pipefail
cd /workspace/alpa-invest/alpasim
export PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts
exec .venv/bin/python /workspace/gate1_finetune.py "$@"

#!/bin/bash
# Remote wrapper: run the Gate-1 extractor with the alpasim venv + tanitad stack.
set -uo pipefail
cd /workspace/alpa-invest/alpasim
export PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts
export CUDA_VISIBLE_DEVICES=""
exec .venv/bin/python /workspace/gate1_extract.py "$@"

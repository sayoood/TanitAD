#!/bin/bash
# Probe 2b driver (CPU only), run AFTER run_repro.sh has finished (so the control C has all cores):
#   D  native256 fp32 12 frames (the SAME 12 steps as the bf16 control C) -> C vs D = the pure bf16-vs-fp32 effect on the PRODUCTION route's own frames
#   E  native256 fp32 101 frames (every frame of both clips) -> a CPU-fp32 reference cache of the production route for the two clips (pool4x10 fp8), for the amendment's CPU identity control
set -e
HERE="$(cygpath -m "$(cd "$(dirname "$0")" && pwd)")"
EXPORT="$(cygpath -m "$(cd "$(dirname "$0")/../../../../../.." && pwd)")"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="$EXPORT/stack;$EXPORT/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
cd "$HERE"
$PY -I repro_route.py --variant native256 --dtype fp32 --frames 12 --threads 24 --out "$HERE/repro_native256_fp32_12" > repro_native256_fp32_12.out 2>&1
$PY -I repro_route.py --variant native256 --dtype fp32 --frames 101 --threads 24 --out "$HERE/repro_native256_fp32_full" > repro_native256_fp32_full.out 2>&1
echo RUN-REPRO2-DONE

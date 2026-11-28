#!/bin/bash
# Probe 2 driver (CPU only). A and B run concurrently (12 threads each), then the bf16 control C (24 threads) re-does the identity control's ROUTE_rebuilt_native256 train cases (12 frames/clip).
#   A  native256 fp32 21 frames  -> route-only gap vs test_cpu_fp32 (device-matched), device+route gap vs test_runscript
#   B  local416  fp32 21 frames  -> does the LOCAL rebuild of the 416 route reproduce Thor's 416 route (device-matched)?
#   C  native256 bf16 12 frames  -> must reproduce identity_control.json ROUTE_rebuilt_native256 (0.0793 / 0.0896)
set -e
HERE="$(cygpath -m "$(cd "$(dirname "$0")" && pwd)")"
EXPORT="$(cygpath -m "$(cd "$(dirname "$0")/../../../../../.." && pwd)")"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="$EXPORT/stack;$EXPORT/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
cd "$HERE"
$PY -I repro_route.py --variant native256 --dtype fp32 --frames 21 --threads 12 --out "$HERE/repro_native256_fp32" > repro_native256_fp32.out 2>&1 &
PA=$!
$PY -I repro_route.py --variant local416 --dtype fp32 --frames 21 --threads 12 --out "$HERE/repro_local416_fp32" > repro_local416_fp32.out 2>&1 &
PB=$!
wait $PA $PB
$PY -I repro_route.py --variant native256 --dtype bf16 --frames 12 --threads 24 --out "$HERE/repro_native256_bf16_control" > repro_native256_bf16_control.out 2>&1
echo RUN-REPRO-DONE

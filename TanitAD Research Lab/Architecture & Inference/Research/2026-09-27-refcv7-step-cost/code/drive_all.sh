#!/bin/bash
# Sequential driver: CPU warm-cache loader bench, then the GPU ladder (each job gated by run_ladder.py).
C="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-step-cost/code"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8
cd "$C" || exit 2
"$PY" run_ladder.py --plan plan_cpu_loader_warm.json --out-root C:/lgt/rc7cost_runs/loader --cpu-only \
  --need-gb 4 --min-free-gb 6 --timeout-s 2400 --gpu-wait-s 14400 >> C:/lgt/rc7cost_runs_loaderwarm_ladder.log 2>&1
"$PY" run_ladder.py --plan plan_gpu.json --out-root C:/lgt/rc7cost_runs/gpu \
  --need-gb 8 --min-free-gb 6 --timeout-s 2400 --gpu-wait-s 21600 >> C:/lgt/rc7cost_runs_gpu_ladder.log 2>&1
echo "DRIVE_DONE $(date +%FT%T)" >> C:/lgt/rc7cost_runs_gpu_ladder.log

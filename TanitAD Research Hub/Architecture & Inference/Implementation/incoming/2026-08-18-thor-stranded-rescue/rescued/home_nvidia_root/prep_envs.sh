#!/bin/bash
# TanitAD Thor env prep (PI rule: TWO venvs, never mixed).
#   tanitad-edge  = use case 1: optimized inference, open-loop, closed-loop (AlpaSim first)
#   tanitad-train = use case 2: training
# EDGE INSTALLS FIRST — the PI's near-term use is eval/inference, and two concurrent
# multi-GB pip downloads over WiFi would just halve each other.
# JetPack 7 / L4T R38 uses the standard SBSA CUDA stack -> official cu130 aarch64 wheels
# are the first choice; the Jetson AI Lab index (jp7) is the fallback. NEVER the x86 wheel.
set -u
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
for V in tanitad-edge tanitad-train; do
  P=$HOME/venvs/$V/bin/pip
  PY=$HOME/venvs/$V/bin/python
  log "=== $V: torch ==="
  $P install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -2
  if ! $PY -c 'import torch' 2>/dev/null; then
    log "$V: cu130 index failed -> jetson-ai-lab jp7 fallback"
    $P install --no-cache-dir torch --index-url https://pypi.jetson-ai-lab.dev/jp7/cu130 2>&1 | tail -2
  fi
  $P install --no-cache-dir numpy 2>&1 | tail -1
  log "=== $V: verify ==="
  $PY -c 'import torch; print("torch", torch.__version__, "cuda_available", torch.cuda.is_available()); import torch as t; x=t.randn(256,256,device="cuda") if t.cuda.is_available() else None; print("matmul_ok" if x is not None and float((x@x).sum())==float((x@x).sum()) else "CPU_ONLY")' 2>&1 | tail -2
done
log '=== edge extras (eval/video deps) ==='
$HOME/venvs/tanitad-edge/bin/pip install --no-cache-dir opencv-python-headless imageio pillow 2>&1 | tail -1
log DONE

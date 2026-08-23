#!/bin/bash
# TensorRT + ONNX toolchain for the EDGE venv (use case 1 per the PI's two-venv rule).
#
# WHY: this unblocks the Production & Optimization stream's #1 latency item, which their own
# backlog recorded as "toolchain-blocked on the dev box; run when a pod is idle or tensorrt +
# onnxruntime-gpu land". Thor is where it lands. Their ONNX IR is already parity-clean at
# opset 17/18 (max|dz| 8.8e-6, no unexportable ops), so the export half of the job is done.
#
# NOTE on TensorRT + venv: the JetPack python bindings are SYSTEM packages under
# /usr/lib/python3*/dist-packages. There is no aarch64 TRT pip wheel matching an L4T runtime,
# so the edge venv reaches them via PYTHONPATH at call time rather than a pip install.
set -u
log() { echo "[$(date -u +%FT%TZ)] $*"; }

log 'apt: TensorRT runtime + dev + python bindings'
printf 'nvidia\n' | sudo -S -p '' apt-get install -y -q \
  tensorrt libnvinfer-bin python3-libnvinfer python3-libnvinfer-dev 2>&1 | tail -3

log 'verify apt side by STATE, not exit code'
ls /usr/src/tensorrt/bin/trtexec 2>/dev/null && echo TRTEXEC_PRESENT
dpkg -l 2>/dev/null | grep -aE 'tensorrt|libnvinfer' | awk '{print $1, $2, $3}' | head -8

log 'pip: onnx + onnxruntime into tanitad-edge'
"$HOME/venvs/tanitad-edge/bin/pip" install -q onnx 2>&1 | tail -1
"$HOME/venvs/tanitad-edge/bin/pip" install -q onnxruntime-gpu 2>&1 | tail -1 || \
  "$HOME/venvs/tanitad-edge/bin/pip" install -q onnxruntime 2>&1 | tail -1

log 'verify python side'
"$HOME/venvs/tanitad-edge/bin/python" -c 'import onnx; print("onnx", onnx.__version__)' 2>&1 | tail -1
"$HOME/venvs/tanitad-edge/bin/python" -c 'import onnxruntime as o; print("ort", o.__version__, o.get_available_providers())' 2>&1 | tail -1
for P in /usr/lib/python3.12/dist-packages /usr/lib/python3/dist-packages; do
  PYTHONPATH=$P "$HOME/venvs/tanitad-edge/bin/python" -c 'import tensorrt as t; print("TRT", t.__version__)' 2>&1 | tail -1
done

log CHAIN_COMPLETE

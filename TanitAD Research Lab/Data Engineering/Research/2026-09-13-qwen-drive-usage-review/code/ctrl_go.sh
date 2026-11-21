#!/bin/bash
# Discriminating control on Thor: THEIR demo, THEIR runner, THEIR visualizer, then
# the same demo degraded one property at a time. GPU must be idle (checked below).
set -o pipefail
export CPATH=/home/nvidia/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12:/home/nvidia/venvs/tanitad-edge/lib/python3.12/site-packages/nvidia/cu13/include
export PATH=/home/nvidia/venvs/tanitad-edge/bin:$PATH
export PYTHONPATH=/home/nvidia/qwendrive/qwen-drive/src
export OMP_NUM_THREADS=6
Q=/home/nvidia/qwendrive
C=$Q/ctrl
PY=/home/nvidia/venvs/tanitad-edge/bin/python
: > $C/status.txt
BUSY=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -ci python)
if [ "${BUSY:-0}" -gt 0 ]; then echo "ZZGPU-BUSY-REFUSEDZZ" >> $C/status.txt; exit 4; fi
cd $Q/qwen-drive
$PY $C/ctrl_make_variants.py data/demo/perception $C > $C/variants.log 2>&1
echo "VARIANTS=$?" >> $C/status.txt
$PY - > $C/ground.log 2>&1 <<'EOF'
import numpy as np, glob, json, os
for fd in sorted(glob.glob("data/demo/perception/*")):
    t = json.load(open(fd + "/frame.json"))["dataset_type"]
    L = np.load(fd + "/lidar.npy"); r = np.hypot(L[:, 0], L[:, 1]); m = (r > 3) & (r < 20)
    h, e = np.histogram(L[m, 2], bins=np.arange(-4, 4.001, 0.05))
    g = np.load(fd + "/gt.npz")["occ"].reshape(200, 200, 16)
    k = np.where(g == 7)[2]
    print(fd[-16:], t, "lidar_ground_z_mode", round(float(e[np.argmax(h)] + 0.025), 3),
          "driveable_zidx_hist", np.bincount(k, minlength=16).tolist() if len(k) else None)
EOF
for V in A0 A0r A1 A2 A3; do
  $PY scripts/run_perception.py --vlm $Q/Qwen-Drive-1.0-4B --model $Q/Qwen-Drive-1.0-4B/perception \
      --frames $C/$V --output $C/${V}_out --attn-implementation sdpa > $C/run_$V.log 2>&1
  echo "RUN_$V=$? npz=$(ls $C/${V}_out/*.npz 2>/dev/null | wc -l)" >> $C/status.txt
  $PY scripts/visualize_perception.py --frames $C/$V --predictions $C/${V}_out --output $C/${V}_vis > $C/vis_$V.log 2>&1
  echo "VIS_$V=$? png=$(ls $C/${V}_vis/*.png 2>/dev/null | wc -l)" >> $C/status.txt
done
$PY $C/ctrl_metrics.py $C > $C/metrics.log 2>&1
echo "METRICS=$?" >> $C/status.txt
echo "ZZALLDONEZZ" >> $C/status.txt

#!/bin/bash
# v2 packing on Thor: THEIR run_perception.py, THEIR visualize_perception.py, unmodified.
set -o pipefail
export CPATH=/home/nvidia/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12:/home/nvidia/venvs/tanitad-edge/lib/python3.12/site-packages/nvidia/cu13/include
export PATH=/home/nvidia/venvs/tanitad-edge/bin:$PATH
export PYTHONPATH=/home/nvidia/qwendrive/qwen-drive/src
export OMP_NUM_THREADS=6
Q=/home/nvidia/qwendrive
V=$Q/v2
PY=/home/nvidia/venvs/tanitad-edge/bin/python
echo "---- $(date +%H:%M:%S) sets: $*" >> $V/status.txt
BUSY=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -ci python)
if [ "${BUSY:-0}" -gt 0 ]; then echo "ZZGPU-BUSY-REFUSEDZZ" >> $V/status.txt; exit 4; fi
cd $Q/qwen-drive
for S in "$@"; do
  [ -d $V/$S ] || { echo "MISSING_$S" >> $V/status.txt; continue; }
  O=$V/out_${S#frames_}
  [ "$S" = "frames_legacy" ] && O=$V/out_legacy
  t0=$(date +%s)
  $PY scripts/run_perception.py --vlm $Q/Qwen-Drive-1.0-4B --model $Q/Qwen-Drive-1.0-4B/perception \
      --frames $V/$S --output $O --attn-implementation sdpa > $V/run_$S.log 2>&1
  echo "RUN_$S=$? npz=$(ls $O/*.npz 2>/dev/null | wc -l) dirs=$(ls -d $V/$S/*/ | wc -l) s=$(( $(date +%s) - t0 ))" >> $V/status.txt
  t0=$(date +%s)
  $PY scripts/visualize_perception.py --frames $V/$S --predictions $O --output ${O}_vis > $V/vis_$S.log 2>&1
  echo "VIS_$S=$? png=$(ls ${O}_vis/*.png 2>/dev/null | wc -l) s=$(( $(date +%s) - t0 ))" >> $V/status.txt
done
$PY $V/ours_metrics.py > $V/ours_metrics.log 2>&1
echo "METRICS=$?" >> $V/status.txt
echo "ZZV2-ALLDONEZZ" >> $V/status.txt

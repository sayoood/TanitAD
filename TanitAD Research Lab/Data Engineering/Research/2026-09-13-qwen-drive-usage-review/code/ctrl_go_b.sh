#!/bin/bash
set -o pipefail
export CPATH=/home/nvidia/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12:/home/nvidia/venvs/tanitad-edge/lib/python3.12/site-packages/nvidia/cu13/include
export PATH=/home/nvidia/venvs/tanitad-edge/bin:$PATH
export PYTHONPATH=/home/nvidia/qwendrive/qwen-drive/src
export OMP_NUM_THREADS=6
Q=/home/nvidia/qwendrive; C=$Q/ctrl; PY=/home/nvidia/venvs/tanitad-edge/bin/python
: > $C/status_b.txt
BUSY=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -ci python)
if [ "${BUSY:-0}" -gt 0 ]; then echo "ZZGPU-BUSY-REFUSEDZZ" >> $C/status_b.txt; exit 4; fi
cd $Q/qwen-drive
$PY $C/ctrl_make_variants_b.py data/demo/perception $C > $C/variants_b.log 2>&1; echo "VARIANTS_B=$?" >> $C/status_b.txt
for V in A4 A5 A6; do
  $PY scripts/run_perception.py --vlm $Q/Qwen-Drive-1.0-4B --model $Q/Qwen-Drive-1.0-4B/perception --frames $C/$V --output $C/${V}_out --attn-implementation sdpa > $C/run_$V.log 2>&1
  echo "RUN_$V=$? npz=$(ls $C/${V}_out/*.npz 2>/dev/null | wc -l)" >> $C/status_b.txt
  $PY scripts/visualize_perception.py --frames $C/$V --predictions $C/${V}_out --output $C/${V}_vis > $C/vis_$V.log 2>&1
  echo "VIS_$V=$? png=$(ls $C/${V}_vis/*.png 2>/dev/null | wc -l)" >> $C/status_b.txt
done
sed -i 's/("A0", "A0r", "A1", "A2", "A3")/("A0", "A0r", "A1", "A2", "A3", "A4", "A5", "A6")/' $C/ctrl_metrics.py
$PY $C/ctrl_metrics.py $C > $C/metrics_b.log 2>&1; echo "METRICS_B=$?" >> $C/status_b.txt
echo ZZCTRLB-DONEZZ >> $C/status_b.txt

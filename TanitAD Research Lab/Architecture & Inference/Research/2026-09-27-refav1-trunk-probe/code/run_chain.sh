#!/usr/bin/env bash
# SPEC A2 chain: [idle GPU 3 min] -> pull+encode 600 TRAIN clips (read-only on Thor) -> extract train features (CPU)
# -> relaunch the R1 full-grid check (its own idle gate). Each stage verified by ARTIFACTS, opaque ZZ tokens.
set -u
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
W=C:/Users/Admin/refav1_trainprobe
LBL="D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/s2_labels_v7.2_train.jsonl.gz"
export PYTHONIOENCODING=utf-8
cd "$W" || exit 2
echo "ZZCH-START $(date '+%F %T')ZZ"
# gate on COMPUTE, not utilization: WDDM "utilization" includes the desktop compositor and a GPU screensaver,
# which never drop to idle; another CUDA job shows up as memory above the ~1.7 GB desktop baseline.
idle=0; waited=0
while [ "$idle" -lt 2 ]; do
  g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')
  case "$g" in ''|*[!0-9]*) g=99999 ;; esac
  if [ "$g" -le 2500 ]; then idle=$((idle + 1)); else idle=0; fi
  [ "$waited" -ge 28800 ] && { echo "ZZCH-GATE-TIMEOUTZZ"; exit 5; }
  sleep 60; waited=$((waited + 60))
done
echo "ZZCH-GPU-IDLE waited=${waited}s $(date '+%T')ZZ"
PYTHONPATH="C:/Users/Admin/tipsnap/c36b6ddd/stack" OMP_NUM_THREADS=6 "$PY" pull_encode.py --n 600 --train-labels "$LBL" > pull_encode.log 2>&1
n=$(ls fp8/*.pt 2>/dev/null | wc -l)
grep -q "ZZENC-DONE" pull_encode.log || { echo "ZZCH-ENC-FAIL n=$n (see pull_encode.log)ZZ"; exit 6; }
echo "ZZCH-ENC-OK n=$n $(date '+%T')ZZ"
PYTHONPATH="C:/Users/Admin/tipsnap/b3f7ea6f/stack;C:/Users/Admin/tipsnap/b3f7ea6f/taniteval" OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" \
  "$PY" C:/Users/Admin/refav1_probe/extract.py --spatial --arm-tool C:/Users/Admin/refav1_dev/taniteval/tools/refav1_arm.py \
  --ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json \
  --cache "$W/fp8" --episodes "$W/eps" --stride 8 --out "$W/feat_train_s8" > extract_train.log 2>&1
m=$(ls feat_train_s8/ep*.npz 2>/dev/null | wc -l)
grep -q "ZZEXTRACT-DONE" extract_train.log || { echo "ZZCH-EXTRACT-FAIL m=$m (see extract_train.log)ZZ"; exit 7; }
echo "ZZCH-EXTRACT-OK episodes=$m $(date '+%T')ZZ"
cd C:/Users/Admin/refav1_r1cap && bash run_r1cap.sh > run_r1cap.out 2>&1
echo "ZZCH-R1-EXIT $(tail -1 run_r1cap.out | tr -d '\r') $(date '+%T')ZZ"

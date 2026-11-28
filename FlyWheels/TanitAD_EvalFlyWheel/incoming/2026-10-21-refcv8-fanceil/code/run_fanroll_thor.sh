#!/bin/bash
# run_fanroll_thor.sh <seed> [--os-only] -- D-REFCV8-FANCEIL fan roll on Thor, under the Thor GPU-lock convention.
# Launch: flock -n /home/nvidia/refcv7_post/thor_gpu.lock bash run_fanroll_thor.sh 1 --os-only
# Opaque markers in fanroll_s<seed>.log: ZZFRSTART / ZZFRBUSY / ZZFREND rc=<n>  (never grep the command text).
set -u
SEED=${1:?seed}; OSO=${2:-}
F=/home/nvidia/r8ns/fanceil; O=$F/out/s$SEED; mkdir -p $O
PY=/home/nvidia/venvs/tanitad-train/bin/python
unset REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION
export R8_LAUNCH_TREE=$F/tree R8_EVAL_KIT=/home/nvidia PF_R8_CODE=$F/code/batt/code PF_NF_HOLD=$F/code/navfix
export R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$F/tree/stack:$F/tree/taniteval
LOG=$O/fanroll_s$SEED.log
echo "ZZFRSTART $(date -Is) pid $$" >> $LOG
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
pys=$(ps -u nvidia -o pid=,comm= | awk '$2 ~ /^python/' | wc -l)
echo "precheck compute_apps=$apps other_python=$pys" >> $LOG
if [ "$apps" != "0" ] || [ "$pys" != "0" ]; then echo "ZZFRBUSY" >> $LOG; exit 3; fi
cd $O
$PY -X utf8 $F/code/fanceil/fan_roll_r8_pf.py --pf-code $F/code/pf/code --remap-json $F/code/fanceil/refcv8_eval_remap_thor_fanceil.json \
  --ckpt /home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt --config /home/nvidia/r8ns/ckpt/config.json --seed $SEED \
  --dump-dir $O/dump --windows-json $F/inputs/windows_s1.json --out-json $O/roll_s$SEED.json \
  --fan-meta-json $O/fan_roll_s$SEED.json --device cuda $OSO >> $LOG 2>&1
echo "ZZFREND rc=$? $(date -Is)" >> $LOG

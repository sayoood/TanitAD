#!/bin/bash
# usage: run_diag.sh <tag> <extra-comma-list> <checks>
set -u
TAG=$1; EXTRA=$2; CHECKS=$3
T=/home/nvidia/refcv6_run/d014ff2fd6; W=/home/nvidia/pregate_d014ff2; PY=/home/nvidia/venvs/tanitad-train/bin/python
AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
if [ "$AV" -lt 40 ]; then echo "ABORT available ${AV} GB < 40"; exit 3; fi
export PYTHONPATH=$T/stack:$T/taniteval:$T/stack/scripts PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${PREGATE_OMP:-4} MKL_NUM_THREADS=${PREGATE_OMP:-4} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=-1
export PREGATE_EXTRA_MODEL_INPUTS="$EXTRA"
cd $W/work
$PY - <<PYEOF
import json
import os
c=json.load(open("$W/work/ctx_model.json")); c["out_dir"]="$W/gate_out_diag_$TAG"; o="${PREGATE_REMAP:-}"; c["options"]["eval_remap_overrides"]=(o or None); c["options"]["eval_kit"]=(os.environ.get("PREGATE_KIT") or None); c["options"]["eval_stamps"]=(os.environ.get("PREGATE_STAMPS") or c["options"].get("eval_stamps"))
os.makedirs(c["out_dir"],exist_ok=True)
json.dump(c,open("$W/work/ctx_diag_$TAG.json","w"),indent=1)
PYEOF
LOG=$W/raw/gate_diag_$TAG.log
nice -n 19 $PY $W/code/gate_patched.py check --ctx $W/work/ctx_diag_$TAG.json --checks $CHECKS > $LOG 2>&1 < /dev/null 200>&- &
CPID=$!
echo $CPID > $W/work/pid_diag_$TAG.txt
( while kill -0 $CPID 2>/dev/null; do A=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); if [ "$A" -lt 25 ]; then echo "WATCHDOG_KILL avail=${A}GB" >> $W/raw/watchdog_diag_$TAG.log; kill $CPID; break; fi; echo "$(date -u +%H:%M:%S) avail=${A}GB" > $W/raw/watchdog_diag_${TAG}_last.txt; sleep 5; done ) 200>&- >/dev/null 2>&1 &
wait $CPID
echo "GATE_CHILD_RC=$?" >> $LOG

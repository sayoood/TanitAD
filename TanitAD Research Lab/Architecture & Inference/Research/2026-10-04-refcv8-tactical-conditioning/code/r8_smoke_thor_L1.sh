#!/bin/bash
# refcv8 FULL-SIZE LAUNCH SMOKE on the COMBINED-LANDING tree (L1: P7 X3 N2 + P8 v9 constraints + P9 emission schedule
# + X10 pose sync + (B) enc8 OFF), replacing the pre-P7 smoke (SPEC_WPB_LADDER sec. 7: "the launch smoke re-runs once
# P7-P9 land"). Waits for the WP-B chain's re-run I-W artifact (or the chain gone), then ONE job under the GPU lock
# (<= 40 min): the canonical argv (stack/ops/runs.d/refcv8-wpb-smoke.argv.json of tree_L1) for 30 steps from
# refcv7-50,400, with ONE smoke-only change: --r8-alloc-emit-start 10 (so both emission regimes run inside 30 steps;
# the launch value is 2000). Then r8_smoke_summary.py, then the run's ckpt.pt is REMOVED (Thor disk: R1 stops < 20 GB).
# The ARTIFACT (runs/refcv8-wpb-smoke-L1/smoke_summary.json) is the evidence, never this script's exit code.
set -u
R=/home/nvidia/refcv8_run
W=/home/nvidia/refcv8_wpb
T=$R/tree_L1
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
CHAIN_PID=${1:?WP-B chain pid}
OUT=$R/runs/refcv8-wpb-smoke-L1
mkdir -p $R/logs $R/runs
export PYTHONPATH=$T/stack:$T/taniteval OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
until [ -s $W/arms/I_W.json ] || ! kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 120 200>&-; done
echo "[r8-smoke-L1] $(date -u +%H:%M:%S) I-W artifact present (PASS=$(python3 -c "import json;print(json.load(open('$W/arms/I_W.json'))['PASS'])" 2>/dev/null)) -> queueing"
ARGV=$($PY -c "
import json
a=json.load(open('$T/stack/ops/runs.d/refcv8-wpb-smoke.argv.json'))['argv']
def setf(a,f,v):
    if f in a:
        i=a.index(f); j=i+1
        while j<len(a) and not a[j].startswith('--'): j+=1
        a[i:j]=[f]+v
    else: a+= [f]+v
    return a
for f,v in (('--steps',['30']),('--log-every',['10']),('--grad-share-every',['30']),('--eval-every',['30']),
            ('--eval-batches',['4']),('--save-every',['30']),('--r8-alloc-emit-start',['10']),
            ('--join-defect-masks',['$T/stack/tanitad/configs/refcv8_join_label_defects.json']),('--out',['$OUT'])):
    a=setf(a,f,v)
print(' '.join(a))")
[ -n "$ARGV" ] || { echo "[r8-smoke-L1] could not build the argv -- stopping"; exit 1; }
cd $T
flock $LOCK timeout 2400 $PY stack/scripts/refc_v3_train.py $ARGV >> $R/logs/smoke_L1_train.log 2>&1 < /dev/null 200>&-
rc=$?   # captured BEFORE any $(...): the old line read $? AFTER $(date) and logged "exit 0" for a crashed trainer
echo "[r8-smoke-L1] $(date -u +%H:%M:%S) trainer exit $rc (the artifact decides)"
$PY $R/code/r8_smoke_summary.py --run $OUT --init /home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt \
    > $R/logs/smoke_L1_summary.log 2>&1 < /dev/null
$PY $R/code/p7_smoke_check.py --run $OUT --stack $T/stack > $R/logs/smoke_L1_p7check.log 2>&1 < /dev/null
echo "[r8-smoke-L1] $(date -u +%H:%M:%S) summary $( [ -s $OUT/smoke_summary.json ] && echo WRITTEN || echo MISSING )"
rm -f /home/nvidia/refcv8_run/runs/refcv8-wpb-smoke-L1/ckpt.pt
echo "[r8-smoke-L1] $(date -u +%H:%M:%S) ckpt removed; free $(df -B1G /home | tail -1 | awk '{print $4}') GB"

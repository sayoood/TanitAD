#!/bin/bash
# refcv8 FULL-SIZE SMOKE on Thor (SPEC_WPB_LADDER sec. 7; the MM's item 5). Queued after the WP-B chain's re-run I-W: waits for
# its artifact (or the chain gone), then ONE job under the GPU lock (<= 40 min): the canonical refcv8 smoke argv
# (stack/ops/runs.d/refcv8-wpb-smoke.argv.json) for 30 steps from refcv7-50,400 (--init-from), every seam on,
# --log-every 10 / --grad-share-every 30 (the share is read once, at the last step, so the 10-20 interval is a clean s/step), then r8_smoke_summary.py.
# The ARTIFACT (runs/refcv8-wpb-smoke/smoke_summary.json) is the evidence, never this script's exit code.
set -u
R=/home/nvidia/refcv8_run
W=/home/nvidia/refcv8_wpb
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
CHAIN_PID=${1:?WP-B chain pid}
OUT=$R/runs/refcv8-wpb-smoke
mkdir -p $R/logs $R/runs
export PYTHONPATH=$R/tree/stack:$R/tree/taniteval OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
# the smoke needs the WP-B tree to have passed its identity gate first: wait for the re-run I-W artifact (or the chain
# gone), then compete for the lock like any other WP-B job (one job, <= 40 min)
until [ -s $W/arms/I_W.json ] || ! kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 120; done
echo "[r8-smoke] $(date -u +%H:%M:%S) I-W artifact present (PASS=$(python3 -c "import json;print(json.load(open('$W/arms/I_W.json'))['PASS'])" 2>/dev/null)) -> queueing the smoke"
ARGV=$($PY -c "
import json,sys
a=json.load(open('$R/tree/stack/ops/runs.d/refcv8-wpb-smoke.argv.json'))['argv']
def setf(a,f,v):
    if f in a:
        i=a.index(f); j=i+1
        while j<len(a) and not a[j].startswith('--'): j+=1
        a[i:j]=[f]+v
    else: a+= [f]+v
    return a
for f,v in (('--steps',['30']),('--log-every',['10']),('--grad-share-every',['30']),('--eval-every',['30']),
            ('--eval-batches',['4']),('--save-every',['30']),('--out',['$OUT'])):
    a=setf(a,f,v)
print(' '.join(a))")
[ -n "$ARGV" ] || { echo "[r8-smoke] could not build the argv -- stopping"; exit 1; }
cd $R/tree
flock $LOCK timeout 2400 $PY stack/scripts/refc_v3_train.py $ARGV >> $R/logs/smoke_train.log 2>&1 < /dev/null 200>&-
echo "[r8-smoke] $(date -u +%H:%M:%S) trainer exit $? (the artifact decides)"
$PY $R/code/r8_smoke_summary.py --run $OUT --init /home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt \
    > $R/logs/smoke_summary.log 2>&1 < /dev/null
echo "[r8-smoke] $(date -u +%H:%M:%S) summary $( [ -s $OUT/smoke_summary.json ] && echo WRITTEN || echo MISSING )"

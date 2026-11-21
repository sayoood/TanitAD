#!/bin/bash
# run_pod_read_r.sh -- runs ON THE POD after FINEGRID_R_DONE.txt exists: the reader of card PREREG_V8F_PROOF_ADD43 (finegrid_r_read.py) on the pod's own files. One CPU job, about 1 h ESTIMATED.
# Start (the lead):  cd /workspace/v8proof/finegrid/rk && nohup setsid bash run_pod_read_r.sh > pod_read_r.nohup 2>&1 < /dev/null &
# The folder rk mirrors the fixed paths of the pilot reader (literal folder `F:` inside rk, as for the pilot's pod_read.sh). Put the files named below there first (compare sha256 on both sides):
#   rk/F:/TanitAD-scratch/v8proof/finegrid_r/{finegrid_r_read.py,controls_devbox_r.json}      rk/F:/TanitAD-scratch/v8proof/finegrid/finegrid_read_n1.py (already there for the pilot)
# Inputs : states + head outputs of the 26 arms (eval split) in $ST ; train logs, r1 logs, G2_GRIDS.json, FINEGRID_R_DONE.txt in $RUN ; the pilot's F states in $PILOT (beside only; missing = not read).
# Outputs: $RK/out_read_r/RESULT_FINEGRID_R.json, VERDICT_FINEGRID_R.md ; $RK/read_r.out ; markers READ_R_STARTED.txt (noclobber: a second start is refused) and READ_R_DONE.txt (rc 0 named / 2 control failed / 3 input missing / 8 no memory).
# Waits for 16 GB of the container memory limit free (memory.max minus the anon line of memory.stat), at most 2 h. No process-name test. Nothing is deleted or moved.
set -u
RK=/workspace/v8proof/finegrid/rk
cd "$RK" || exit 9
V='F:/TanitAD-scratch/v8proof'
RUN=${RUN:-/workspace/v8proof/runs/finegrid_r}
ST=${ST:-/workspace/v8proof/finegrid_r/states}
PILOT=${PILOT:-/workspace/v8proof/finegrid/states}
OUTD=${OUTD:-$RK/out_read_r}
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
( set -o noclobber; ts > READ_R_STARTED.txt ) 2>/dev/null || { echo "[$(ts)] REFUSED: READ_R_STARTED.txt exists"; exit 3; }
[ -f "$RUN/FINEGRID_R_DONE.txt" ] || { echo "[$(ts)] REFUSED: no $RUN/FINEGRID_R_DONE.txt"; echo "rc=4 no done marker $(ts)" > READ_R_DONE.txt; exit 4; }
freegb() { local a m; a=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat); m=$(cat /sys/fs/cgroup/memory.max); echo $(( (m - a) / 1073741824 )); }
n=0
while true; do
  f=$(freegb)
  [ "$f" -ge 16 ] && { echo "[$(ts)] container memory free $f GB: starting the reader"; break; }
  n=$((n+1)); [ "$n" -gt 120 ] && { echo "[$(ts)] ZZ POD-READ-R STOP: under 16 GB free for 2 h"; echo "rc=8 no memory $(ts)" > READ_R_DONE.txt; exit 8; }
  sleep 60
done
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
/workspace/venv/bin/python "$V/finegrid_r/finegrid_r_read.py" --in-dir "$ST" --run-dir "$RUN" --g2 "$RUN/G2_GRIDS.json" --controls-json "$V/finegrid_r/controls_devbox_r.json" \
  --done-marker "$RUN/FINEGRID_R_DONE.txt" --pilot-f-dir "$PILOT" --g7-tol 1e-5 --out-dir "$OUTD" > "$RK/read_r.out" 2>&1
rc=$?
echo "rc=$rc $(ts)" > READ_R_DONE.txt
echo "[$(ts)] ZZ POD-READ-R DONE rc=$rc"

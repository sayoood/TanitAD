#!/usr/bin/env bash
# podrun_export_traces.sh -- pod side, CPU only: runs export_traces.py on the banked T1-kin runs of the three forms (RR, RLc, VD5c), both model seeds, with the environment of podread_visrule10k.sh.
# Reads dumps of the run folders under $W/t1 (override the folder with ROLLTRACE_PULL); writes ONE new folder $R/out (refuses if it exists). It deletes, moves and stops nothing.
# DRY=1 prints the command and makes nothing.
# Last line: ZZ PODRUN-ROLLTRACE DONE rc <n>   or   ZZ PODRUN-ROLLTRACE STOP <why>   or (DRY) ZZ PODRUN-ROLLTRACE DRY.
set -u
V=/workspace/v8proof; W=$V/integ10k; P=$V/podread; RD=$P/root/integ10k; CODE=$P/code
R=$V/rolltrace; K=$R/kit; OUT=$R/out
PL=${ROLLTRACE_PULL:-$W/t1}
PY=/workspace/venv/bin/python
stop() { echo "ZZ PODRUN-ROLLTRACE STOP $*"; exit 3; }
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
CMD="cd $RD && $PY $K/export_traces.py --pull $PL --grid-dir $W/longroll --join $HEADWAY_JOIN --code $CODE --out $OUT"
if [ "${DRY:-0}" = 1 ]; then
  echo "DRY: no folder is made, nothing is run"
  echo "output folder: $OUT (refused if it exists)"
  echo "$CMD > $R/export_traces.log"
  echo "ZZ PODRUN-ROLLTRACE DRY"
  exit 0
fi
[ -e "$OUT" ] && stop "ALREADY-EXPORTED $OUT exists"
[ -f "$K/export_traces.py" ] || stop "MISSING-KIT-FILE export_traces.py"
[ -f "$RD/read_longroll10k.py" ] || stop "MISSING-INPUT $RD/read_longroll10k.py"
[ -f "$W/longroll/grid_p30.json" ] || stop "MISSING-INPUT grid_p30.json"
[ -f "$HEADWAY_JOIN" ] || stop "MISSING-INPUT join_eval141.npz"
for r in RQ50KF5P30Hb20 RLcKF5P30Hb20 RQ50KF5P30Hb20vVD5c; do for s in 0 1; do [ -d "$PL/I10${r}_s$s" ] || stop "MISSING-RUN $PL/I10${r}_s$s"; done; done
mm=$(cat /sys/fs/cgroup/memory.max); an=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
if [ "$mm" != max ]; then free_gb=$(( (mm - an) / 1073741824 )); echo "container free ${free_gb} GB"; [ "$free_gb" -ge 8 ] || stop "MEMORY free ${free_gb} GB"; fi
( cd "$RD" && "$PY" "$K/export_traces.py" --selftest --code "$CODE" > "$R/selftest_pod.log" 2>&1 ); st=$?
tail -n 1 "$R/selftest_pod.log" | cut -c1-120
[ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
( cd "$RD" && "$PY" "$K/export_traces.py" --pull "$PL" --grid-dir "$W/longroll" --join "$HEADWAY_JOIN" --code "$CODE" --out "$OUT" > "$R/export_traces.log" 2>&1 ); rc=$?
grep -E '^ZZ EXPORT-TRACES|Traceback|Error|REFUSED|STOP|MISSING|WINDOWS' "$R/export_traces.log" | cut -c1-300 | head -n 12
echo "ZZ PODRUN-ROLLTRACE DONE rc $rc"

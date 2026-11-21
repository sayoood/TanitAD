#!/usr/bin/env bash
# Wait on the polar probe's ARTIFACT (never its exit code), then fan out:
#   - polar exact bootstrap (CPU)
#   - the planner seed floor (GPU)
set -u
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8
RAW=/c/Users/Admin/wpd-probe/raw
POL=$RAW/panel_rep_pol.json
for i in $(seq 1 300); do
  [ -f "$POL" ] && grep -q "^WROTE" "$RAW/probe_rep_pol.log" 2>/dev/null && break
  grep -qE "Traceback|ValueError|CUDA out of memory" "$RAW/probe_rep_pol.log" 2>/dev/null && {
      echo "POLAR_PROBE_FAILED"; tail -20 "$RAW/probe_rep_pol.log"; exit 3; }
  sleep 10
done
[ -f "$POL" ] || { echo "POLAR_ARTIFACT_ABSENT"; exit 4; }
grep -E "^\[tok_|^\[pair\]" "$RAW/probe_rep_pol.log" | tail -20

cd /c/Users/Admin/wpd-probe/code || exit 9
echo "=== launching polar exact bootstrap (CPU) ==="
"$PY" b5_fast_boot.py --bank 'C:\Users\Admin\wpd-probe\bank_rep' --geom pol \
      --n-boot 2000 --arms tok_D0,tok_D0b,tok_D0c,tok_D1,tok_D2 \
      --out 'C:\Users\Admin\wpd-probe\raw\boot_rep_pol.json' \
      > "$RAW/boot_rep_pol.log" 2>&1 &
BOOTPID=$!
echo "POLAR_BOOT_PID=$BOOTPID"

echo "=== launching planner seed floor (GPU) ==="
/c/Users/Admin/wpd-probe/code/b17_planner_seed_floor.sh
echo "B17_RC=$?"
wait $BOOTPID
echo "POLAR_BOOT_RC=$?"
grep -E "^\[draws\]|^\[pair\]|^WROTE" "$RAW/boot_rep_pol.log" | tail -20
echo "B18_ALL_DONE"

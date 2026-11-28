#!/usr/bin/env bash
# podread_lookgap.sh -- pod-side read of card PREREG_V8F_PROOF_ADD108 (LOOKGAP-T0). CPU only (CUDA_VISIBLE_DEVICES=-1), no rollout.
# Stage controls: needs DONE_FIT.txt and no STOP_FIT.txt; refuses if LOOKGAP_CONTROLS.json exists (never reads twice into one file); runs the reader's selftest on the pod first, then
#   read_lookgap.py --stage controls (writes $O/LOOKGAP_CONTROLS.json; the values are in the file and are not printed; only the m<i> PASS|FAIL lines and the reader's last line are).
# Stage names: read_lookgap.py --stage names on that file (refuses unless m0 to m6 are all present and pass), writes LOOKGAP_RESULT.json and the names log read_names.log; then appends
#   the stage A score lines of the two score logs under a heading line.
# It deletes, moves and stops nothing.
# Usage: bash podread_lookgap.sh controls|names
# Last line: ZZ PODREAD-LOOKGAP <stage> DONE rc <n>   or   ZZ PODREAD-LOOKGAP STOP <why>.
set -u
STAGE=${1:-}
ROOT=${LG_ROOT:-/workspace/v8proof/lookgap}
K=${LG_KIT:-$ROOT/kit_read}
OUT=${LG_OUT:-$ROOT/read}
PY=${LG_PY:-/workspace/venv/bin/python}
LAT=${LG_LOOKAT:-/workspace/v8proof/lookat/kit/read_lookat.py}
# LG_EXTRA: extra reader arguments (the tests of the kit point the reader at a tiny tree); empty on the pod
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-LOOKGAP STOP $*"; exit 3; }
case "$STAGE" in controls|names) ;; *) stop "usage: controls|names";; esac
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(tr -d ' \r\n' < "${LG_CG:-/sys/fs/cgroup}/memory.max" 2>/dev/null); an=$(awk '/^anon /{print $2}' "${LG_CG:-/sys/fs/cgroup}/memory.stat" 2>/dev/null)
if [[ "$mm" =~ ^[0-9]+$ ]] && [[ "$an" =~ ^[0-9]+$ ]]; then
  free_gb=$(( (mm - an) / 1073741824 ))
  [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
fi
[ -f "$K/read_lookgap.py" ] || stop "MISSING-KIT-FILE $K/read_lookgap.py"
[ -f "$LAT" ] || stop "MISSING-KIT-FILE $LAT"
mkdir -p "$OUT" || stop "CANNOT-MAKE $OUT"
if [ "$STAGE" = controls ]; then
  [ -f "$ROOT/DONE_FIT.txt" ] || stop "QUEUE-NOT-DONE"
  [ -e "$ROOT/STOP_FIT.txt" ] && stop "QUEUE-STOPPED"
  [ -e "$OUT/LOOKGAP_CONTROLS.json" ] && stop "ALREADY-READ controls"
  [ -f "$K/lead_pytest.out" ] || stop "MISSING-INPUT $K/lead_pytest.out"
  "$PY" "$K/read_lookgap.py" --selftest --lookat-reader "$LAT" > "$OUT/selftest_pod.log" 2>&1; st=$?
  tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120
  [ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
  "$PY" "$K/read_lookgap.py" --stage controls --lookgap-root "$ROOT" --lookat-reader "$LAT" --out "$OUT/LOOKGAP_CONTROLS.json" ${LG_EXTRA:-} > "$OUT/read_controls.log" 2>&1; rc=$?
  grep -E '^m[0-6] (PASS|FAIL)$|^ZZ READ-LOOKGAP|Traceback|REFUSED' "$OUT/read_controls.log" | cut -c1-200 | head -n 20
  echo "ZZ PODREAD-LOOKGAP controls DONE rc $rc $(ts)"
else
  [ -f "$OUT/LOOKGAP_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
  [ -e "$OUT/LOOKGAP_RESULT.json" ] && stop "ALREADY-READ names"
  "$PY" "$K/read_lookgap.py" --stage names --lookat-reader "$LAT" --controls "$OUT/LOOKGAP_CONTROLS.json" --out "$OUT/LOOKGAP_RESULT.json" --log "$OUT/read_names.log" ${LG_EXTRA:-} > "$OUT/read_names.out" 2>&1; rc=$?
  if [ "$rc" = 0 ] || [ "$rc" = 5 ]; then
    { echo "stage A scores (frames with a label lead and a full history; not the row set of this card):"; grep -h '^SCORE ' "$ROOT/queue/logs/score_train.log" "$ROOT/queue/logs/score_eval.log" 2>/dev/null; } >> "$OUT/read_names.log"
  fi
  tail -n 1 "$OUT/read_names.out" | cut -c1-200
  echo "ZZ PODREAD-LOOKGAP names DONE rc $rc $(ts)"
fi

#!/usr/bin/env bash
# podread_lookat.sh -- pod-side read of card PREREG_V8F_PROOF_ADD107 (LOOKAT-T0). CPU only (CUDA_VISIBLE_DEVICES=-1), no rollout.
# Stage controls: needs DONE_LOOKAT.txt, no STOP_LOOKAT.txt, kit/lead_pytest.out (k6) and IDENTITY_K5.json; runs the reader's selftest, then read_lookat.py --stage controls (writes $O/read/LOOKAT_CONTROLS.json; the values
#   are in the file and are not printed; the stage prints only the control lines).
# Stage names: read_lookat.py --stage names on that file (refuses unless k0 to k6 are all present and pass), writes $O/read/LOOKAT_RESULT.json and the names log $O/read/read_names.log.
# Never reads twice into one file. It deletes, moves and stops nothing.
# Usage: bash podread_lookat.sh controls|names
# Last line: ZZ PODREAD-LOOKAT <stage> DONE rc <n>   or   ZZ PODREAD-LOOKAT STOP <why>.
set -u
STAGE=${1:-}
V=/workspace/v8proof
R1=${LK_RUN1:-$V/cropread/run1}; O=${LK_ROOT:-$V/lookat}; K=${LK_KIT:-$O/kit}; OUT=$O/read
BOX=${LK_BOX:-$V/cropread/enc_out/box}; EJ=${LK_EVAL_JOIN:-$V/hierrep/pins/join_eval141.npz}
PY=${LK_PY:-/workspace/venv/bin/python}
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-LOOKAT STOP $*"; exit 3; }
case "$STAGE" in controls|names) ;; *) stop "usage: controls|names";; esac
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(tr -d ' \r\n' < "${LK_CG:-/sys/fs/cgroup}/memory.max" 2>/dev/null); an=$(awk '/^anon /{print $2}' "${LK_CG:-/sys/fs/cgroup}/memory.stat" 2>/dev/null)
if [[ "$mm" =~ ^[0-9]+$ ]] && [[ "$an" =~ ^[0-9]+$ ]]; then free_gb=$(( (mm - an) / 1073741824 )); echo "[$(ts)] container free ${free_gb} GB"; [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"; fi
[ -f "$K/read_lookat.py" ] || stop "MISSING-KIT-FILE $K/read_lookat.py"
mkdir -p "$OUT" || stop "CANNOT-MAKE $OUT"
if [ "$STAGE" = controls ]; then
  [ -f "$O/DONE_LOOKAT.txt" ] || stop "QUEUE-NOT-DONE"
  [ -e "$O/STOP_LOOKAT.txt" ] && stop "QUEUE-STOPPED"
  [ -e "$OUT/LOOKAT_CONTROLS.json" ] && stop "ALREADY-READ controls"
  for f in "$K/lead_pytest.out" "$O/IDENTITY_K5.json" "$EJ" "$R1/lists/train_final.txt" "$R1/lists/eval141.txt" "$R1/fits/D_H5_s0/FIT_RECORD.json"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
  ( sha256sum "$K/read_lookat.py" "$K/locator_fit.py" "$K/lead_pytest.out" "$EJ" "$O/IDENTITY_K5.json" "$O"/fits/LOC_s*/FIT_RECORD.json ) > "$OUT/READER_HASHES.txt"
  "$PY" "$K/read_lookat.py" --selftest > "$OUT/selftest_pod.log" 2>&1; st=$?
  tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120
  [ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
  "$PY" "$K/read_lookat.py" --stage controls --root "$R1" --out-root "$O" --box-dir "$BOX" --eval-join "$EJ" --out "$OUT/LOOKAT_CONTROLS.json" > "$OUT/read_controls.log" 2>&1; rc=$?
  grep -E '^k[0-6] (PASS|FAIL)|^k4 note|^ZZ READ-LOOKAT|Traceback|Error|REFUSED|STOP' "$OUT/read_controls.log" | cut -c1-300 | head -n 20
  echo "ZZ PODREAD-LOOKAT controls DONE rc $rc $(ts)"
else
  [ -f "$OUT/LOOKAT_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
  [ -e "$OUT/LOOKAT_RESULT.json" ] && stop "ALREADY-READ names"
  "$PY" "$K/read_lookat.py" --stage names --controls "$OUT/LOOKAT_CONTROLS.json" --out "$OUT/LOOKAT_RESULT.json" --log "$OUT/read_names.log" > "$OUT/read_names.out" 2>&1; rc=$?
  tail -n 1 "$OUT/read_names.out" | cut -c1-200
  echo "ZZ PODREAD-LOOKAT names DONE rc $rc $(ts)"
fi

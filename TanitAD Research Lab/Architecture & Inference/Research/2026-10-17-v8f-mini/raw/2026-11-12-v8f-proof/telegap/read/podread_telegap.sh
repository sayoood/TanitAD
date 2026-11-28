#!/usr/bin/env bash
# podread_telegap.sh -- pod-side read of card PREREG_V8F_PROOF_ADD122 (TELEGAP-T0). CPU only (CUDA_VISIBLE_DEVICES=-1), no rollout.
# Stage controls: needs TELEGAP_FIT_DONE.txt and no stop marker; refuses if TELEGAP_CONTROLS.json exists (never reads twice into one file); runs the reader's selftest on the pod first, then
#   read_telegap.py --stage controls with the arguments of $K/read_args.txt (one token per line: every --dump NAME=PATH and the input paths; see TELEGAP_READ_DEFS.md).
#   Only the t<i> PASS|FAIL lines and the reader's last line are printed; every value goes to $OUT/TELEGAP_CONTROLS.json.
# Stage names: read_telegap.py --stage names on that file (refuses unless t0 to t6 are all present and pass); writes TELEGAP_RESULT.json and read_names.log; prints the reader's last line only.
# It deletes, moves and stops nothing.
# Usage: bash podread_telegap.sh controls|names
# Last line: ZZ PODREAD-TELEGAP <stage> DONE rc <n> <time>   or   ZZ PODREAD-TELEGAP STOP <why>.
set -u
STAGE=${1:-}
ROOT=${TG_ROOT:-/workspace/v8proof/telegap}
K=${TG_KIT:-$ROOT/kit/read}
OUT=${TG_OUT:-$ROOT/read}
PY=${TG_PY:-/workspace/venv/bin/python}
LGR=${TG_LOOKGAP:-/workspace/v8proof/lookgap/kit_read/read_lookgap.py}
LAT=${TG_LOOKAT:-/workspace/v8proof/lookat/kit/read_lookat.py}
ARGS_FILE=${TG_ARGS:-$K/read_args.txt}
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-TELEGAP STOP $*"; exit 3; }
case "$STAGE" in controls|names) ;; *) stop "usage: controls|names";; esac
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
mm=$(tr -d ' \r\n' < "${TG_CG:-/sys/fs/cgroup}/memory.max" 2>/dev/null); an=$(awk '/^anon /{print $2}' "${TG_CG:-/sys/fs/cgroup}/memory.stat" 2>/dev/null)
if [[ "$mm" =~ ^[0-9]+$ ]] && [[ "$an" =~ ^[0-9]+$ ]]; then
  free_gb=$(( (mm - an) / 1073741824 ))
  [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
fi
for f in "$K/read_telegap.py" "$LGR" "$LAT"; do
  [ -f "$f" ] || stop "MISSING-KIT-FILE $f"
done
mkdir -p "$OUT" || stop "CANNOT-MAKE $OUT"
cd "$OUT" || stop "CANNOT-ENTER $OUT"
if [ "$STAGE" = controls ]; then
  [ -f "$ROOT/TELEGAP_FIT_DONE.txt" ] || stop "QUEUE-NOT-DONE"
  { [ -e "$ROOT/TELEGAP_FIT_STOP.txt" ] || [ -e "$ROOT/TELEGAP_ENC_STOP.txt" ]; } && stop "QUEUE-STOPPED"
  [ -e "$OUT/TELEGAP_CONTROLS.json" ] && stop "ALREADY-READ controls"
  [ -f "$ARGS_FILE" ] || stop "MISSING-INPUT $ARGS_FILE"
  mapfile -t RARGS < <(tr -d '\r' < "$ARGS_FILE" | sed '/^$/d')
  "$PY" "$K/read_telegap.py" --selftest --lookgap-reader "$LGR" --lookat-reader "$LAT" > "$OUT/selftest_pod.log" 2>&1; st=$?
  tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120 | sed '/hf_/d'
  [ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
  "$PY" "$K/read_telegap.py" --stage controls --lookgap-reader "$LGR" --lookat-reader "$LAT" --out "$OUT/TELEGAP_CONTROLS.json" "${RARGS[@]}" > "$OUT/read_controls.log" 2>&1; rc=$?
  grep -E '^t[0-6] (PASS|FAIL)$' "$OUT/read_controls.log" | head -n 7
  tail -n 1 "$OUT/read_controls.log" | cut -c1-200 | sed '/hf_/d'
  echo "ZZ PODREAD-TELEGAP controls DONE rc $rc $(ts)"
else
  [ -f "$OUT/TELEGAP_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
  [ -e "$OUT/TELEGAP_RESULT.json" ] && stop "ALREADY-READ names"
  "$PY" "$K/read_telegap.py" --stage names --lookgap-reader "$LGR" --lookat-reader "$LAT" --controls "$OUT/TELEGAP_CONTROLS.json" --out "$OUT/TELEGAP_RESULT.json" --log "$OUT/read_names.log" > "$OUT/read_names.out" 2>&1; rc=$?
  tail -n 1 "$OUT/read_names.out" | cut -c1-200 | sed '/hf_/d'
  echo "ZZ PODREAD-TELEGAP names DONE rc $rc $(ts)"
fi

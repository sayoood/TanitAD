#!/usr/bin/env bash
# podread_boxsize.sh -- pod-side read of card PREREG_V8F_PROOF_ADD124 (BOXSIZE-T0). CPU only (CUDA_VISIBLE_DEVICES=-1), no rollout.
# Needs BOXSIZE_DONE.txt and no BOXSIZE_STOP.txt in the card root. Never reads twice into one file. Runs the reader's selftest on the pod first.
# Stage controls: read_boxsize.py --stage controls with the arguments of $K/read_args.txt (one token per line); prints only the s<i> PASS|FAIL lines and the reader's last line;
#   every value goes to $OUT/BOXSIZE_CONTROLS.json.
# Stage names: read_boxsize.py --stage names on that file with the same arguments (refuses unless s0 to s6 all pass and the inputs are unchanged); writes BOXSIZE_RESULT.json and read_names.log.
# It deletes, moves and stops nothing.
# Usage: bash podread_boxsize.sh controls|names
# Last line: ZZ PODREAD-BOXSIZE <stage> DONE rc <n> <time>   or   ZZ PODREAD-BOXSIZE STOP <why>.
set -u
STAGE=${1:-}
ROOT=${BS_ROOT:-/workspace/v8proof/boxsize}
K=${BS_KIT:-$ROOT/kit/read}
OUT=${BS_OUT:-$ROOT/read}
PY=${BS_PY:-/workspace/venv/bin/python}
ARGS_FILE=${BS_ARGS:-$K/read_args.txt}
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ PODREAD-BOXSIZE STOP $*"; exit 3; }
case "$STAGE" in controls|names) ;; *) stop "usage: controls|names";; esac
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
CG=${BS_CG:-/sys/fs/cgroup}; mm=""; an=""
[ -r "$CG/memory.max" ] && mm=$(tr -d ' \r\n' < "$CG/memory.max"); [ -r "$CG/memory.stat" ] && an=$(awk '/^anon /{print $2}' "$CG/memory.stat")
if [[ "$mm" =~ ^[0-9]+$ ]] && [[ "$an" =~ ^[0-9]+$ ]]; then
  free_gb=$(( (mm - an) / 1073741824 ))
  [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
fi
[ -f "$K/read_boxsize.py" ] || stop "MISSING-KIT-FILE $K/read_boxsize.py"
[ -f "$ROOT/BOXSIZE_DONE.txt" ] || stop "QUEUE-NOT-DONE"
[ -e "$ROOT/BOXSIZE_STOP.txt" ] && stop "QUEUE-STOPPED"
[ -f "$ARGS_FILE" ] || stop "MISSING-INPUT $ARGS_FILE"
mkdir -p "$OUT" || stop "CANNOT-MAKE $OUT"
cd "$OUT" || stop "CANNOT-ENTER $OUT"
mapfile -t RARGS < <(tr -d '\r' < "$ARGS_FILE" | sed '/^$/d')
if [ "$STAGE" = controls ]; then
  [ -e "$OUT/BOXSIZE_CONTROLS.json" ] && stop "ALREADY-READ controls"
  "$PY" "$K/read_boxsize.py" --selftest > "$OUT/selftest_pod.log" 2>&1; st=$?
  tail -n 1 "$OUT/selftest_pod.log" | cut -c1-120 | sed '/hf_/d'
  [ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
  "$PY" "$K/read_boxsize.py" --stage controls --out "$OUT/BOXSIZE_CONTROLS.json" "${RARGS[@]}" > "$OUT/read_controls.log" 2>&1; rc=$?
  grep -E '^s[0-6] (PASS|FAIL)$' "$OUT/read_controls.log" | head -n 7
  tail -n 1 "$OUT/read_controls.log" | cut -c1-200 | sed '/hf_/d'
  echo "ZZ PODREAD-BOXSIZE controls DONE rc $rc $(ts)"
else
  [ -f "$OUT/BOXSIZE_CONTROLS.json" ] || stop "NO-CONTROLS-FILE"
  [ -e "$OUT/BOXSIZE_RESULT.json" ] && stop "ALREADY-READ names"
  "$PY" "$K/read_boxsize.py" --selftest > "$OUT/selftest_pod_names.log" 2>&1; st=$?
  [ "$st" = 0 ] || stop "SELFTEST-ON-POD rc $st"
  "$PY" "$K/read_boxsize.py" --stage names --controls "$OUT/BOXSIZE_CONTROLS.json" --out "$OUT/BOXSIZE_RESULT.json" --log "$OUT/read_names.log" "${RARGS[@]}" > "$OUT/read_names.out" 2>&1; rc=$?
  tail -n 1 "$OUT/read_names.out" | cut -c1-200 | sed '/hf_/d'
  echo "ZZ PODREAD-BOXSIZE names DONE rc $rc $(ts)"
fi

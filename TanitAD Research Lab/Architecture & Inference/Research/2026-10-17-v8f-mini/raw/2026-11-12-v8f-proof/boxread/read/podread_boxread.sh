#!/usr/bin/env bash
# podread_boxread.sh -- pod-side read of card PREREG_V8F_PROOF_ADD127 (BOXREAD-T0). CPU only (CUDA_VISIBLE_DEVICES=-1), no rollout.
# Needs BOXREAD_DET_DONE.txt and no BOXREAD_STOP.txt in the card root (the detector kit's markers). Reads the arguments of read_args.txt beside it (one token per line).
# Mode controls: read_boxread.py controls; prints only its ctl lines, the b5 share line and its ZZ line; every value goes to BOXREAD_CONTROLS.json in the out dir.
# Mode names: read_boxread.py names; refuses unless the controls file in the out dir says all pass; writes BOXREAD_RESULT.json and BOXREAD_VERDICT.md.
# It deletes, moves and stops nothing. A result file that exists is never overwritten (the reader writes a _2 file).
# Usage: bash podread_boxread.sh controls|names
# Last line: ZZ POD-BOXREAD <mode> DONE rc <n> <time>   or   ZZ POD-BOXREAD STOP <why>.
set -u
MODE=${1:-}
ROOT=${BR_ROOT:-/workspace/v8proof/boxread}
K=${BR_KIT:-$ROOT/kit/read}
OUT=${BR_OUT:-$ROOT/read_out}
PY=${BR_PY:-/workspace/venv/bin/python}
ARGS_FILE=${BR_ARGS:-$K/read_args.txt}
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ POD-BOXREAD STOP $*"; exit 3; }
case "$MODE" in controls|names) ;; *) stop "usage: controls|names";; esac
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export BOXSIZE_READER=${BOXSIZE_READER:-/workspace/v8proof/boxsize/kit/read/read_boxsize.py}
CG=${BR_CG:-/sys/fs/cgroup}; mm=""; an=""
[ -r "$CG/memory.max" ] && mm=$(tr -d ' \r\n' < "$CG/memory.max"); [ -r "$CG/memory.stat" ] && an=$(awk '/^anon /{print $2}' "$CG/memory.stat")
if [[ "$mm" =~ ^[0-9]+$ ]] && [[ "$an" =~ ^[0-9]+$ ]]; then
  free_gb=$(( (mm - an) / 1073741824 ))
  [ "$free_gb" -ge 10 ] || stop "MEMORY free ${free_gb} GB"
fi
[ -f "$K/read_boxread.py" ] || stop "MISSING-KIT-FILE $K/read_boxread.py"
[ -f "$K/box_geom.py" ] || stop "MISSING-KIT-FILE $K/box_geom.py"
[ -f "$ROOT/BOXREAD_DET_DONE.txt" ] || stop "DETECTOR-PASS-NOT-DONE"
[ -e "$ROOT/BOXREAD_STOP.txt" ] && stop "STOPPED"
[ -f "$ARGS_FILE" ] || stop "MISSING-INPUT $ARGS_FILE"
mkdir -p "$OUT" || stop "CANNOT-MAKE $OUT"
mapfile -t RARGS < <(tr -d '\r' < "$ARGS_FILE" | sed '/^$/d')
cd "$OUT" || stop "CANNOT-ENTER $OUT"
if [ "$MODE" = controls ]; then
  "$PY" "$K/read_boxread.py" controls --out-dir "$OUT" "${RARGS[@]}" > "$OUT/read_controls.log" 2>&1; rc=$?
  grep -E '^(ctl b[0-7] (PASS|FAIL)|b5 share|ZZ BOXREAD)' "$OUT/read_controls.log" | cut -c1-160
else
  "$PY" "$K/read_boxread.py" names --out-dir "$OUT" "${RARGS[@]}" > "$OUT/read_names.log" 2>&1; rc=$?
  grep -E '^(ZZ BOXREAD|REFUSED)' "$OUT/read_names.log" | cut -c1-200
fi
echo "ZZ POD-BOXREAD $MODE DONE rc $rc $(ts)"

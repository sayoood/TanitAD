#!/usr/bin/env bash
# podread_boxfuse.sh -- pod-side read of card PREREG_V8F_PROOF_ADD129 (BOXFUSE-T0). CPU only (CUDA_VISIBLE_DEVICES=-1), no rollout, no detector run, no fit.
# Reads the arguments of read_args.txt beside it (one token per line). The detector files are those of card ADD127's passes.
# Mode controls: prints only the ctl lines, the gate share lines and the ZZ line; every value goes to FUSE_CONTROLS.json in the out dir.
# Mode names: refuses unless the controls file of this card in the out dir says all pass.
# It deletes, moves and stops nothing. A result file that exists is never overwritten (the reader writes a _2 file).
# Usage: bash podread_boxfuse.sh controls|names
# Last line: ZZ POD-BOXFUSE <mode> DONE rc <n> <time>   or   ZZ POD-BOXFUSE STOP <why>.
set -u
MODE=${1:-}
ROOT=${BR_ROOT:-/workspace/v8proof/boxread}
K=${BRF_KIT:-$ROOT/kit_f}
OUT=${BRF_OUT:-$ROOT/read_out_f}
PY=${BR_PY:-/workspace/venv/bin/python}
ARGS_FILE=${BRF_ARGS:-$K/read_args.txt}
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ POD-BOXFUSE STOP $*"; exit 3; }
case "$MODE" in controls|names) ;; *) stop "usage: controls|names";; esac
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export BOXSIZE_READER=${BOXSIZE_READER:-/workspace/v8proof/boxsize/kit/read/read_boxsize.py}
export BOXREAD_READER=${BOXREAD_READER:-$ROOT/kit/read/read_boxread.py}
export TANITAD_STACK=${TANITAD_STACK:-/workspace/v8proof/tree/stack}
export PYTHONPATH=${PYTHONPATH:-/workspace/v8proof/tree/stack}
CG=${BR_CG:-/sys/fs/cgroup}; mm=""; an=""
[ -r "$CG/memory.max" ] && mm=$(tr -d ' \r\n' < "$CG/memory.max"); [ -r "$CG/memory.stat" ] && an=$(awk '/^anon /{print $2}' "$CG/memory.stat")
if [[ "$mm" =~ ^[0-9]+$ ]] && [[ "$an" =~ ^[0-9]+$ ]]; then
  free_gb=$(( (mm - an) / 1073741824 ))
  [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
fi
[ -f "$K/read_boxfuse.py" ] || stop "MISSING-KIT-FILE $K/read_boxfuse.py"
[ -f "$BOXREAD_READER" ] || stop "MISSING-KIT-FILE $BOXREAD_READER"
[ -f "$ARGS_FILE" ] || stop "MISSING-INPUT $ARGS_FILE"
mkdir -p "$OUT" || stop "CANNOT-MAKE $OUT"
mapfile -t RARGS < <(tr -d '\r' < "$ARGS_FILE" | sed '/^$/d')
cd "$OUT" || stop "CANNOT-ENTER $OUT"
if [ "$MODE" = controls ]; then
  "$PY" "$K/read_boxfuse.py" controls --out-dir "$OUT" "${RARGS[@]}" > "$OUT/read_controls.log" 2>&1; rc=$?
  grep -E '^(ctl f[0-5] (PASS|FAIL)|f1 shares|f3 share|ZZ BOXFUSE)' "$OUT/read_controls.log" | cut -c1-200
else
  "$PY" "$K/read_boxfuse.py" names --out-dir "$OUT" "${RARGS[@]}" > "$OUT/read_names.log" 2>&1; rc=$?
  grep -E '^(ZZ BOXFUSE|REFUSED)' "$OUT/read_names.log" | cut -c1-200
fi
echo "ZZ POD-BOXFUSE $MODE DONE rc $rc $(ts)"

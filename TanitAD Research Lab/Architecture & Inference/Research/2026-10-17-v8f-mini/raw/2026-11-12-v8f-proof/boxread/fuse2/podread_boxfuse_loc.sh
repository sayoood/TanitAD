#!/usr/bin/env bash
# podread_boxfuse_loc.sh -- pod-side read of card PREREG_V8F_PROOF_ADD130 (BOXFUSE-T0-LOCFIT). CPU only, no rollout, no detector run, no fit.
# Usage: bash podread_boxfuse_loc.sh controls LOC_s1|LOC_s2   |   names LOC_s1|LOC_s2   |   summary
# controls: prints only the ctl lines, the gate share lines and the ZZ line of one fit. names: writes that fit's values, prints no name; refuses unless both
# fits' controls say all pass. summary: the card's two names from the two result files. It deletes, moves and stops nothing.
# Last line: ZZ POD-LOCFIT <mode> <fit> DONE rc <n> <time>   or   ZZ POD-LOCFIT STOP <why>.
set -u
MODE=${1:-}; FIT=${2:-}
ROOT=${BR_ROOT:-/workspace/v8proof/boxread}
K=${BRL_KIT:-$ROOT/kit_l}
PY=${BR_PY:-/workspace/venv/bin/python}
ts() { date -u +%H:%M:%SZ; }
stop() { echo "ZZ POD-LOCFIT STOP $*"; exit 3; }
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export BOXSIZE_READER=${BOXSIZE_READER:-/workspace/v8proof/boxsize/kit/read/read_boxsize.py}
export BOXREAD_READER=${BOXREAD_READER:-$ROOT/kit/read/read_boxread.py}
export BOXFUSE_READER=${BOXFUSE_READER:-$ROOT/kit_f/read_boxfuse.py}
export TANITAD_STACK=${TANITAD_STACK:-/workspace/v8proof/tree/stack}
export PYTHONPATH=${PYTHONPATH:-/workspace/v8proof/tree/stack}
CG=/sys/fs/cgroup; mm=""; an=""
[ -r "$CG/memory.max" ] && mm=$(tr -d ' \r\n' < "$CG/memory.max"); [ -r "$CG/memory.stat" ] && an=$(awk '/^anon /{print $2}' "$CG/memory.stat")
if [[ "$mm" =~ ^[0-9]+$ ]] && [[ "$an" =~ ^[0-9]+$ ]]; then
  free_gb=$(( (mm - an) / 1073741824 ))
  [ "$free_gb" -ge 20 ] || stop "MEMORY free ${free_gb} GB"
fi
[ -f "$K/read_boxfuse_loc.py" ] || stop "MISSING-KIT-FILE"
O1=$ROOT/read_out_l/LOC_s1; O2=$ROOT/read_out_l/LOC_s2; OS=$ROOT/read_out_l/summary
if [ "$MODE" = summary ]; then
  "$PY" "$K/read_boxfuse_loc.py" summary "$O1" "$O2" "$OS" "$ROOT/read_out_f/FUSE_RESULT.json" 2>&1 | grep -E '^(ZZ LOCFIT|REFUSED)' | cut -c1-200
  echo "ZZ POD-LOCFIT summary - DONE rc ${PIPESTATUS[0]} $(ts)"; exit 0
fi
case "$MODE" in controls|names) ;; *) stop "usage";; esac
case "$FIT" in LOC_s1) OUT=$O1; OTHER=$O2;; LOC_s2) OUT=$O2; OTHER=$O1;; *) stop "usage";; esac
[ -f "$K/read_args.txt" ] && [ -f "$K/TESTS_EPOCH.txt" ] || stop "MISSING-INPUT"
EP=$(tr -d ' \r\n' < "$K/TESTS_EPOCH.txt")
mkdir -p "$OUT" || stop "CANNOT-MAKE $OUT"
mapfile -t RARGS < <(tr -d '\r' < "$K/read_args.txt" | sed '/^$/d')
cd "$OUT" || stop "CANNOT-ENTER $OUT"
"$PY" "$K/read_boxfuse_loc.py" "$MODE" --out-dir "$OUT" "${RARGS[@]}" --locator-fit "$FIT" --other-controls "$OTHER/FUSE_CONTROLS.json" \
  --test-out "loc=$ROOT/lead_tests/lead_pytest_loc.out" --test-before "loc=$EP" > "$OUT/read_$MODE.log" 2>&1; rc=$?
grep -E '^(ctl f[0-5] (PASS|FAIL)|f1 shares|f3 share|ZZ BOXFUSE|REFUSED)' "$OUT/read_$MODE.log" | cut -c1-200
echo "ZZ POD-LOCFIT $MODE $FIT DONE rc $rc $(ts)"

#!/bin/bash
# podread_dscale_eqp.sh <controls|names> -- pod driver of the reader of card ADD125 (read_dscale_eqp.py). CPU only.
# Arguments of the reader: read_args.txt beside this script, one token per line (--banked-run-dir, --banked-in-dir, --banked-done, --q-run-dir, --q-in-dir, --q-done, --new-run-dir, --new-in-dir,
# --new-done, --pack-dir, --test-out, optional --test-before and --B). It adds --out-dir <new-run-dir>/read itself. Needs the new run's done file. Writes into <new-run-dir>/read/; the reader never
# overwrites a result file of the same mode (a second call writes ..._2). No rm, no mv, no kill.
# Environment: PY (default /workspace/venv/bin/python), DSCALE_LEAD_KIT_DIR (folder of the banked reader; default two folders above this script), FINEGRID_BASE_DIR if the banked chain needs it,
# D2X_READER (the reader of card ADD123; default ../../add123/read/read_dscale2x.py beside this script), EQP_RUN_SCRIPT (the landed run script of card ADD123).
# Last line: ZZ POD-EQP <mode> DONE rc <rc> <time> | STOP <why>
set -u
MODE="${1:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/workspace/venv/bin/python}"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { echo "ZZ POD-EQP STOP $*"; exit 3; }
case "$MODE" in controls|names) ;; *) stop "MODE must be controls or names, got '$MODE'" ;; esac
[ -f "$HERE/read_args.txt" ] || stop "NO-READ-ARGS-FILE"
[ -f "$HERE/read_dscale_eqp.py" ] || stop "NO-READER"
ARGS=()
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%$'\r'}"
  [ -n "$line" ] && ARGS+=("$line")
done < "$HERE/read_args.txt"
argval() { local i; for ((i = 0; i < ${#ARGS[@]}; i++)); do if [ "${ARGS[$i]}" = "$1" ]; then echo "${ARGS[$((i + 1))]:-}"; return; fi; done; }
NEWRUN="$(argval --new-run-dir)"; NEWDONE="$(argval --new-done)"
[ -n "$NEWRUN" ] || stop "NO --new-run-dir IN read_args.txt"
[ -n "$NEWDONE" ] || stop "NO --new-done IN read_args.txt"
[ -f "$NEWDONE" ] || stop "NO-DONE-MARKER $NEWDONE"
for t in "${ARGS[@]}"; do [ "$t" = "--out-dir" ] && stop "--out-dir is set by this script"; done
OUT="$NEWRUN/read"
mkdir -p "$OUT" || stop "CANNOT-MAKE $OUT"
if [ -z "$(argval --test-before)" ]; then
  ARGS+=(--test-before "$(date +%s)")      # the read start: the kit's pytest output file has to be older than this
fi
sha256sum "$HERE/read_dscale_eqp.py" | cut -c1-16
KITDIR="${DSCALE_LEAD_KIT_DIR:-$(cd "$HERE/../.." && pwd)}"
export DSCALE_LEAD_KIT_DIR="$KITDIR"
[ -f "$KITDIR/dscale_lead_read.py" ] || stop "NO-BANKED-READER $KITDIR"
cd "$KITDIR" || stop "NO-DIR"          # the banked reader resolves its fixed paths from its own folder (as the read of card ADD45 did)
n=1; LOG="$OUT/read_${MODE}.out"
while [ -e "$LOG" ]; do n=$((n + 1)); LOG="$OUT/read_${MODE}_${n}.out"; done
"$PY" "$HERE/read_dscale_eqp.py" "$MODE" "${ARGS[@]}" --out-dir "$OUT" > "$LOG" 2>&1
rc=$?
cut -c1-160 "$LOG" | grep -E "^ctl |^beside |^ZZ " | tail -n 14
echo "ZZ POD-EQP $MODE DONE rc $rc $(ts)"
exit $rc

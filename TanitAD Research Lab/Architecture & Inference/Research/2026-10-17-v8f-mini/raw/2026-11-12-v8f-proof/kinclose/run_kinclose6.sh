#!/usr/bin/env bash
# run_kinclose6.sh -- pod chain of card PREREG_V8F_PROOF_ADD93 (GAPSPLIT), tier T0: self-test of kinclose6.py (control u4), then the read (controls u0 to u3, values, names). Numpy only, CPU.
# New files only: no removal, no move, no stopping, no process listing. Markers in $W: START_GAPSPLIT.txt (a second start is refused), GAPSPLIT_DONE.txt (rc of the read) or
# GAPSPLIT_STOP.txt (a failure). DRY=1 prints every command and runs nothing. Tools expected in $W/tools: kinclose6.py kinclose5.py kinclose4.py kinclose3.py kinclose2.py; in $CT: closeread_fit.py closest.py.
set -u
V=/workspace/v8proof
W=${KC_W:-$V/kinclose6}
PY=${KC_PY:-/workspace/venv/bin/python}
CT=${KC_CRTOOLS:-$V/closeread/tools}
DUMP=${KC_DUMP:-$V/closeread/dump/both}
EPMAP=${KC_EPMAP:-$V/closeread/episode_map.json}
A81C=${KC_ADD81_CONTROLS:-$V/tokclose/score/TOKCLOSE_CONTROLS.json}
A92=${KC_ADD92_RESULT:-$V/kinclose5/out/GAPREQ_RESULT.json}
MINGB=${KC_MIN_FREE_GB:-20}
DRY=${DRY:-0}
TOOL=$W/tools/kinclose6.py
OUT=$W/out
LOGD=$W/logs
export PYTHONDONTWRITEBYTECODE=1

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { if [ "$DRY" = 1 ]; then echo "STOP $*"; exit 3; fi; echo "$(ts) $*" > "$W/GAPSPLIT_STOP.txt"; exit 3; }

SELF=("$PY" "$TOOL" --selftest --closeread-tools "$CT")
READ=("$PY" "$TOOL" --closeread-tools "$CT" read --dump-dir "$DUMP" --episode-map "$EPMAP" --add81-controls "$A81C" --add92-result "$A92" --out-dir "$OUT")
if [ "$DRY" = 1 ]; then
  echo "CMD: memory free > $MINGB GB (memory.max minus anon), else STOP"; echo "CMD: ${SELF[*]}"; echo "CMD: ${READ[*]}"; echo "DRY run complete"; exit 0
fi
[ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }
( set -C; echo "$(ts) pid $$" > "$W/START_GAPSPLIT.txt" ) 2>/dev/null || { echo "REFUSED: $W/START_GAPSPLIT.txt exists (a chain was started)"; exit 2; }
mkdir -p "$LOGD"
for f in "$TOOL" "$W/tools/kinclose5.py" "$W/tools/kinclose4.py" "$W/tools/kinclose3.py" "$W/tools/kinclose2.py" "$CT/closeread_fit.py" "$CT/closest.py" "$DUMP/reader_s0_score.npz" "$DUMP/reader_s1_score.npz" "$EPMAP" "$A81C" "$A92"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
[ -e "$OUT" ] && stop "OUTPUT-EXISTS $OUT"
mmax=$(tr -d ' \r\n' < /sys/fs/cgroup/memory.max); anon=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat | head -n 1)
[[ "$mmax" =~ ^[0-9]+$ ]] && [[ "$anon" =~ ^[0-9]+$ ]] || stop "MEMORY-PROBE-UNREADABLE"
[ $(( (mmax - anon) / 1073741824 )) -gt "$MINGB" ] || stop "MEMORY-FREE-TOO-LOW"
"${SELF[@]}" > "$LOGD/selftest.log" 2>&1 || stop "SELFTEST-FAIL"
tail -n 1 "$LOGD/selftest.log" | grep -q "ZZ GAPSPLIT-SELFTEST rc 0" || stop "SELFTEST-NOT-GREEN"
"${READ[@]}" > "$LOGD/read.log" 2>&1
rc=$?
[ $rc -le 1 ] || stop "READ-FAILED rc $rc"
echo "$(ts) read rc $rc (0 = controls pass and names written; 1 = a control failed, no name)" > "$W/GAPSPLIT_DONE.txt"
exit 0

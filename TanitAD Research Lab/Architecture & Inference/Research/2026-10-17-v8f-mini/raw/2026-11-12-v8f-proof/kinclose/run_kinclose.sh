#!/usr/bin/env bash
# run_kinclose.sh -- pod chain of card PREREG_V8F_PROOF_ADD88 (KINCLOSE), tier T0: self-test, ego speed of the 141 clips, then the read (controls, arms, names). Numpy only except `ego` (torch).
# New files only: no removal, no move, no stopping, no process listing. Markers in $W: START_KINCLOSE.txt (a second start is refused), KINCLOSE_DONE.txt (rc of the read) or KINCLOSE_STOP.txt (a failure).
# DRY=1 prints every command and runs nothing. Tools expected in $W/tools: kinclose.py; in $CT: closeread_fit.py closest.py.
set -u
V=/workspace/v8proof
W=${KC_W:-$V/kinclose}
PY=${KC_PY:-/workspace/venv/bin/python}
CT=${KC_CRTOOLS:-$V/closeread/tools}
DUMP=${KC_DUMP:-$V/closeread/dump/both}
EST=${KC_EST:-$V/closeread/est}
EPMAP=${KC_EPMAP:-$V/closeread/episode_map.json}
EVALC=${KC_EVAL_CACHE:-$V/integ10k/eval_cache}
A81R=${KC_ADD81_RESULT:-$V/tokclose/score/TOKCLOSE_RESULT.json}
A81C=${KC_ADD81_CONTROLS:-$V/tokclose/score/TOKCLOSE_CONTROLS.json}
MINGB=${KC_MIN_FREE_GB:-20}
MM=${KC_MEMMAX_FILE:-/sys/fs/cgroup/memory.max}; MS=${KC_MEMSTAT_FILE:-/sys/fs/cgroup/memory.stat}
DRY=${DRY:-0}
TOOL=$W/tools/kinclose.py
EGO=$W/ego_v_eval.npz
OUT=$W/out
LOGD=$W/logs
export PYTHONDONTWRITEBYTECODE=1

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stop() { if [ "$DRY" = 1 ]; then echo "STOP $*"; exit 3; fi; echo "$(ts) $*" > "$W/KINCLOSE_STOP.txt"; exit 3; }
free_gb() { # container memory free in GB (memory.max minus the anon line of memory.stat); empty on failure
  local mmax anon av
  [ -r "$MM" ] && [ -r "$MS" ] || return 1
  mmax=$(tr -d ' \r\n' < "$MM"); anon=$(awk '$1=="anon"{print $2}' "$MS" | head -n 1)
  [[ "$anon" =~ ^[0-9]+$ ]] || return 1
  if [ "$mmax" = max ]; then
    av=$(awk '$1=="MemAvailable:"{print $2*1024}' /proc/meminfo | head -n 1); [[ "$av" =~ ^[0-9]+$ ]] || return 1
    awk -v b="$av" 'BEGIN{printf "%.1f", b/1073741824}'
  else
    [[ "$mmax" =~ ^[0-9]+$ ]] || return 1
    awk -v b="$((mmax - anon))" 'BEGIN{printf "%.1f", b/1073741824}'
  fi
}

if [ "$DRY" != 1 ]; then
  [ -d "$W" ] || { echo "REFUSED: no $W"; exit 2; }
  ( set -C; echo "$(ts) pid $$" > "$W/START_KINCLOSE.txt" ) 2>/dev/null || { echo "REFUSED: $W/START_KINCLOSE.txt exists (a chain was started)"; exit 2; }
  mkdir -p "$LOGD"
  for f in "$TOOL" "$CT/closeread_fit.py" "$CT/closest.py" "$DUMP/reader_s0_score.npz" "$DUMP/reader_s1_score.npz" "$EST/closest_L25_s0.npz" "$EST/closest_L25_s1.npz" "$EPMAP" "$A81R" "$A81C"; do
    [ -f "$f" ] || stop "MISSING-INPUT $f"
  done
  [ -e "$EGO" ] && stop "OUTPUT-EXISTS $EGO"
  [ -e "$OUT" ] && stop "OUTPUT-EXISTS $OUT"
  g=$(free_gb) || stop "MEMORY-PROBE-UNREADABLE"
  awk -v f="$g" -v m="$MINGB" 'BEGIN{exit !(f > m)}' || stop "MEMORY-FREE-TOO-LOW $g GB (need more than $MINGB)"
fi

SELF=("$PY" "$TOOL" --selftest --closeread-tools "$CT")
EGOC=("$PY" "$TOOL" ego --eval-cache "$EVALC" --dump-dir "$DUMP" --out "$EGO")
READ=("$PY" "$TOOL" read --dump-dir "$DUMP" --ego "$EGO" --est-dir "$EST" --episode-map "$EPMAP" --add81-result "$A81R" --add81-controls "$A81C" --closeread-tools "$CT" --out-dir "$OUT")
if [ "$DRY" = 1 ]; then
  echo "CMD: memory free > $MINGB GB (memory.max minus anon), else STOP"
  echo "CMD: ${SELF[*]}"; echo "CMD: ${EGOC[*]}"; echo "CMD: ${READ[*]}"; echo "DRY run complete"; exit 0
fi

"${SELF[@]}" > "$LOGD/selftest.log" 2>&1 || stop "SELFTEST-FAIL"
tail -n 1 "$LOGD/selftest.log" | grep -q "ZZ KINCLOSE-SELFTEST rc 0" || stop "SELFTEST-NOT-GREEN"
"${EGOC[@]}" > "$LOGD/ego.log" 2>&1 || stop "EGO-FAILED"
[ -f "$EGO" ] || stop "EGO-OUTPUT-MISSING"
"${READ[@]}" > "$LOGD/read.log" 2>&1
rc=$?
[ $rc -le 1 ] || stop "READ-FAILED rc $rc"
echo "$(ts) read rc $rc (0 = controls pass and names written; 1 = a control failed, no name)" > "$W/KINCLOSE_DONE.txt"
exit 0

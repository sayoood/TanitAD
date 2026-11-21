#!/bin/bash
# brake_locus_reroll_after_40k.sh -- detached waiter: runs the brake-locus CPU re-roll (brake_locus_reroll_r8.py, chunk by chunk) once the step-40k ship step is done.
#   START  when E:/refcv8_eval_kit/chain/ship_r8.log has a line STARTING with the ship-end marker that names step40000ema_unc, OR raw/ship_r8/SHIP_R8_step40000ema_unc.json
#          exists (whichever first).  If neither by 2026-10-20 06:00 (+0200) start anyway, PROVIDED the 40k chain is not running a CPU lane (its lane pid is alive AND the
#          last marker of chain_40k.log is not a wait marker); otherwise keep waiting.
#   STOP   never START a chunk after 2026-10-20 15:00 (+0200), or once chain_final.log shows ANY marker beyond its wait state (the final checkpoint chain gets the box),
#          or when the stop file exists.  Stops between chunks; the state is each chunk's progress.json (resumable: re-run this script).
#   GATE   every chunk waits for >= 5 GB free COMMIT (two reads 20 s apart); ONE model process at a time; OMP_NUM_THREADS=6; CPU only (CUDA_VISIBLE_DEVICES=-1).
#   MARKERS opaque ZZBLR<NAME>ZZ -> E:/refcv8_eval_kit/chain/brake_reroll.log.  Terminal: STOPTIME | STOPFINAL | STOPFILE | ALLDONE | ABORT; DUP = another waiter is alive.
#   TEST   every external fact is an env var (BLR_*), see the defaults below.  Launch:  nohup bash brake_locus_reroll_after_40k.sh > /dev/null 2>&1 < /dev/null &
HERE="$(cd "$(dirname "$0")" && pwd)"
KIT="${BLR_KIT:-E:/refcv8_eval_kit}"
CHAIN="$KIT/chain"
LOG="${BLR_LOG:-$CHAIN/brake_reroll.log}"
PIDF="${BLR_PIDF:-$CHAIN/brake_reroll.pid}"
STOPF="${BLR_STOPFILE:-$CHAIN/brake_reroll.stop}"
SHIPLOG="${BLR_SHIP_LOG:-$CHAIN/ship_r8.log}"
SHIPJSON="${BLR_SHIP_JSON:-$HERE/../raw/ship_r8/SHIP_R8_step40000ema_unc.json}"
C40LOG="${BLR_C40_LOG:-$CHAIN/chain_40k.log}"
C40LANEPID="${BLR_C40_LANE_PID:-$CHAIN/chain_40k_lane.pid}"
FINALLOG="${BLR_FINAL_LOG:-$CHAIN/chain_final.log}"
TREE_TAG="${BLR_TREE_TAG:-step40000ema_unc}"
DEADLINE_START="${BLR_DEADLINE_START:-2026-10-20 06:00:00 +0200}"
DEADLINE_STOP="${BLR_DEADLINE_STOP:-2026-10-20 15:00:00 +0200}"
PY="${BLR_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
TOOL="${BLR_TOOL:-$HERE/brake_locus_reroll_r8.py}"
POLL="${BLR_POLL_S:-120}"
MEMPOLL="${BLR_MEM_POLL_S:-60}"
MINGB="${BLR_MIN_GB:-5}"
MAXTRY="${BLR_MAX_TRY:-3}"
CHUNKLOGD="${BLR_CHUNK_LOG_DIR:-$KIT/brake_locus_scratch/logs}"
mkdir -p "$CHUNKLOGD" "$CHAIN" 2>/dev/null

mark() { echo "ZZBLR$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
now_s() { if [ -n "$BLR_NOW_EPOCH" ]; then echo "$BLR_NOW_EPOCH"; else date +%s; fi; }
t_start="$(date -d "$DEADLINE_START" +%s 2>/dev/null)"
t_stop="$(date -d "$DEADLINE_STOP" +%s 2>/dev/null)"
if [ "${#t_start}" -lt 9 ] || [ "${#t_stop}" -lt 9 ]; then echo "ZZBLRABORTZZ $(date +%FT%T%z) cannot parse the deadlines" >> "$LOG"; exit 2; fi

# ---- single instance
if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null && [ "$(cat "$PIDF")" != "$$" ]; then mark DUP "another waiter is alive (pid $(cat "$PIDF"))"; exit 5; fi
echo $$ > "$PIDF"

free_commit_gb() {   # decimal GB; the dev box prints a comma decimal; empty on failure (an empty read is INCONCLUSIVE, never free)
  if [ -n "$BLR_FREE_GB" ]; then echo "$BLR_FREE_GB"; return; fi
  powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" 2>/dev/null | tr -d '\r' | sed 's/,/./' | head -n 1
}
ge() { awk -v a="$1" -v b="$2" 'BEGIN{ if (a ~ /^[0-9]+(\.[0-9]+)?$/ && a+0 >= b+0) exit 0; exit 1 }'; }

ship_done() {
  if [ -f "$SHIPLOG" ] && grep -E "^ZZSHIPENDZZ .*$TREE_TAG" "$SHIPLOG" > /dev/null 2>&1; then echo "ship-end-marker"; return 0; fi
  if [ -f "$SHIPJSON" ]; then echo "ship-json"; return 0; fi
  return 1
}
c40_lane_active() {   # the lane pid is alive AND the last chain_40k marker is not a wait marker
  local lp last
  [ -s "$C40LANEPID" ] || return 1
  lp="$(cat "$C40LANEPID")"
  kill -0 "$lp" 2> /dev/null || return 1
  last="$(grep -o '^ZZC40[A-Z0-9]*ZZ' "$C40LOG" 2> /dev/null | tail -n 1)"
  case "$last" in ZZC40LANEWAITZZ|ZZC40SNAPWAITZZ|ZZC40MEMWAITZZ) return 1 ;; esac
  return 0
}
final_busy() {   # any ZZCFN marker other than START / PULLSTART / DUP / *WAIT means the final chain is doing work
  [ -f "$FINALLOG" ] || return 1
  grep -o '^ZZCFN[A-Z0-9]*ZZ' "$FINALLOG" 2> /dev/null | sed -e 's/^ZZCFN//' -e 's/ZZ$//' | grep -v -E '^(START|PULLSTART|DUP)$|WAIT$' | grep -q . && return 0
  return 1
}
stop_reason() {   # prints a terminal marker name when no NEW chunk may start
  if [ -e "$STOPF" ]; then echo STOPFILE; return 0; fi
  if [ "$(now_s)" -ge "$t_stop" ]; then echo STOPTIME; return 0; fi
  if final_busy; then echo STOPFINAL; return 0; fi
  return 1
}

mark START "pid $$ tag $TREE_TAG start-after $DEADLINE_START stop-at $DEADLINE_STOP tool $TOOL"
# ---------------------------------------------------------------- phase 1: wait for the start condition
_n=0
while true; do
  if r="$(stop_reason)"; then mark "$r" "before start"; exit 0; fi
  if why="$(ship_done)"; then mark GO "start condition: $why"; break; fi
  if [ "$(now_s)" -ge "$t_start" ]; then
    if c40_lane_active; then
      _n=$((_n+1)); [ $((_n % 15)) -eq 1 ] && mark DEADLINEWAIT "past the start deadline with no ship end, but the 40k chain runs a CPU lane"
    else
      mark GO "start condition: deadline fallback (no ship end by $DEADLINE_START; no 40k CPU lane active)"; break
    fi
  else
    _n=$((_n+1)); [ $((_n % 30)) -eq 1 ] && mark WAIT "waiting for the step-40k ship end ($TREE_TAG)"
  fi
  sleep "$POLL"
done

# ---------------------------------------------------------------- phase 2: chunk loop
skip=""
declare -A tries
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 R8_NO_POSE_SYNC=1
export PYTHONPATH="${BLR_PYTHONPATH:-C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$(cd "$HERE" && pwd -W 2> /dev/null || pwd)}"
next_chunk() { "$PY" -X utf8 "$TOOL" next --skip "$skip" 2> /dev/null | grep -o 'ZZBLRNEXTZZ [A-Za-z0-9_-]*' | tail -n 1 | cut -d' ' -f2; }
while true; do
  if r="$(stop_reason)"; then mark "$r" "no new chunk will start; state is resumable (re-run this script)"; exit 0; fi
  nxt="$(next_chunk)"
  if [ -z "$nxt" ]; then mark ABORT "the tool next-chunk query returned nothing (see $TOOL)"; exit 3; fi
  if [ "$nxt" = "NONE" ]; then mark ALLDONE "every chunk complete (skipped: ${skip:-none})"; exit 0; fi
  # ---- memory gate: two reads 20 s apart (the COMMIT limit binds, not RAM)
  _m=0
  while true; do
    if r="$(stop_reason)"; then mark "$r" "while waiting for commit headroom before $nxt"; exit 0; fi
    g1="$(free_commit_gb)"; sleep "${BLR_GATE_GAP_S:-20}"; g2="$(free_commit_gb)"
    if ge "$g1" "$MINGB" && ge "$g2" "$MINGB"; then break; fi
    _m=$((_m+1)); [ $((_m % 20)) -eq 1 ] && mark MEMWAIT "$nxt: free commit reads $g1 / $g2 GB, need $MINGB (an empty read counts as not free)"
    sleep "$MEMPOLL"
  done
  tries[$nxt]=$(( ${tries[$nxt]:-0} + 1 ))
  mark CHUNKSTART "$nxt try ${tries[$nxt]} free commit $g2 GB"
  if [ -n "$BLR_STUB_RUN" ]; then bash -c "$BLR_STUB_RUN $nxt" >> "$CHUNKLOGD/chunk_$nxt.log" 2>&1 < /dev/null
  else "$PY" -X utf8 -u "$TOOL" roll --chunk "$nxt" >> "$CHUNKLOGD/chunk_$nxt.log" 2>&1 < /dev/null; fi
  # the success test is the ARTIFACT (the chunk progress), never an exit code
  after="$(next_chunk)"
  if [ "$after" = "$nxt" ]; then
    mark CHUNKINCOMPLETE "$nxt still has windows left after try ${tries[$nxt]} (log $CHUNKLOGD/chunk_$nxt.log)"
    if [ "${tries[$nxt]}" -ge "$MAXTRY" ]; then skip="${skip:+$skip,}$nxt"; mark CHUNKFAIL "$nxt skipped after $MAXTRY tries"; fi
  else
    mark CHUNKDONE "$nxt"
  fi
done

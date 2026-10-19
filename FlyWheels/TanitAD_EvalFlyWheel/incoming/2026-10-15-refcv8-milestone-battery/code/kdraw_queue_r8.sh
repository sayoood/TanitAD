#!/bin/sh
# kdraw_queue_r8.sh -- roll inference seeds 2..7 of one refcv8 EMA checkpoint for the K-draw ensemble read (PREREG_KDRAW_E1.md).  NEW file.
#   start (detached, ONCE):  nohup sh <PKG>/code/kdraw_queue_r8.sh < /dev/null > /e/refcv8_eval_kit/chain/kdraw_queue.out 2>&1 &
#   alive?                   ps -ef | grep -c '[k]draw_queue_r8'   and   grep ZZKDQ /e/refcv8_eval_kit/chain/kdraw_queue.log | tail
#   resumable: re-run the same command; a job whose artifact already says ok is skipped.   Final checkpoint:  KD_STEP=48852 sh kdraw_queue_r8.sh
#
# JOBS (KD_JOBS, default "ctrl 2 3 4 5 6 7", in that order so set A = {0,1,2,3} completes first):
#   ctrl  prefix re-rolls of seed 1 (os-only), seed 0 (full) and seed 0 (os-only) vs the BANKED dumps  -> $KIT/kdraw/<tree>/KDRAW_CONTROL.json
#   <k>   seed k, os-only, the banked windows file byte for byte                                       -> $KIT/kdraw/<tree>/dump_s<k>/KDRAW_VERIFY.json
# GATES before every GPU segment (a segment = the jobs run under ONE GPU-lock hold, at most KD_SEG_MAX = 4 jobs):
#   Y1  the float64 probe: not running, and not PENDING behind force_s2 (f64_after_fs2.sh alive, force_s2 terminal, no probe terminal marker); the probe drives
#       free commit below 4 GB and aborts if anything else runs     (cap KD_F64_YIELD_S, default 14400 s)
#   Y2  force_s2 goes FIRST: its launcher is alive, its gate G1-G3 is open and G4 has not opened yet -> wait (cap KD_FS2_YIELD_S, default 3600 s)
#   M   Win32_OperatingSystem.FreeVirtualMemory >= KD_NEED_GB (10) on TWO checks 60 s apart (the dev box COMMIT limit binds)
#   L   the shared O_EXCL GPU lock (gpu_lock.py; a foreign holder is NEVER broken; cap KD_LOCK_MAXWAIT_S, default 28800 s)
# Between the jobs of a segment Y1/Y2 and a single memory check are re-evaluated; a yield releases the lock and the segment ends.
# LAST STEP, on success OR failure (EXIT trap, after the GPU lock is released):  echo "released by kdraw <ts>" > /c/lgs/RELEASE_R7
# Assert on ARTIFACTS, never on an exit code, never through a pipe.  No git; nothing is written to the repo or to G:.
KD_STEP="${KD_STEP:-30000}"
TREE="step${KD_STEP}ema_unc"
PKGW="${R8_PKGW:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
PKGL="${R8_PKGL:-/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
BAT="${R8_BAT:-C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code}"
KITW="${R8_KITW:-E:/refcv8_eval_kit}"
KIT="${R8_KIT:-/e/refcv8_eval_kit}"
PY="${R8_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
ROLL="$PKGW/code/kdraw_roll_r8.py"
POLL="${KD_POLL_S:-60}"
GAP="${KD_GATE_GAP_S:-60}"
NEED="${KD_NEED_GB:-10}"
JOBS="${KD_JOBS:-ctrl 2 3 4 5 6 7}"
SEGMAX="${KD_SEG_MAX:-4}"
LOCK_MAXWAIT="${KD_LOCK_MAXWAIT_S:-28800}"
MEM_MAXWAIT="${KD_MEM_MAXWAIT_S:-43200}"
F64_CAP="${KD_F64_YIELD_S:-14400}"
FS2_CAP="${KD_FS2_YIELD_S:-3600}"
RELEASE="${KD_RELEASE_FILE:-/c/lgs/RELEASE_R7}"
DRY="${KD_DRY:-0}"
F64LOG="${KD_F64LOG:-$PKGL/raw/step30000/wrapper_fp64/raw/run.log}"
QLOG="${KD_QLOG:-$KIT/chain/queue_30000_r8.log}"
FLL="${KD_FLL:-$KIT/force_lon_scratch/roll_full.log}"
FS2LOG="${KD_FS2LOG:-$KIT/chain/force_s2.log}"
LOG="${KD_LOG:-$KIT/chain/kdraw_queue.log}"
KD_ROOT="$KIT/kdraw/$TREE"
JOB="refcv8-kdraw-$TREE"
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6
mkdir -p "$KIT/chain" "$KD_ROOT"
mark() { echo "ZZKDQ$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
jq_ok() {   # $1 = file, $2 = python boolean expression over `j`; exit 0 iff the file parses and the expression is true
  _f=$(cygpath -m "$1" 2>/dev/null || echo "$1")
  "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$_f"
}
# ---- single instance ------------------------------------------------------------------------------------------------ #
PIDF="$KIT/chain/kdraw_queue.pid"
if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then mark DUP "another kdraw queue (pid $(cat "$PIDF")) is alive; exiting"; exit 5; fi
echo $$ > "$PIDF"
HAVE_LOCK=0
gpu_release() {
  [ "$HAVE_LOCK" = 1 ] || return 0
  "$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$KIT/chain/kdraw_lockwait.log" 2>&1
  "$PY" "$BAT/chain_marker.py" remove --chain "$JOB" --pid $$ >> "$KIT/chain/kdraw_lockwait.log" 2>&1
  HAVE_LOCK=0; mark RELEASED "$JOB"
}
finish() {
  gpu_release
  [ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"
  if [ "$DRY" = 1 ] || [ "${KD_NO_RELEASE:-0}" = 1 ]; then mark NORELEASE "dry run or KD_NO_RELEASE: $RELEASE not written"; return 0; fi
  echo "released by kdraw $(date +%FT%T%z)" > "$RELEASE"      # LAST step, success or failure
  mark RELEASE_R7 "wrote $RELEASE"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP
mark QUEUESTART "pid $$ tree $TREE jobs '$JOBS' need ${NEED} GB seg_max $SEGMAX dry $DRY"
# ---- yields ------------------------------------------------------------------------------------------------------------ #
f64_term() {   # a terminal marker AFTER the last wait marker (the file also holds stale lines from aborted attempts)
  [ -f "$F64LOG" ] || return 1
  c=$(awk '/ZZF64WAITZZ/{c=0; next} /ZZF64(DONE|ARMFAIL|COMBINEFAIL|TIMEOUT)ZZ/{c++} END{print c+0}' "$F64LOG" 2>/dev/null)
  [ "${c:-0}" -ge 1 ]
}
f64_alive() { c=$(ps -ef 2>/dev/null | grep -c '[r]un_fp64_30k'); [ "${c:-0}" -ge 1 ] 2>/dev/null; }
fs2_terminal() { grep -qE 'ZZFS2(ANALYZEDONE|ANALYZEFAIL|ROLLFAIL|ROLLINCOMPLETE|SELECTFAIL|FAIL|TIMEOUT|DRYEND)ZZ' "$FS2LOG" 2>/dev/null; }
f64v5_pending() {   # the post-force_s2 float64 job (/c/lgs/f64_after_fs2.sh) is alive, force_s2 is terminal, and its probe has no terminal marker after the last wait
  c=$(ps -ef 2>/dev/null | grep -c '[f]64_after_fs2'); [ "${c:-0}" -ge 1 ] 2>/dev/null || return 1
  f64_term && return 1
  fs2_terminal || return 1          # still behind force_s2: the force_s2 yield / the memory gate cover that phase
  return 0
}
f64_hold() { { f64_alive && ! f64_term; } || f64v5_pending; }
fl_busy() {    # the force_lon roll is alive: no end marker AND its log was written within 15 minutes
  grep -q ZZFLDRIVERENDZZ "$FLL" 2>/dev/null && return 1
  [ -f "$FLL" ] || return 1
  [ -n "$(find "$FLL" -mmin -15 2>/dev/null)" ]
}
fs2_ready_pending() {   # force_s2's launcher is alive, G1-G3 are open, and its GATEOPEN has not been logged since its last ARMED
  c=$(ps -ef 2>/dev/null | grep -c '[f]orce_s2_launcher'); [ "${c:-0}" -ge 1 ] 2>/dev/null || return 1
  awk '/ZZFS2ARMEDZZ/{a=NR; o=0} /ZZFS2GATEOPENZZ/{o=1} END{exit (a>0 && o==0)?0:1}' "$FS2LOG" 2>/dev/null || return 1
  grep -q ZZQQUEUEENDZZ "$QLOG" 2>/dev/null || return 1
  if f64_term || ! f64_alive; then :; else return 1; fi
  fl_busy && return 1
  return 0
}
IGN_F64=0; IGN_FS2=0
yield_reason() {   # prints the reason to keep waiting, or nothing (a yield whose cap was hit is ignored for the rest of the run)
  if [ "$IGN_F64" != 1 ] && f64_hold; then echo "float64-probe"; return; fi
  if [ "$IGN_FS2" != 1 ] && fs2_ready_pending; then echo "force_s2-goes-first"; return; fi
}
wait_yields() {
  t0=$(date +%s); last=""
  while :; do
    el=$(( $(date +%s) - t0 ))
    if [ "$IGN_F64" != 1 ] && [ "$el" -ge "$F64_CAP" ] && f64_hold; then IGN_F64=1; mark YIELDCAP "float64 probe still running after ${F64_CAP}s: no longer yielding to it"; fi
    if [ "$IGN_FS2" != 1 ] && [ "$el" -ge "$FS2_CAP" ] && fs2_ready_pending; then IGN_FS2=1; mark YIELDCAP "force_s2 still pending after ${FS2_CAP}s: no longer yielding to it"; fi
    r=$(yield_reason)
    [ -z "$r" ] && return 0
    [ "$r" != "$last" ] && { mark YIELD "$r"; last="$r"; }
    sleep "$POLL"
  done
}
# ---- memory ------------------------------------------------------------------------------------------------------------ #
free_gb() { v=$("$PY" "$ROLL" freegb 2>/dev/null | tr -d '\r '); case "$v" in ''|*[!0-9.]*) echo 0;; *) echo "$v";; esac; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit (a+0>=b+0)?0:1}'; }
mem_gate() {   # two checks 60 s apart, both >= NEED
  t0=$(date +%s); n=0
  while :; do
    h1=$(free_gb)
    if ge "$h1" "$NEED"; then
      sleep "$GAP"
      h2=$(free_gb)
      if ge "$h2" "$NEED"; then mark MEMOK "free commit ${h1}/${h2} GB"; return 0; fi
      mark MEMPAIR "second check ${h2} GB < ${NEED}: the pair restarts"
    fi
    n=$((n+1)); [ $((n % 30)) -eq 1 ] && mark MEMWAIT "free commit ${h1} GB < ${NEED}"
    [ $(( $(date +%s) - t0 )) -gt "$MEM_MAXWAIT" ] && { mark MEMTIMEOUT "${MEM_MAXWAIT}s"; return 1; }
    sleep "$POLL"
  done
}
# ---- GPU lock ---------------------------------------------------------------------------------------------------------- #
gpu_acquire() {
  rm -f "$KIT/kdraw/lock_$TREE.json"                       # a stale ok=true from an earlier run must not pass for this acquire
  "$PY" "$BAT/gpu_lock.py" acquire --job "$JOB" --pid $$ --max-wait-s "$LOCK_MAXWAIT" --out "$KITW/kdraw/lock_$TREE.json" >> "$KIT/chain/kdraw_lockwait.log" 2>&1
  jq_ok "$KIT/kdraw/lock_$TREE.json" "j.get('ok') is True" || { mark NOLOCK "not acquired within ${LOCK_MAXWAIT}s (a foreign holder is never broken)"; return 3; }
  HAVE_LOCK=1
  "$PY" "$BAT/chain_marker.py" create --chain "$JOB" --pid $$ >> "$KIT/chain/kdraw_lockwait.log" 2>&1
  mark LOCKED "$JOB"
}
# ---- jobs -------------------------------------------------------------------------------------------------------------- #
job_done() {
  case "$1" in
    ctrl) jq_ok "$KD_ROOT/KDRAW_CONTROL.json" "'verdict' in j and len(j.get('cases',{}))>=3" ;;
    *)    jq_ok "$KD_ROOT/dump_s$1/KDRAW_VERIFY.json" "j.get('ok') is True" ;;
  esac
}
job_run() {
  case "$1" in
    ctrl) "$PY" "$ROLL" control --tree "$TREE" >> "$KIT/chain/kdraw_queue_py.log" 2>&1 ;;
    *)    "$PY" "$ROLL" roll --seed "$1" --tree "$TREE" >> "$KIT/chain/kdraw_queue_py.log" 2>&1 ;;
  esac
}
NFAIL=0; SEGN=0; ABORT=0
for j in $JOBS; do
  [ "$ABORT" = 1 ] && break
  if job_done "$j"; then mark JOBHAVE "$j"; continue; fi
  while :; do
    if [ "$HAVE_LOCK" = 1 ]; then
      # inside a segment: yields and one memory check; on any trouble give the card back and start a new segment
      if [ -n "$(yield_reason)" ] || ! ge "$(free_gb)" "$NEED" || [ "$SEGN" -ge "$SEGMAX" ]; then
        mark SEGEND "segment ends before job $j (yield / memory / seg_max)"; gpu_release; SEGN=0; continue
      fi
    else
      wait_yields
      mem_gate || { ABORT=1; break; }
      if [ "$DRY" = 1 ]; then mark DRYJOB "would acquire the GPU lock and run job $j"; break; fi
      gpu_acquire || { ABORT=1; break; }
      SEGN=0
      wait_yields                      # a yield that appeared while we waited for the lock: honour it before touching the card
      if ! ge "$(free_gb)" "$NEED"; then mark SEGEND "memory dropped while waiting for the lock"; gpu_release; continue; fi
    fi
    mark JOBSTART "$j"
    job_run "$j"
    if job_done "$j"; then mark JOBDONE "$j"; else mark JOBFAIL "$j (artifact not ok; see $KIT/chain/kdraw_queue_py.log and $KD_ROOT)"; NFAIL=$((NFAIL+1)); fi
    SEGN=$((SEGN+1))
    break
  done
done
gpu_release
mark QUEUEEND "jobs '$JOBS' attempted; failures $NFAIL; abort $ABORT"

#!/bin/sh
# chain_r7_matched_phase.sh -- the refcv7 matched-phase batteries (PREREG_R7_MATCHED_PHASE_TURN.md): refcv7 RAW 20,000 -> 30,000 -> 15,000 through the
# refcv8 battery path, both inference seeds, standalone pose clock, then the CPU analyses (r7m_analysis.sh, a detached child).
#
#   ⛔ FAIL-CLOSED GATE.  The pre-registration's known-value control (section 2) is binding: this chain REFUSES to start unless the control artifact of the
#   chosen path reads PASS, or the Master Mind has written an explicit amendment artifact (raw/r7_matched_phase/CONTROL_AMENDMENT.json carrying
#   `approved_by`, `path`, `amended_criterion`, `date`).  2026-10-18: the wrapper path and the native path both read FAIL (KNOWN_VALUE_CONTROL*.md) -- so as
#   shipped this chain does NOT run.  It is here, tested for syntax and for its refusal, so that the decision costs one command.
#   usage: R7M_PATH=wrapper|ceiloff nohup sh chain_r7_matched_phase.sh < /dev/null > ... &
#     wrapper : run_battery_r7m.py as is (max-speed ceiling filter ACTING, like every refcv8 battery; seed-0 rolls also bank the filter-OFF arm)
#     ceiloff : the same with the ceiling filter switched OFF in the main forward (R7M_CEIL_OFF=1; the instrument of the banked refcv7-50,400)
#   Order and queue discipline (brief): 20,000, 30,000, then 15,000 ONLY after the refcv8 step-30,000 queue has written its terminal marker; before 20,000/30,000
#   a battery does not START while that queue is pending (its checkpoint pair was seen stable) -- it keeps its place in the lock order; a lock is never broken.
#   Every GPU step: wait_commit -> shared lock (job refcv7-matched-step<N>) -> battery (resumable) -> release (EXIT trap).  Markers ZZR7M...ZZ in
#   E:/refcv8_eval_kit/chain/chain_r7_matched_phase.log.  Assert on ARTIFACTS, never on an exit code / `$?` through a pipe.
. "$(dirname "$0")/r7m_common.sh"
PATHV="${R7M_PATH:-wrapper}"
case "$PATHV" in
  wrapper) CTLJ="$RAWL/KNOWN_VALUE_CONTROL.json"; unset R7M_CEIL_OFF ;;
  ceiloff) CTLJ="$RAWL/KNOWN_VALUE_CONTROL_ceiloff.json"; export R7M_CEIL_OFF=1 ;;
  *) mark CHAINBADPATH "R7M_PATH='$PATHV' (wrapper|ceiloff)"; exit 2 ;;
esac
AMEND="$RAWL/CONTROL_AMENDMENT.json"
PIDF="$KIT/chain/chain_r7_matched_phase.pid"
if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then mark CHAINDUP "another chain (pid $(cat "$PIDF")) is alive; exiting"; exit 5; fi
if jq_ok "$CTLJ" "str(j.get('verdict','')).startswith('PASS')"; then GATE="control PASS ($CTLJ)"
elif jq_ok "$AMEND" "j.get('approved_by') and j.get('path')=='$PATHV' and j.get('amended_criterion') and j.get('date')"; then GATE="control AMENDED by the Master Mind ($AMEND)"
else
  mark CHAINREFUSED "path $PATHV: control artifact $CTLJ does not read PASS and no matching $AMEND exists -- VOID per PREREG section 2; nothing launched"
  exit 6
fi
echo $$ > "$PIDF"
trap 'trap - TERM INT HUP; exit 143' TERM INT HUP
trap '[ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"' EXIT
mark CHAINSTART "pid $$ path $PATHV; gate: $GATE"
note "chain armed (pid $$) path $PATHV; gate: $GATE"
# ---- the CPU analyses run in a detached child (they wait for their inputs; no lock) --------------------------------------------------------- #
if ! ( [ -s "$KIT/chain/r7m_analysis.pid" ] && kill -0 "$(cat "$KIT/chain/r7m_analysis.pid")" 2>/dev/null ); then
  nohup sh "$PKGL/code/r7m_analysis.sh" < /dev/null > "$KIT/chain/r7m_analysis.out" 2>&1 &
  mark ANALYSISSPAWNED "r7m_analysis.sh pid $!"
fi
# ---- queue discipline -------------------------------------------------------------------------------------------------------------------------- #
queue_terminal() {   # true iff the refcv8 step-30,000 queue / its waiter reached a terminal state
  grep -q "ZZQQUEUEENDZZ" "$QLOG" 2>/dev/null && return 0
  grep -q -e "ZZW30ENDZZ" -e "ZZW30PULLFAILZZ" -e "ZZW30TIMEOUTZZ" -e "ZZW30PULLNOFILESZZ" "$WLOG" 2>/dev/null && return 0
  return 1
}
queue_pending() {    # true iff that queue has been told its checkpoint pair is stable (or has started) and is not terminal
  queue_terminal && return 1
  grep -q "ZZW30STABLEZZ" "$WLOG" 2>/dev/null && return 0
  grep -q "ZZQQUEUESTARTZZ" "$QLOG" 2>/dev/null && return 0
  return 1
}
# ---- one battery stage ---------------------------------------------------------------------------------------------------------------------------- #
battery_stage() (   # $1 = step ; $2 = hard|soft (hard: wait for the queue's terminal marker first; soft: only while it is pending)
  N=$1; MODE=$2; TAG=r7step${N}_unc; JOB=refcv7-matched-step$N
  if pass_done "$TAG"; then mark BATTHAVE "$TAG"; exit 0; fi
  verify_ckpt "$N" || { mark CKPTBAD "ckpt_$N.pt md5 != MD5SUMS"; exit 4; }
  n=0
  if [ "$MODE" = "hard" ]; then
    while ! queue_terminal; do n=$((n+1)); [ $((n % 24)) -eq 1 ] && mark WAITQUEUE "N=$N waits for the refcv8 step-30,000 queue's terminal marker"; [ "$n" -ge 864 ] && { mark WAITQUEUETIMEOUT "N=$N: 72 h"; exit 4; }; sleep 300; done
    mark QUEUETERMINAL "N=$N proceeds"
  else
    while queue_pending; do n=$((n+1)); [ $((n % 15)) -eq 1 ] && mark YIELD "N=$N yields to the refcv8 step-30,000 queue"; [ "$n" -ge 1440 ] && { mark YIELDTIMEOUT "N=$N: 48 h"; exit 4; }; sleep 120; done
  fi
  mark BATTSTART "$TAG ($PATHV)"
  gpu_acquire "$JOB" || exit 3
  STOPW="$KIT/chain/r7m_gpu_watch_$JOB.stop"; rm -f "$STOPW"
  "$PY" "$BAT/gpu_watch.py" --out "$KITW/chain/r7m_gpu_watch_$JOB.jsonl" --stop-file "$KITW/chain/r7m_gpu_watch_$JOB.stop" >> "$KIT/chain/r7m_lockwait_$JOB.log" 2>&1 &
  trap ': > "$STOPW"; gpu_release "$JOB"' EXIT
  trap 'exit 143' TERM INT HUP
  r7_pass "$TAG" "$CKPT_DIR/ckpt_$N.pt"
  if ! pass_done "$TAG"; then mark BATTRETRY "$TAG incomplete; retrying once (resumes from complete seed dumps)"; r7_pass "$TAG" "$CKPT_DIR/ckpt_$N.pt"; fi
  if pass_done "$TAG"; then
    "$PY" -I -X utf8 "$PKGW/code/r7m_stamp.py" --tree "$ARCW/$TAG" --out "$RAWW/BATTERY_STAMP_$TAG.json" --ceiling-filter-off "${R7M_CEIL_OFF:-0}" >> "$LOG" 2>&1
    if [ -s "$RAWL/BATTERY_STAMP_$TAG.json" ]; then mark BATTDONE "$TAG banked: $RAWW/BATTERY_STAMP_$TAG.json"; else mark BATTDONENOSTAMP "$TAG complete, stamp not written"; fi
  else mark BATTFAIL "$TAG: no complete battery_summary (see chain/r7m_run_$TAG.log)"; exit 4; fi
)
battery_stage 20000 soft || mark STAGEFAILED "20000 (continuing)"
battery_stage 30000 soft || mark STAGEFAILED "30000 (continuing)"
battery_stage 15000 hard || mark STAGEFAILED "15000"
mark CHAINEND "all stages attempted"
note "chain end"

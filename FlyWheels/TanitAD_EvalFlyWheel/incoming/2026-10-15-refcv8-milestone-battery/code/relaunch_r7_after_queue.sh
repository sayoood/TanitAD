#!/bin/sh
# relaunch_r7_after_queue.sh -- lock ORDER courtesy (2026-10-18 ~17:45): the SR-TTC S0 dump (another session) asked to take the
#   dev-box GPU lock right after the refcv8 step-30,000 queue ends and BEFORE refcv7-matched-step15000 / refcv7-ceilon-step50400.
#   Those two waiters were stopped (explicit PIDs) before they held anything; this restarts them GRACE seconds after the queue's
#   terminal marker, so S0 (if it is polling) holds the lock first and the refcv7 jobs then wait on it the normal way.  If S0 has
#   not taken the lock by then, the refcv7 jobs simply go first.  No lock is ever touched here.  Both chains skip finished work
#   (pass_done / HAVE), so a restart re-runs nothing that is complete.  Markers ZZRL...ZZ in chain/relaunch_r7.log.
KIT=/e/refcv8_eval_kit; P=/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code
LOG=$KIT/chain/relaunch_r7.log; QLOG=$KIT/chain/queue_30000_r8.log; WLOG=$KIT/chain/wait_30000_r8.log
GRACE="${RL_GRACE_S:-1200}"
mk() { echo "ZZRL$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mk START "pid $$ grace ${GRACE}s"
n=0
until grep -q "ZZQQUEUEENDZZ" "$QLOG" 2>/dev/null || grep -q -e "ZZW30PULLFAILZZ" -e "ZZW30TIMEOUTZZ" -e "ZZW30PULLNOFILESZZ" "$WLOG" 2>/dev/null; do
  n=$((n+1)); [ $n -gt 1440 ] && { mk END "TIMEOUT (72 h) waiting for the refcv8 30k queue"; exit 3; }
  [ $((n % 20)) -eq 1 ] && mk WAIT "queue not terminal"
  sleep 180
done
mk QEND "queue terminal; grace ${GRACE}s for S0"
sleep "$GRACE"
R7M_PATH=ceiloff nohup sh "$P/chain_r7_matched_phase.sh" < /dev/null > "$KIT/chain/chain_r7_matched_phase.relaunch.out" 2>&1 &
mk RELAUNCH "chain_r7_matched_phase.sh (ceiloff) pid $!"
nohup sh "$P/chain_r7on_50400.sh" < /dev/null > /c/lgs/chain_r7on_50400.relaunch.out 2>&1 &
mk RELAUNCH "chain_r7on_50400.sh pid $!"
mk END "done"

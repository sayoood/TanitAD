#!/bin/sh
# chain_r7on_50400.sh -- the CEILING-ON re-roll of refcv7-50,400 (both inference seeds, full battery), eval only, dev-box GPU.
#   Why (AMENDMENT A-R7M-1, 0cfa7c6): every refcv8 battery rolled `os` with the max-speed ceiling ON, while the banked refcv7-50,400
#   reference was rolled by a deleted tree whose ceiling never acted (4,754 / 4,754).  NR_MATCHED_INSTRUMENT.md re-read BAR-R8-NR on
#   the matched OFF instrument -- but only at inference seed 0 (seed-1 refcv8 rolls carry no filter-off arm).  ONE ceiling-ON roll of
#   refcv7-50,400 makes every refcv8 battery (5k..20k banked, 30k, final) matched at BOTH seeds on the deployed instrument, with no
#   refcv8 re-roll.  Path = `run_battery_r7m.py` with R7M_CEIL_OFF UNSET (= the launch tree as built; the control smoke
#   r7ctl_step50400_smoke ran exactly this path).  Tag r7on_step50400_unc -> C:/Users/Admin/refcv8_eval_archive/battery/.
#   Order: waits until the refcv8 step-30,000 queue has written its terminal marker (it never delays refcv8 results), then the
#   shared GPU lock (never breaks a foreign lock).  Assert on ARTIFACTS; markers ZZR7ON...ZZ in chain/chain_r7on_50400.log.
. "$(dirname "$0")/r7m_common.sh"
unset R7M_CEIL_OFF R7M_E9_NO_CEIL
LOG="$KIT/chain/chain_r7on_50400.log"
mark() { echo "ZZR7ON$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
TAG=r7on_step50400_unc; JOB=refcv7-ceilon-step50400
mark START "pid $$"
if pass_done "$TAG"; then mark HAVE "$ARCW/$TAG complete"; mark END "have"; exit 0; fi
verify_ckpt 50400 || { mark CKPTBAD "ckpt_50400.pt md5 != MD5SUMS"; mark END "ABORT"; exit 2; }
n=0
until grep -q -e "ZZQQUEUEENDZZ" "$QLOG" 2>/dev/null || grep -q -e "ZZW30PULLFAILZZ" -e "ZZW30TIMEOUTZZ" "$WLOG" 2>/dev/null; do
  n=$((n+1)); [ $n -gt 720 ] && { mark END "TIMEOUT waiting for the refcv8 30k queue"; exit 3; }
  [ $((n % 20)) -eq 1 ] && mark WAITQ "refcv8 30k queue not finished"
  sleep 180
done
mark QDONE "refcv8 30k queue finished (or its waiter gave up)"
gpu_acquire "$JOB" || { mark END "ABORT no lock"; exit 3; }
STOPW="$KIT/chain/r7m_gpu_watch_$JOB.stop"; rm -f "$STOPW"
"$PY" "$BAT/gpu_watch.py" --out "$KITW/chain/r7m_gpu_watch_$JOB.jsonl" --stop-file "$KITW/chain/r7m_gpu_watch_$JOB.stop" >> "$KIT/chain/r7m_lockwait_$JOB.log" 2>&1 &
trap ': > "$STOPW"; gpu_release "$JOB"' EXIT
mark ROLL "$TAG (ceiling ON, both seeds)"
r7_pass "$TAG" "$CKPT_DIR/ckpt_50400.pt"
if ! pass_done "$TAG"; then mark RETRY "incomplete, resuming from complete seed dumps"; r7_pass "$TAG" "$CKPT_DIR/ckpt_50400.pt"; fi
if pass_done "$TAG"; then mark DONE "$ARCW/$TAG"; else mark FAIL "no complete battery_summary (chain/r7m_run_$TAG.log)"; fi
mark END "done"

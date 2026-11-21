#!/bin/bash
# watch_l40.sh -- poll the restage_v2 driver on the L40 pod every 5 min over ONE ssh call per poll (no model turns).
# Pod address comes from C:/Users/Admin/freept/pod_direct.txt via pd.sh; nothing pod-specific is written here.
# Run:   bash C:/Users/Admin/par18/restage_v2/watch_l40.sh                (foreground; or under Monitor / run_in_background)
#        WATCH_PHASE=guard bash .../watch_l40.sh                          (re-run after WATCH_CHAIN_LAUNCHED: ignore the launch line, wait only for the guard)
# Knobs: WATCH_INTERVAL (s, default 300), WATCH_MAX_POLLS (default 400 = 33 h), WATCH_LOG (local progress log).
#
# EXIT LINES (first token is the distinct marker; exit code in brackets):
#   WATCH_DRIVER_FAILED <reason>        [2]  $O/FAILED exists or driver.log has a ZZRS_FAIL line (the driver stopped itself)
#   WATCH_DRIVER_DEAD <last log line>   [2]  driver pid gone, no FAILED, no final-chain exit (killed / pod restarted)
#   WATCH_CHAIN_LAUNCHED <line>         [0]  chain.log has "train.py preflight OK; launching" AFTER the driver logged ZZRS_FINAL_CHAIN launching
#   WATCH_GUARD_OK <line>               [0]  guard.log ZZGUARD_OK (first 5 labeller events: no_parts == 0)
#   WATCH_ALERT_COVERAGE <line>         [3]  ALERT_COVERAGE.txt / ZZGUARD_ALERT (labeller no_parts != 0; training is NOT killed by the guard)
#   WATCH_GUARD_TIMEOUT <line>          [3]  ZZGUARD_TIMEOUT (no labeller event in 4 h: nothing judged)
#   WATCH_ALERT_ARGS <line>             [3]  ALERT_ARGS.txt (ARGS sha differs from 570619873b1ebaa0)
#   WATCH_CHAIN_EXITED <line>           [2]  ZZRS_FINAL_CHAIN_EXIT before a launch / guard verdict
#   WATCH_POD_UNREACHABLE               [4]  3 consecutive polls could not run the remote status command
#   WATCH_MAX_POLLS                     [5]  gave up (nothing decisive happened)
# Priority within one poll: FAILED > ALERT_ARGS > GUARD result > CHAIN_LAUNCHED > CHAIN_EXITED > DRIVER_DEAD.
INTERVAL="${WATCH_INTERVAL:-300}"; MAXP="${WATCH_MAX_POLLS:-400}"; PHASE="${WATCH_PHASE:-launch}"
LOG="${WATCH_LOG:-C:/Users/Admin/par18/restage_v2/watch_l40.progress.log}"
PD="${WATCH_PD:-C:/Users/Admin/par18/pd.sh}"; RETRY="${WATCH_RETRY_S:-60}"

REMOTE='
O=/workspace/ops/restage_v2; P6=/workspace/data/refe_p6
echo "R_OK=1"
echo "R_T=$(date -u +%H:%M:%S)"
if [ -f $O/FAILED ]; then echo "R_FAILED=$(head -c 300 $O/FAILED | tr "\n" " ")"; fi
echo "R_ZZFAIL=$(n=$(grep -a -n "driver start pid" $O/driver.log 2>/dev/null | tail -n 1 | cut -d: -f1); tail -n +${n:-1} $O/driver.log 2>/dev/null | grep -a -E "^[0-9T:Z-]+ ZZRS_FAIL " | tail -n 1 | cut -c1-300)"
echo "R_OKS=$(grep -a -c -E "^[0-9T:Z-]+ ZZRS_OK " $O/driver.log 2>/dev/null)"
echo "R_LAST=$(tail -n 1 $O/driver.log 2>/dev/null | cut -c1-200)"
pid=$(sed -n "s/.*driver start pid \([0-9][0-9]*\).*/\1/p" $O/driver.log 2>/dev/null | tail -n 1)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "R_ALIVE=1"; else echo "R_ALIVE=0"; fi
echo "R_FCL=$(grep -a -c -E "^[0-9T:Z-]+ ZZRS_FINAL_CHAIN launching" $O/driver.log 2>/dev/null)"
echo "R_FCE=$(grep -a -E "^[0-9T:Z-]+ ZZRS_FINAL_CHAIN_EXIT " $O/driver.log 2>/dev/null | tail -n 1 | cut -c1-250)"
echo "R_LAUNCH=$(grep -a -E "train.py preflight OK; launching" $P6/chain.log 2>/dev/null | tail -n 1 | cut -c1-200)"
echo "R_GOK=$(grep -a "^ZZGUARD_OK" $O/guard.log 2>/dev/null | tail -n 1 | cut -c1-300)"
echo "R_GAL=$(grep -a "^ZZGUARD_ALERT" $O/guard.log 2>/dev/null | tail -n 1 | cut -c1-300)"
echo "R_GTO=$(grep -a "^ZZGUARD_TIMEOUT" $O/guard.log 2>/dev/null | tail -n 1 | cut -c1-300)"
if [ -f $O/ALERT_COVERAGE.txt ]; then echo "R_ACOV=$(head -c 300 $O/ALERT_COVERAGE.txt | tr "\n" " ")"; fi
if [ -f $O/ALERT_ARGS.txt ]; then echo "R_AARGS=$(head -c 300 $O/ALERT_ARGS.txt | tr "\n" " ")"; fi
echo "R_ARGSOK=$(grep -a -E "^[0-9T:Z-]+ ZZRS_ARGS_OK" $O/driver.log 2>/dev/null | tail -n 1 | cut -c1-200)"
'

fails=0; n=0
v() { printf '%s\n' "$OUT" | sed -n "s/^$1=//p" | head -n 1; }   # value of an R_ key
say() { echo "$1"; echo "$(date -u +%FT%TZ) $1" >> "$LOG"; }
while [ $n -lt $MAXP ]; do
  n=$((n + 1))
  OUT="$(printf '%s' "$REMOTE" | bash $PD 90 2>/dev/null)"
  if [ "$(v R_OK)" != 1 ]; then
    fails=$((fails + 1)); echo "$(date -u +%FT%TZ) poll $n: remote status failed ($fails/3)" >> "$LOG"
    if [ $fails -ge 3 ]; then say "WATCH_POD_UNREACHABLE 3 consecutive failed polls (poll $n)"; exit 4; fi
    sleep "$RETRY"; continue          # retry sooner after a failed poll; three in a row end the watch
  fi
  fails=0
  echo "$(date -u +%FT%TZ) poll $n: pod $(v R_T)Z steps_ok=$(v R_OKS) alive=$(v R_ALIVE) last: $(v R_LAST)" >> "$LOG"
  F="$(v R_FAILED)"; [ -z "$F" ] && F="$(v R_ZZFAIL)"
  [ -n "$F" ] && { say "WATCH_DRIVER_FAILED $F"; exit 2; }
  A="$(v R_AARGS)";  [ -n "$A" ] && { say "WATCH_ALERT_ARGS $A"; exit 3; }
  G="$(v R_GOK)";    [ -n "$G" ] && { say "WATCH_GUARD_OK $G (args: $(v R_ARGSOK))"; exit 0; }
  G="$(v R_ACOV)";   [ -z "$G" ] && G="$(v R_GAL)"; [ -n "$G" ] && { say "WATCH_ALERT_COVERAGE $G"; exit 3; }
  G="$(v R_GTO)";    [ -n "$G" ] && { say "WATCH_GUARD_TIMEOUT $G"; exit 3; }
  L="$(v R_LAUNCH)"
  if [ "$PHASE" != guard ] && [ -n "$L" ] && [ "$(v R_FCL)" -ge 1 ] 2>/dev/null; then say "WATCH_CHAIN_LAUNCHED $L (args: $(v R_ARGSOK); re-run with WATCH_PHASE=guard for the coverage guard)"; exit 0; fi
  E="$(v R_FCE)";    [ -n "$E" ] && { say "WATCH_CHAIN_EXITED $E"; exit 2; }
  if [ "$(v R_ALIVE)" = 0 ] && [ "$(v R_FCL)" = 0 ]; then say "WATCH_DRIVER_DEAD pid not running, no FAILED marker; last: $(v R_LAST)"; exit 2; fi
  sleep "$INTERVAL"
done
say "WATCH_MAX_POLLS $MAXP polls without a decisive line"; exit 5

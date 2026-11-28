#!/bin/bash
# watch_e.sh -- poll the Amendment (e) relaunch every WATCH_INTERVAL s, ONE ssh call per poll (no model turns).  Address via pd.sh (nothing pod-specific here).
# WATCH_PHASE=launch (default): exit 0 on "train.py preflight OK; launching" after the relaunch's ARGS line; =guard: wait only for the coverage guard's verdict.
# EXITS: WATCH_FAILED [2] relaunch FAILED / chain REFUSED / chain exit | WATCH_LAUNCHED [0] | WATCH_GUARD_OK [0] | WATCH_ALERT [3] coverage alert / guard timeout | WATCH_UNREACHABLE [4]
INTERVAL="${WATCH_INTERVAL:-240}"; MAXP="${WATCH_MAX_POLLS:-300}"; PHASE="${WATCH_PHASE:-launch}"; PD=C:/Users/Admin/par18/pd.sh
LOG=C:/Users/Admin/par18/relaunch_e/watch_e.progress.log
REMOTE='
O=/workspace/ops/relaunch_e; P6=/workspace/data/refe_p6
echo "R_OK=1"; echo "R_T=$(date -u +%H:%M:%S)"
[ -f $O/FAILED ] && echo "R_FAILED=$(head -c 300 $O/FAILED | tr "\n" " ")"
n=$(grep -a -n "ARGS sha f74aee47cf0b708f" $P6/chain.log | head -n 1 | cut -d: -f1)
echo "R_REF=$(tail -n +${n:-999999999} $P6/chain.log | grep -a "REFUSED:" | tail -n 1 | cut -c1-300)"
echo "R_LAUNCH=$(tail -n +${n:-999999999} $P6/chain.log | grep -a "train.py preflight OK; launching" | tail -n 1 | cut -c1-200)"
echo "R_GATE=$(tail -n +${n:-999999999} $P6/chain.log | grep -a "launch gate PASS" | tail -n 1 | cut -c1-200)"
echo "R_LAST=$(tail -n 1 $P6/chain.log | cut -c1-200)"
echo "R_EXIT=$(grep -a ZZRE_CHAIN_EXIT $O/relaunch_e.log | tail -n 1 | cut -c1-250)"
echo "R_GOK=$(grep -a "^ZZGUARD_OK" $O/guard.log 2>/dev/null | tail -n 1 | cut -c1-300)"
echo "R_GAL=$(grep -a -E "^ZZGUARD_(ALERT|TIMEOUT)" $O/guard.log 2>/dev/null | tail -n 1 | cut -c1-300)"
[ -f $O/ALERT_COVERAGE.txt ] && echo "R_ACOV=$(head -c 300 $O/ALERT_COVERAGE.txt | tr "\n" " ")"
'
v() { printf '%s\n' "$OUT" | sed -n "s/^$1=//p" | head -n 1; }
say() { echo "$1"; echo "$(date -u +%FT%TZ) $1" >> "$LOG"; }
fails=0; n=0
while [ $n -lt $MAXP ]; do
  n=$((n + 1)); OUT="$(printf '%s' "$REMOTE" | bash $PD 90 2>/dev/null)"
  if [ "$(v R_OK)" != 1 ]; then fails=$((fails + 1)); [ $fails -ge 3 ] && { say "WATCH_UNREACHABLE poll $n"; exit 4; }; sleep 60; continue; fi
  fails=0; echo "$(date -u +%FT%TZ) poll $n: $(v R_T)Z last: $(v R_LAST)" >> "$LOG"
  F="$(v R_FAILED)"; [ -z "$F" ] && F="$(v R_REF)"; [ -n "$F" ] && { say "WATCH_FAILED $F"; exit 2; }
  G="$(v R_GOK)"; [ -n "$G" ] && { say "WATCH_GUARD_OK $G"; exit 0; }
  G="$(v R_ACOV)"; [ -z "$G" ] && G="$(v R_GAL)"; [ -n "$G" ] && { say "WATCH_ALERT $G"; exit 3; }
  L="$(v R_LAUNCH)"; [ "$PHASE" != guard ] && [ -n "$L" ] && { say "WATCH_LAUNCHED $L | $(v R_GATE)"; exit 0; }
  E="$(v R_EXIT)"; [ -n "$E" ] && { say "WATCH_FAILED $E"; exit 2; }
  sleep "$INTERVAL"
done
say "WATCH_MAX_POLLS"; exit 5

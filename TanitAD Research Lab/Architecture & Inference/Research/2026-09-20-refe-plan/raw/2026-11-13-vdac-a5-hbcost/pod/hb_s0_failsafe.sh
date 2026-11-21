#!/usr/bin/env bash
# Failsafe for hb_s0_grace.sh: at 14:00Z, if the HB-s0 timeout process is still stopped, SIGCONT it (so the pause script can never hang).
TPID=249827; WANT="timeout 36000 bash /workspace/refe-plan-vdac/refe/vdac_chain.sh HB-s0"; L=/workspace/data/refe_vdac/hb_s0_grace.log
sleep $(( $(date -u -d '2026-11-13T14:00:00Z' +%s) - $(date +%s) ))
s=$(ps -o stat= -p $TPID 2>/dev/null | tr -d ' ')
case "$s" in T*) [ "$(ps -o args= -p $TPID 2>/dev/null)" = "$WANT" ] && kill -CONT $TPID && echo "$(date -u +%FT%TZ) FAILSAFE SIGCONT $TPID" >> $L ;; esac

#!/usr/bin/env bash
# HB-s0 grace (MM, 2026-11-13 ~05:50Z). HB-s0 trains at 3.24 s/update (batch 4 x accum 2, the card's 9.2 fallback);
# projected ZZVDAC_CHAIN_DONE ~12:48Z against the pause timeout (36,000 s from 02:49:20Z = 12:49:20Z).
# If the chain is still running at 12:46:00Z AND the trainer has logged update >= 9000, SIGSTOP the `timeout`
# process (its SIGALRM then waits) and SIGCONT it when the chain exits, or at the hard ceiling 13:49:20Z (+1 h),
# whichever is first. At the ceiling the timeout fires as written (the card's NOT RUN). The pause script decides
# by grepping the chain log for "ZZVDAC_CHAIN_DONE HB-s0", not by rc (vdac_pause_p7.sh:64-65).
OUT=/workspace/data/refe_vdac; L=$OUT/hb_s0_grace.log
TPID=249827; WANT="timeout 36000 bash /workspace/refe-plan-vdac/refe/vdac_chain.sh HB-s0"
say() { echo "$(date -u +%FT%TZ) $*" >> $L; }
alive() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' '); [ -n "$s" ] && [ "${s#Z}" = "$s" ]; }
STOPPED=0
cont() { if [ "$STOPPED" = 1 ]; then kill -CONT $TPID 2>/dev/null; say "SIGCONT timeout $TPID ($1)"; STOPPED=0; fi; }
trap 'cont exit' EXIT
trap 'cont signal; exit 1' TERM INT HUP
[ "$(ps -o args= -p $TPID 2>/dev/null)" = "$WANT" ] || { say "REFUSE: pid $TPID is not '$WANT'"; exit 1; }
CPID=$(ps -eo pid=,ppid=,args= | awk -v p=$TPID '$2==p && $3=="bash" && $4=="/workspace/refe-plan-vdac/refe/vdac_chain.sh" {print $1}' | head -1)
[ -n "$CPID" ] || { say "REFUSE: no chain child of $TPID"; exit 1; }
ACT=$(date -u -d '2026-11-13T12:46:00Z' +%s); CEIL=$(date -u -d '2026-11-13T13:49:20Z' +%s)
say "armed: timeout $TPID, chain $CPID; acts at 12:46:00Z if the chain is still running and update >= 9000; ceiling 13:49:20Z"
done_() { grep -q "ZZVDAC_CHAIN_DONE HB-s0" $OUT/chain_hb-s0.log 2>/dev/null && echo yes || echo no; }
while [ "$(date +%s)" -lt "$ACT" ]; do
  alive $CPID || { say "chain exited before 12:46Z (CHAIN_DONE $(done_)): no action"; say "ZZHB_GRACE_END"; exit 0; }
  sleep 20
done
alive $CPID || { say "chain exited (CHAIN_DONE $(done_)): no action"; say "ZZHB_GRACE_END"; exit 0; }
U=$(grep '"event": "train"' $OUT/hb-s0/sft.jsonl | tail -1 | grep -oE '"update": [0-9]+' | grep -oE '[0-9]+$')
if [ "${U:-0}" -lt 9000 ]; then say "update ${U:-?} < 9000 at 12:46Z: no grace, the timeout fires as written"; say "ZZHB_GRACE_END"; exit 0; fi
[ "$(ps -o args= -p $TPID 2>/dev/null)" = "$WANT" ] || { say "REFUSE at act: pid $TPID changed"; exit 1; }
kill -STOP $TPID && STOPPED=1 && say "SIGSTOP timeout $TPID at update $U; chain $CPID still running"
while alive $CPID && [ "$(date +%s)" -lt "$CEIL" ]; do sleep 10; done
if alive $CPID; then say "ceiling 13:49:20Z reached with the chain running: releasing the timeout, it fires as written"; else say "chain exited (CHAIN_DONE $(done_))"; fi
cont release
say "ZZHB_GRACE_END"

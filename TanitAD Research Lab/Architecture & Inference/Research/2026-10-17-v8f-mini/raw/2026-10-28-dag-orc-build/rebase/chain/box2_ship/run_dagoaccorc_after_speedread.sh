#!/usr/bin/env bash
# PI go 2026-10-27, verbatim: "Yes, queue DAG-ORC on box 2 after the SPEEDREAD pull, through its launch gate."
# Starts chain_v8f_dagoaccorc.sh (Amendment 1: DAG-ORC re-based on DAG-OACC; its own gates, guard and launch-gate token) once the SPEEDREAD pull has ENDED (ZZSPEEDREAD-PULL-DONE
# or -STOP; the pull's trap releases BOX2_BUSY.txt either way) and BOX2_BUSY.txt is gone. AFTER_OUT = the DAG-OACC r2 log (already DONE).
P=/f/TanitAD-scratch/speedread/pull.out; B=/c/Users/Admin/box2_ship; W=$B/logs/dagoaccorc_waiter.out
echo "$(date '+%m-%d %H:%M:%S') waiter start: waiting for the SPEEDREAD pull's end and no BOX2_BUSY.txt" >> $W
for i in $(seq 1 720); do
  grep -aq 'ZZSPEEDREAD-PULL-DONE\|ZZSPEEDREAD-PULL-STOP' $P 2>/dev/null && [ ! -f $B/BOX2_BUSY.txt ] && break
  sleep 60
done
grep -aq 'ZZSPEEDREAD-PULL-DONE\|ZZSPEEDREAD-PULL-STOP' $P 2>/dev/null && [ ! -f $B/BOX2_BUSY.txt ] \
  || { echo "$(date '+%m-%d %H:%M:%S') ZZDAGOACCORC-WAITER-STOP the pull did not end / BOX2_BUSY still held after 12 h" >> $W; exit 3; }
echo "$(date '+%m-%d %H:%M:%S') waiter GO: $(grep -a 'ZZSPEEDREAD-PULL-' $P | tail -1)" >> $W
# This chain queues A and REP only (~2.8 h, within the PI's ~3 h); B (p 0.1) is NOT queued (Amendment 1 section 4): it is trained only on A's HARMS, with its own PI go.
AFTER_OUT=C:/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out bash $B/chain_v8f_dagoaccorc.sh < /dev/null > $B/logs/chain_v8f_dagoaccorc.out 2>&1
echo "$(date '+%m-%d %H:%M:%S') chain_v8f_dagoaccorc.sh rc=$? ($(tail -1 $B/logs/chain_v8f_dagoaccorc.out | cut -c1-200))" >> $W

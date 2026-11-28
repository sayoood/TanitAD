#!/bin/bash
# DSCALE10K block-0 pilot launcher. Waits for the STRAT1 S10 chain to end and the GPU to drop below 3,500 MiB, then runs pilot_run.py. Never starts block 1.
LOG=/root/strat1/logs/s10.log
until tail -c 400 "$LOG" 2>/dev/null | grep -qE "ZZ STRAT1 S10 (DONE|FAIL) ZZ"; do sleep 30; done
echo "$(date -u +%FT%TZ) strat1 s10 sentinel seen: $(tail -c 400 $LOG | grep -oE 'ZZ STRAT1 S10 (DONE|FAIL) ZZ' | tail -1)"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" -lt 3500 ]; do sleep 30; done
echo "$(date -u +%FT%TZ) gpu below 3500 MiB, starting pilot_run.py"
cd /root/dscale10k
/root/venv/bin/python /root/dscale10k/probe/pilot_run.py
rc=$?
[ $rc -eq 0 ] || echo "ZZ DSCALE10K PILOT FAIL ZZ pilot_run.py rc $rc (see PILOT_REPORT.json why, or probe/*.log)"

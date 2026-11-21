#!/bin/bash
# start_early.sh -- MM: start sequencer v2, stop v1 (its EXIT trap removes the marker), re-set the marker, start the early N1DM chain. Idempotent checks by /proc.
M=/root/N1DW_READY; N=/root/strat1/n1dm
grep -q "n1dm_after_rhk2[.]sh" /proc/[0-9]*/cmdline 2>/dev/null || { nohup setsid bash $N/n1dm_after_rhk2.sh < /dev/null > /dev/null 2>&1 & sleep 3; }
for p in $(grep -l "n1dm_after_rhk[.]sh" /proc/[0-9]*/cmdline 2>/dev/null | cut -d/ -f3); do kill $p; done
for i in $(seq 1 20); do grep -q "n1dm_after_rhk[.]sh" /proc/[0-9]*/cmdline 2>/dev/null || break; sleep 3; done; touch $M     # v1 handles SIGTERM after its sleep, then its EXIT trap removes the marker: wait for it to go, then re-set
grep -q "n1dm_chain_early[.]sh" /proc/[0-9]*/cmdline 2>/dev/null && { echo "early chain already running"; exit 1; }
nohup setsid bash $N/n1dm_chain_early.sh < /dev/null > /dev/null 2>&1 &
sleep 3; echo "started"

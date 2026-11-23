#!/bin/bash
# launch_rk.sh -- REVISION 2 (commit 1a8b11a1, Amendment 1). MM only, AFTER the PI go with its cost is in, the kit is shipped and verified (SHA256_RK2.txt), and PREP_RK2.md reads all-green (tree, both label caches, G-SUITE at 1a8b11a1).
# Runs from the kit folder /root/rollhist/kit_rk. Same shape as launch_ra.sh: marker watcher first, then the chain. Refuses if any chain runs.
# The chain waits (cap 12 h) for the ADD123 chain's last line, then VALIDATES (seconds, before the lock) the tree, the G-SUITE result, both caches and the argv, then waits on /root/gpu_queue.lock (flock) and holds it to the end.
# Same RH_READOUT as the RH chain: causal_g1x. `DRY=1 bash /root/rollhist/kit_rk/rh_rk_chain.sh` is the writes-nothing walk of the same validations.
K=$(cd "$(dirname "$0")" && pwd)
ps -eo args | grep -q "[r]h_rk_chain.sh" && { echo "an rh_rk_chain.sh is running; refusing"; exit 1; }
ps -eo args | grep -q "[r]h_ra_chain.sh" && { echo "an rh_ra_chain.sh is running; refusing"; exit 1; }
ps -eo args | grep -q "[r]h_rl_chain.sh" && { echo "an rh_rl_chain.sh is running; refusing"; exit 1; }
ps -eo args | grep -q "[r]h_chain.sh" && { echo "an rh_chain.sh is running; refusing"; exit 1; }
ps -eo args | grep -q "[r]h_rk_marker_watch.sh" && { echo "an rh_rk_marker_watch.sh is running; refusing"; exit 1; }
nohup setsid bash $K/rh_rk_marker_watch.sh < /dev/null > /dev/null 2>&1 &
sleep 1
RH_READOUT=causal_g1x nohup setsid bash $K/rh_rk_chain.sh < /dev/null > /dev/null 2>&1 &
echo "ZZ RK LAUNCHED ZZ"

#!/bin/bash
# launch_ki.sh -- MM only, AFTER the PI go with its cost is in, the kit is shipped and verified (SHA256_KI.txt), and PREP_KI.md reads all-green (tree, G-SUITE at 4097a89e, dry mint).
# Runs from the kit folder /root/rollhist/kit_ki. Same shape as launch_rk.sh: marker watcher first, then the chain. REFUSES if ANY rh_*_chain.sh or rh_*_marker_watch.sh process runs (any ROLLHIST chain, any watcher).
# The chain waits for no other chain: it VALIDATES (seconds, before the lock) the kit, the tree, the G-SUITE result, the argv and the gate-record inputs, then waits on /root/gpu_queue.lock (flock) and holds it to the end.
# Same RH_READOUT as the RH chain: causal_g1x. `DRY=1 bash /root/rollhist/kit_ki/rh_ki_chain.sh` is the writes-nothing walk of the same validations.
K=$(cd "$(dirname "$0")" && pwd)
ps -eo args | grep -Eq "[r]h_([a-z0-9]+_)*chain[.]sh" && { echo "an rh_*chain.sh is running; refusing"; ps -eo args | grep -E "[r]h_([a-z0-9]+_)*chain[.]sh" | head -3; exit 1; }
ps -eo args | grep -Eq "[r]h_([a-z0-9]+_)*marker_watch[.]sh" && { echo "an rh_*marker_watch.sh is running; refusing"; ps -eo args | grep -E "[r]h_([a-z0-9]+_)*marker_watch[.]sh" | head -3; exit 1; }
nohup setsid bash $K/rh_ki_marker_watch.sh < /dev/null > /dev/null 2>&1 &
sleep 1
RH_READOUT=causal_g1x nohup setsid bash $K/rh_ki_chain.sh < /dev/null > /dev/null 2>&1 &
echo "ZZ KI LAUNCHED ZZ"

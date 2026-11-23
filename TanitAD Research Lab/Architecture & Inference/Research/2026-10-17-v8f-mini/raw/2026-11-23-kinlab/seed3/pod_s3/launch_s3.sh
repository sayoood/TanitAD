#!/bin/bash
# launch_s3.sh -- MM only, AFTER the PI go with its cost is in, the kit is shipped and verified (SHA256_S3.txt, SHARED_S3.sha256), and PREP_S3 reads all-green (trees, dry mint, DRY walk).
# Runs from the kit folder /root/rollhist/kit_s3. Same shape as launch_ki.sh: marker watcher first, then the chain. REFUSES if ANY rh_*chain.sh or rh_*marker_watch.sh process runs (any ROLLHIST chain, any watcher).
# The chain waits for no other chain: it VALIDATES (seconds, before the lock) the kit, the shared pod files, both trees, both G-SUITE results, UE_s2.A and the argv, then takes flock on /root/gpu_queue.lock and holds it to the end.
# Readout causal_g1x (the RH bank and arms). `DRY=1 bash /root/rollhist/kit_s3/rh_s3_chain.sh` is the writes-nothing walk of the same validations.
K=$(cd "$(dirname "$0")" && pwd)
ps -eo args | grep -Eq "[r]h_([a-z0-9]+_)*chain[.]sh" && { echo "an rh_*chain.sh is running; refusing"; ps -eo args | grep -E "[r]h_([a-z0-9]+_)*chain[.]sh" | head -3; exit 1; }
ps -eo args | grep -Eq "[r]h_([a-z0-9]+_)*marker_watch[.]sh" && { echo "an rh_*marker_watch.sh is running; refusing"; ps -eo args | grep -E "[r]h_([a-z0-9]+_)*marker_watch[.]sh" | head -3; exit 1; }
nohup setsid bash $K/rh_s3_marker_watch.sh < /dev/null > /dev/null 2>&1 &
sleep 1
RH_READOUT=causal_g1x nohup setsid bash $K/rh_s3_chain.sh < /dev/null > /dev/null 2>&1 &
echo "ZZ S3 LAUNCHED ZZ"

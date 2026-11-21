#!/bin/bash
# launch_p03.sh -- MM only, AFTER the dry mint (DRY_MINT_PLAN.md) passed and Amendment 7 is frozen. Same shape as relaunch_rh.sh: marker watcher first, then the chain. Refuses if either chain runs.
# The chain waits on /root/gpu_queue.lock (flock), so it starts when the current lock holder releases it; it holds the lock to the end. Same RH_READOUT as the RH chain: causal_g1x.
ps -eo args | grep -q "[r]h_p03_chain.sh" && { echo "an rh_p03_chain.sh is running; refusing"; exit 1; }
ps -eo args | grep -q "[r]h_chain.sh" && { echo "an rh_chain.sh is running; refusing"; exit 1; }
nohup setsid bash /root/rollhist/rh_p03_marker_watch.sh < /dev/null > /dev/null 2>&1 &
sleep 1
RH_READOUT=causal_g1x nohup setsid bash /root/rollhist/rh_p03_chain.sh < /dev/null > /dev/null 2>&1 &
echo "ZZ P03 LAUNCHED ZZ"

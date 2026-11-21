#!/bin/bash
# launch_k.sh -- MM only, AFTER the dry mint (DRY_MINT_PLAN.md) passed and Amendment 8 is frozen. Same shape as relaunch_rh.sh: marker watcher first, then the chain. Refuses if either chain runs.
# The chain waits on /root/gpu_queue.lock (flock), so it starts when the current lock holder releases it; it holds the lock to the end. RH_READOUT=causal_g0x (readout K; the RH chain's causal_g1x is not touched).
ps -eo args | grep -q "[r]h_k_chain.sh" && { echo "an rh_k_chain.sh is running; refusing"; exit 1; }
ps -eo args | grep -q "[r]h_chain.sh" && { echo "an rh_chain.sh is running; refusing"; exit 1; }
ps -eo args | grep -q "[r]h_p03_chain.sh" && { echo "an rh_p03_chain.sh is running; refusing"; exit 1; }
nohup setsid bash /root/rollhist/rh_k_marker_watch.sh < /dev/null > /dev/null 2>&1 &
sleep 1
RH_READOUT=causal_g0x nohup setsid bash /root/rollhist/rh_k_chain.sh < /dev/null > /dev/null 2>&1 &
echo "ZZ RHK LAUNCHED ZZ"

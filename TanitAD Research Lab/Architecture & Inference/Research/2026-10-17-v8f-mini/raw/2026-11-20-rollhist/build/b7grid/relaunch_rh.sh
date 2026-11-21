#!/bin/bash
# MM 2026-11-21 (Amendment 4): relaunch the ROLLHIST chain, marker watcher first. Refuses if a chain runs.
ps -eo args | grep -q "[r]h_chain.sh" && { echo "an rh_chain.sh is running; refusing"; exit 1; }
nohup setsid bash /root/rollhist/rh_marker_watch.sh < /dev/null > /dev/null 2>&1 &
sleep 1
RH_READOUT=causal_g1x nohup setsid bash /root/rollhist/rh_chain.sh < /dev/null > /dev/null 2>&1 &
echo "ZZ RELAUNCHED ZZ"

#!/bin/bash
# MM 2026-11-21 (N1DH Amendment 2): relaunch the N1DH chain as n1dh_launch_after_rollhist.sh does (marker watcher, then chain). Refuses if a chain runs.
ps -eo args | grep -q "[n]1dh_chain.sh" && { echo "an n1dh_chain.sh is running; refusing"; exit 1; }
nohup setsid bash /root/strat1/n1dh/n1dh_marker_watch.sh < /dev/null > /dev/null 2>&1 &
sleep 1
nohup setsid bash /root/strat1/n1dh/n1dh_chain.sh < /dev/null > /dev/null 2>&1 &
echo "ZZ N1DH RELAUNCHED ZZ"

#!/bin/bash
# n1dm_launch.sh -- start the N1DM chain. STAGED, NOT STARTED: running N1DM needs the PI's second go with the cost (the first go, "Build both", covers building and pre-registering only).
# Run it only after n1dm_stage.sh (tree, kit, placeholders substituted) and the dry mint (dry_mint_n1dm.sh, rc 0 and checks.json read in full). Unlike n1dh_launch_after_rollhist.sh it waits for no other chain: the GPU queue lock inside
# n1dm_chain.sh is the only order. It refuses if a chain is already running. It starts the marker watcher (sets the pod owner's marker, removes it when the chain holds the lock), then the chain.
L=/root/strat1/n1dm/n1dm_launch.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
ps -eo args | grep -q "[n]1dm_chain.sh" && { say "an n1dm_chain.sh is running; refusing"; exit 1; }
[ -e /root/strat1/n1dm/n1dm_go.token ] || { say "no n1dm_go.token (the PI's second go is recorded there by the MM); refusing"; exit 2; }
nohup setsid bash /root/strat1/n1dm/n1dm_marker_watch.sh < /dev/null > /dev/null 2>&1 &
sleep 1
nohup setsid bash /root/strat1/n1dm/n1dm_chain.sh < /dev/null > /dev/null 2>&1 &
say "marker watcher and chain started"

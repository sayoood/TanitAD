#!/bin/bash
# start_after_rhk.sh -- idempotent starter of n1dm_after_rhk.sh (a script file, so no ssh command line self-matches the check)
grep -q 'n1dm_after_rhk[.]sh' /proc/[0-9]*/cmdline 2>/dev/null && { echo "already running"; exit 1; }
nohup setsid bash /root/strat1/n1dm/n1dm_after_rhk.sh < /dev/null > /dev/null 2>&1 &
sleep 2; echo "started"

#!/usr/bin/env bash
# Waker: exits when the DAG-ORC (DAG-OACC base) waiter goes (or stops), so the chain's own waker can be armed. Script polling, not model turns.
W=/c/Users/Admin/box2_ship/logs/dagoaccorc_waiter.out
for i in $(seq 1 720); do grep -aq 'waiter GO\|ZZDAGOACCORC-WAITER-STOP' $W 2>/dev/null && { tail -3 $W; exit 0; }; sleep 60; done
echo "WAKER TIMEOUT"

#!/bin/bash
# k2dc02tail_chain.sh -- ONE detached chain: build (identity control + K2DC02 tail) -> stage A. Ends at ZZK2DC02TAILCHAIN-A-DONEZZ or a named ABORT/FAILED marker (in $W/CHAIN_STATUS.txt and stdout).
# Stage B is NOT chained: it REFUSES until the K0-tail planner roll exists (see k2dc02tail_stageB_launch.sh header and HANDOFF).
export PATH=/usr/bin:/bin:$PATH
W=/home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02; S=$W/scripts
m() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $W/CHAIN_STATUS.txt; echo "ZZK2DC02TAILCHAIN-$1ZZ ${*:2}"; }
m START "pid $$"
bash $S/k2dc02tail_build.sh < /dev/null; rc=$?; [ $rc = 0 ] || { m "BUILD-FAILED" "rc $rc (see $W/BUILD_STATUS.txt)"; exit 3; }
m BUILD-DONE ""
bash $S/k2dc02tail_stageA.sh < /dev/null; rc=$?; [ $rc = 0 ] || { m "A-FAILED" "rc $rc (see $W/stageA/STATUS.txt)"; exit 3; }
m A-DONE "stage A done; stage B waits on the K0-tail planner roll (MM decision)"

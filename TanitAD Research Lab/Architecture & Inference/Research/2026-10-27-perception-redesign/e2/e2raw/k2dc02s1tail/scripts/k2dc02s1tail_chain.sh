#!/bin/bash
# k2dc02s1tail_chain.sh -- ONE detached chain: build (identity control + K2DC02s1 tail) -> stage A. Ends at ZZK2DC02S1TAILCHAIN-A-DONEZZ or a named ABORT/FAILED marker (in $W/CHAIN_STATUS.txt and stdout).
# Stage B is NOT chained: it REFUSES until the K0-tail planner roll exists (see k2dc02s1tail_stageB_launch.sh header and HANDOFF).
export PATH=/usr/bin:/bin:$PATH
DRY=0; DA=; if [ "${K2S_DRY:-0}" = 1 ] || [ "${1:-}" = DRY ]; then DRY=1; DA=DRY; fi     # DRY: each step runs its own DRY mode; nothing is written
W=/home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02s1; S=$W/scripts
[ $DRY = 1 ] && S=${K2S_SCRIPTS:-$(cd "$(dirname "$0")" && pwd)}        # dry: run the scripts next to this one (the run dir does not exist yet)
m() { [ $DRY = 1 ] || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $W/CHAIN_STATUS.txt; echo "ZZK2DC02S1TAILCHAIN-$1ZZ ${*:2}"; }
m START "pid $$"
bash $S/k2dc02s1tail_build.sh $DA < /dev/null; rc=$?; [ $rc = 0 ] || { m "BUILD-FAILED" "rc $rc (see $W/BUILD_STATUS.txt)"; exit 3; }
m BUILD-DONE ""
bash $S/k2dc02s1tail_stageA.sh $DA < /dev/null; rc=$?; [ $rc = 0 ] || { m "A-FAILED" "rc $rc (see $W/stageA/STATUS.txt)"; exit 3; }
m A-DONE "stage A done; stage B waits on the reproduction roll (k2dc02s1tail_repro_roll.sh)"

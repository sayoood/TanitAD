#!/bin/bash
# k2dc02tail_chain_all.sh -- MM 4a3d05, 2026-11-20. PREREG_K2DC02TAIL + AMENDMENT 1, serial on the Thor GPU, zero training:
#   build (identity control = tensor identity, A1) -> stage A -> comparator re-roll (A2) -> stage B launch (seeds K0, starts the roll; it ends AWAIT-S6C).
# Each step's own markers stay in its own status file. This wrapper writes ALLSTEP lines to $W/CHAIN_ALL_STATUS.txt and stops at the first failure.
export PATH=/usr/bin:/bin:$PATH
W=/home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02; S=$W/scripts
m() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $W/CHAIN_ALL_STATUS.txt; }
m "ALLSTEP START pid $$"
IDCTL_TENSOR_EQUAL=1 BANKED_PROV_SHA_PIN=9065c97666dfad6d9f259a7124f96cd50aa6dfe88ea0fd04356f016f09f72f8a bash $S/k2dc02tail_chain.sh < /dev/null
grep -q 'A-DONE' $W/CHAIN_STATUS.txt || { m "ALLSTEP STOP after build/stage A: $(tail -1 $W/CHAIN_STATUS.txt)"; exit 3; }
m "ALLSTEP A-DONE"
bash $S/k2dc02tail_cmp_reroll.sh < /dev/null > $W/cmp_reroll.log 2>&1; rc=$?
grep -q 'ZZK2DC02TAILCMP-DONEZZ' $W/cmp_reroll.log && [ $rc = 0 ] || { m "ALLSTEP STOP cmp re-roll rc $rc: $(tail -1 $W/cmp_reroll.log | cut -c1-200)"; exit 4; }
m "ALLSTEP CMP-DONE"
bash $S/k2dc02tail_stageB_launch.sh < /dev/null > $W/stageB_launch.log 2>&1; rc=$?
grep -q 'ZZK2DC02TAILB-STARTED' $W/stageB_launch.log && [ $rc = 0 ] || { m "ALLSTEP STOP stage B launch rc $rc: $(tail -1 $W/stageB_launch.log | cut -c1-200)"; exit 5; }
m "ALLSTEP B-STARTED $(grep ZZK2DC02TAILB-STARTED $W/stageB_launch.log | cut -c1-200)"

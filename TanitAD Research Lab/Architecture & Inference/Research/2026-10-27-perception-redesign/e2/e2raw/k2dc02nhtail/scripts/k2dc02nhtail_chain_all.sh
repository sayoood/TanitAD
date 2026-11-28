#!/bin/bash
# k2dc02nhtail_chain_all.sh -- MM 4a3d05, 2026-11-20. PREREG_K2DC02TAIL + AMENDMENT 1 (as PREREG_K2DC02NH sec. 2 requires), serial on the Thor GPU, zero training:
#   build (identity control = tensor identity, A1) -> stage A -> comparator re-roll (A2) -> stage B launch (seeds K0, starts the roll; it ends AWAIT-S6C).
# Each step's own markers stay in its own status file. This wrapper writes ALLSTEP lines to $W/CHAIN_ALL_STATUS.txt and stops at the first failure.
# Differences from k2dc02tail_chain_all.sh (PREREG_K2DC02NH_2026-11-20.md): ARM K2DC02nh; the comparator re-roll (cmp_reroll) is REPLACED by the pin check of the existing K2DC02-tail rolls
# (k2dc02nhtail_pin_check.sh, run FIRST, read-only) and the one-seed K0t reproduction roll (k2dc02nhtail_repro_roll.sh, PREREG_K2DC02S1 sec. 2 item 6).
# DRY (K2S_DRY=1 or first arg DRY): a SWEEP that runs the pin check and then every step in its own DRY mode, does not stop at a refusal, writes nothing, and names the FIRST refusal.
export PATH=/usr/bin:/bin:$PATH
DRY=0; if [ "${K2S_DRY:-0}" = 1 ] || [ "${1:-}" = DRY ]; then DRY=1; fi
W=/home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02nh; S=$W/scripts
[ $DRY = 1 ] && S=${K2S_SCRIPTS:-$(cd "$(dirname "$0")" && pwd)}
m() { [ $DRY = 1 ] || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $W/CHAIN_ALL_STATUS.txt; }
if [ $DRY = 1 ]; then
  FIRST=; sweep() { local tag=$1; shift; echo "=== [dry] $tag"; "$@" < /dev/null 2>&1 | sed 's/^/    /'; local rc=${PIPESTATUS[0]}; echo "=== [dry] $tag rc $rc"; [ $rc = 0 ] || [ -n "$FIRST" ] || FIRST="$tag (rc $rc)"; }
  sweep pin-check bash $S/k2dc02nhtail_pin_check.sh
  sweep build bash $S/k2dc02nhtail_build.sh DRY
  sweep stageA bash $S/k2dc02nhtail_stageA.sh DRY
  sweep repro-roll bash $S/k2dc02nhtail_repro_roll.sh DRY
  sweep stageB bash $S/k2dc02nhtail_stageB_launch.sh DRY
  echo "ZZK2DC02NHTAILALL-DRY-SWEEP-DONEZZ first refusal: ${FIRST:-none}"; exit 0
fi
m "ALLSTEP START pid $$"
bash $S/k2dc02nhtail_pin_check.sh < /dev/null > $W/pin_check.log 2>&1; rc=$?
grep -q 'ZZK2DC02NHTAILPIN-OKZZ' $W/pin_check.log && [ $rc = 0 ] || { m "ALLSTEP STOP pin check rc $rc: $(tail -1 $W/pin_check.log | cut -c1-200)"; exit 2; }
m "ALLSTEP PIN-OK"
IDCTL_TENSOR_EQUAL=1 BANKED_PROV_SHA_PIN=9065c97666dfad6d9f259a7124f96cd50aa6dfe88ea0fd04356f016f09f72f8a bash $S/k2dc02nhtail_chain.sh < /dev/null
grep -q 'A-DONE' $W/CHAIN_STATUS.txt || { m "ALLSTEP STOP after build/stage A: $(tail -1 $W/CHAIN_STATUS.txt)"; exit 3; }
m "ALLSTEP A-DONE"
bash $S/k2dc02nhtail_repro_roll.sh < /dev/null > $W/repro_roll.log 2>&1; rc=$?
grep -q 'ZZK2DC02NHTAILREPRO-DONEZZ' $W/repro_roll.log && [ $rc = 0 ] || { m "ALLSTEP STOP reproduction roll rc $rc: $(tail -1 $W/repro_roll.log | cut -c1-200)"; exit 4; }
m "ALLSTEP REPRO-DONE"
bash $S/k2dc02nhtail_stageB_launch.sh < /dev/null > $W/stageB_launch.log 2>&1; rc=$?
grep -q 'ZZK2DC02NHTAILB-STARTED' $W/stageB_launch.log && [ $rc = 0 ] || { m "ALLSTEP STOP stage B launch rc $rc: $(tail -1 $W/stageB_launch.log | cut -c1-200)"; exit 5; }
m "ALLSTEP B-STARTED $(grep ZZK2DC02NHTAILB-STARTED $W/stageB_launch.log | cut -c1-200)"

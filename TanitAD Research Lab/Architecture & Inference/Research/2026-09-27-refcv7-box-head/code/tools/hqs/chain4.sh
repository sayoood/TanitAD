#!/bin/bash
# GPU chain v3 (Master Mind 2026-09-27 ~11:15, A14.1): after NEW-2's A12_DONE and my HQS_READY: the three one-frame tests
# (heatmap, learned_ref, the unanchored red arm), then G-BOX-OVERFIT on the passing anchored arms, learned_ref FIRST;
# stop at the first PASS. The corrected MAIN and the +R6 rung are deprioritised (not run here). The paused
# refcv6-head control (pid 3603614, state T) is tolerated and left untouched.
D=/home/nvidia/bx_anch_1116
P2=3653915
H=$D/HQS_READY
C=3603614
gpu_free() {
  while true; do
    others=""
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
      if { [ "$p" = "$C" ] || [ "$p" = "$P2" ]; } && grep -q "^State:[[:space:]]*T" /proc/$p/status 2>/dev/null; then continue; fi
      others="$others $p"
    done
    [ -z "$others" ] && return 0
    echo "GPU busy ($others) $(date '+%F %T %Z')" >> $D/chain.log
    sleep 30
  done
}
while [ ! -f "$H" ]; do sleep 5; done
echo "HQS_READY seen (A12 paused by the Master Mind) $(date +%s) $(date '+%F %T %Z')" >> $D/chain.log
sleep 2
for Q in learned heatmap learned_ref; do
  gpu_free
  echo "one-frame $Q start $(date +%s) $(date '+%F %T %Z')" >> $D/chain.log
  $D/run_of.sh $D $Q > $D/of_$Q.log 2>&1
  echo "one-frame $Q exited rc=$? $(date +%s) $(date '+%F %T %Z')" >> $D/chain.log
done
ORDER=$(/home/nvidia/venvs/tanitad-train/bin/python $D/decide.py $D)
echo "decision: [$ORDER] $(cat $D/DECISION.json | tr '\n' ' ')" >> $D/chain.log
for Q in $ORDER; do
  gpu_free
  echo "G-BOX-OVERFIT $Q start $(date +%s) $(date '+%F %T %Z')" >> $D/chain.log
  $D/run_gbo.sh $D $Q > $D/gbo_$Q.log 2>&1
  R=$(/home/nvidia/venvs/tanitad-train/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))['RESULT'])" $D/gbo_$Q.json 2>/dev/null)
  echo "G-BOX-OVERFIT $Q exited, RESULT=$R $(date +%s) $(date '+%F %T %Z')" >> $D/chain.log
  [ "$R" = "PASS" ] && break
done
echo "chain3 done $(date '+%F %T %Z')" >> $D/chain.log

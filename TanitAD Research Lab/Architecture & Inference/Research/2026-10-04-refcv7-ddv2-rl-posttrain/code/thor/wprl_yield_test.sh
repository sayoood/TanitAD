#!/bin/bash
# CPU-only test of the in-process lock YIELD on a PRIVATE lock file (never the GPU lock): the
# trainer must release the lock to a queued job (after >= min-segment-minutes = 0), wait until that
# job HOLDS it, then re-acquire and continue -- and the windows/RNG must continue unchanged.
source /home/nvidia/refcv7_post/rl/wprl_env.sh
export CUDA_VISIBLE_DEVICES=""
L=/tmp/wprl_yield_test.lock
D=$R/dry/yield
rm -rf $D; mkdir -p $D
cd $TREE
$PY stack/scripts/ddv2_rl_refcv7.py train --device cpu --arm rl --steps 3 --batch 2 --micro 1 \
   --windows $R/out/windows_train.json --out-dir $D --inproc-lock $L --min-segment-minutes 0 \
   --segment-minutes 0 --ckpt-every 0 --workers 2 > $D/train.log 2>&1 < /dev/null &
TP=$!
# wait until the trainer holds the private lock, then queue a dummy job behind it
for i in $(seq 1 600); do grep -q "GPU lock acquired" $D/train.log && break; sleep 2; done
echo "[yield] trainer holds the lock at $(date -u +%T)"
setsid bash -c "flock $L bash -c 'echo DUMMY_GOT_LOCK \$(date -u +%T) >> $D/dummy.log; sleep 15; echo DUMMY_DONE \$(date -u +%T) >> $D/dummy.log'" < /dev/null > /dev/null 2>&1 &
wait $TP
echo "[yield] trainer rc=$? at $(date -u +%T)"
cat $D/dummy.log
grep -E "RELEASED|RE-ACQUIRED|acquired|end:" $D/train.log
cat $D/segments.jsonl

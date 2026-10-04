#!/bin/bash
# WP-RL: run ONE arm to completion on Thor, segment by segment (SPEC_RL sec. 9).
#   ./wprl_arm.sh <argv_json>      (an ARGV_<arm>.json written by wprl_argv.py; its sha256 must be
#                                    listed in the Master Mind's PASS token for WP-RL)
# The trainer holds the shared GPU lock IN-PROCESS (--inproc-lock): it builds the model and the
# 4,369-clip dataset ONCE, trains <= 45 min per segment, checkpoints, releases the lock, waits while
# any other job holds or waits on it, and re-acquires -- no rebuild. A crash restarts from
# ckpt_latest.pt (same windows, same RNG streams) at most 3 times. The ARTIFACTS (DONE.json,
# segments.jsonl, metrics.jsonl) are the evidence; the exit code only steers this loop.
source /home/nvidia/refcv7_post/rl/wprl_env.sh
ARGV_JSON=$1
TOKEN=$R/gate/PASS_WPRL.json
if [ ! -s "$TOKEN" ]; then echo "[arm] REFUSED: no launch-gate PASS token at $TOKEN"; exit 3; fi
SHA=$(sha256sum "$ARGV_JSON" | cut -d' ' -f1)
if ! grep -q "$SHA" "$TOKEN"; then echo "[arm] REFUSED: argv sha256 $SHA is not in the PASS token"; exit 3; fi
OUT=$($PY -c "import json,sys; a=json.load(open('$ARGV_JSON')); print(a[a.index('--out-dir')+1])")
mkdir -p $OUT
cp "$ARGV_JSON" $OUT/argv.json
cd $TREE
tries=0
while true; do
  if [ -s $OUT/DONE.json ]; then echo "[arm] $(date -u +%FT%TZ) DONE $OUT"; break; fi
  if [ -e $OUT/STOP ]; then echo "[arm] $(date -u +%FT%TZ) STOP file -- not relaunching"; break; fi
  echo "[arm] $(date -u +%FT%TZ) process start (try $tries) $OUT"
  $PY -c "import json,os,sys; a=json.load(open('$ARGV_JSON')); os.execv(sys.executable, [sys.executable] + a)" >> $OUT/train.log 2>> $OUT/train.stderr.log < /dev/null
  rc=$?
  echo "[arm] $(date -u +%FT%TZ) process rc=$rc $(tail -1 $OUT/segments.jsonl 2>/dev/null)"
  if [ -s $OUT/DONE.json ]; then continue; fi
  tries=$((tries+1))
  if [ $tries -ge 3 ]; then echo "[arm] three failed processes -- stopping"; break; fi
  sleep 120
done

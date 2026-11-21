#!/bin/sh
# The step-5,000 milestone, by hand or from the waiter: waiter_milestone.sh 5000 polls Thor for the
# COMPLETE ckpt_5000.pt, pulls it md5-verified (step key asserted locally), then calls
# chain_milestone.sh 5000. Run the chain directly ONLY when D:/refcv7_eval_kit/ckpt/ckpt_5000.pt and
# D:/refcv7_eval_kit/chain/pull_5000_OK.json already exist.
#   nohup sh chain_step5000.sh wait  > /d/refcv7_eval_kit/chain/waiter_5000.out 2>&1 &   # the armed form
#   sh chain_step5000.sh run                                                             # checkpoint already pulled
HERE=$(cd "$(dirname "$0")" && pwd)
case "${1:-wait}" in
  wait) exec sh "$HERE/waiter_milestone.sh" 5000 ;;
  run)  exec sh "$HERE/chain_milestone.sh" 5000 ;;
  *) echo "usage: chain_step5000.sh [wait|run]"; exit 2 ;;
esac

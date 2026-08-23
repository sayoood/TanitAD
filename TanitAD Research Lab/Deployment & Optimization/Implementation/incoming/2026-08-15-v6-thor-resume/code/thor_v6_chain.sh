#!/bin/bash
# The full v6-resume chain on Thor, SEQUENCED (the concurrent version choked the
# box off the network on 2026-08-15). Persistent paths only — /tmp did not
# survive the reboot. Logs: ~/logs/.
#
#   1. finish the corpus pull (resumable; 465/2403 train files already on disk)
#   2. verify the pull BY LOADING (the pull script's own [verify] steps)
#   3. launch the trainer detached via thor_v6_launch.py (exact banked args)
#
# Speed measurement happens from the DEV BOX by polling train_log.jsonl growth
# (marginal s/step over >=3 logged points) — nothing extra runs here.
set -u
mkdir -p ~/logs
L=~/logs/v6_pull.log

echo "[chain] $(date -u +%FT%TZ) starting pull (resume, max_workers=4)" >> ~/logs/v6_chain.log
~/venvs/tanitad-train/bin/python -u ~/thor_v6_pull.py > "$L" 2>&1
RC=$?
echo "[chain] pull rc=$RC" >> ~/logs/v6_chain.log
if ! grep -q "ALL DONE" "$L"; then
  echo "[chain] REFUSING TO LAUNCH: pull did not reach ALL DONE (rc=$RC)" >> ~/logs/v6_chain.log
  exit 1
fi

echo "[chain] $(date -u +%FT%TZ) pull verified; launching trainer" >> ~/logs/v6_chain.log
setsid nohup ~/venvs/tanitad-train/bin/python -u ~/thor_v6_launch.py \
  > ~/logs/v6_train.out 2>&1 < /dev/null &
echo "[chain] trainer pid=$!" >> ~/logs/v6_chain.log

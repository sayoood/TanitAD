#!/bin/bash
# pace-based auto-restart (2026-07-10): ride the post-restart honeymoon.
# Restart when fewer than 50 steps progress in 15 min (pace < 200/h).
LOG=/workspace/experiments/p0-sB01-realmix.log
LAST=0
while true; do
  sleep 900
  CUR=$(grep -o "\"step\": [0-9]*" $LOG | tail -1 | grep -o "[0-9]*")
  [ -z "$CUR" ] && continue
  D=$((CUR - LAST))
  if [ "$LAST" != "0" ] && [ "$D" -lt 50 ] && [ "$CUR" -lt 29900 ]; then
    echo "$(date -u) pace collapse: +$D steps/15min at $CUR — restarting" >> /workspace/experiments/watchdog.log
    pkill -f "train_worldmode[l]"
  fi
  LAST=$CUR
done

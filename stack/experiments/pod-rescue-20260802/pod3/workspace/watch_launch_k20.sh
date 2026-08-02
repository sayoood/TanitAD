#!/bin/bash
# Fires run_v1k20_v2bal.sh the moment the v2bal corpus reaches 9000 clips.
# WHY A WATCHER: the delta relay takes ~20h and my orchestration loop has already been shown to
# break silently (the ScheduleWakeup that did not re-arm). Making the launch depend on a poll THAT
# LIVES ON THE POD removes the orchestrator from the critical path entirely.
# The launcher itself re-checks the count and refuses at !=9000, so this is belt-and-braces.
C=/workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d
W=/workspace/watch_k20.log
echo "=== watcher armed $(date -u +%FT%TZ) ===" >> "$W"
while true; do
  N=$(ls "$C" 2>/dev/null | grep -c 'v2ep.pt$')
  if [ "$N" -ge 9000 ]; then
    # let the tar finish flushing before reading the files
    sleep 120
    N2=$(ls "$C" 2>/dev/null | grep -c 'v2ep.pt$')
    echo "$(date -u +%FT%TZ) corpus complete ($N2) -> launching k20" >> "$W"
    setsid nohup bash /workspace/run_v1k20_v2bal.sh </dev/null >/dev/null 2>&1 &
    exit 0
  fi
  echo "$(date -u +%FT%TZ) $N/9000" >> "$W"
  sleep 600
done

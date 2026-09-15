#!/bin/bash
# Publishes the corpus semantic maps every 6 h (corpus_publisher.py); once the production supervisor has written DONE, one --final run
# marks the manifest COMPLETE and the loop exits. STOP_PUBLISH = clean off-switch. Lock on fd 201, closed on every child.
C=/home/nvidia/sam3map/corpus; PY=/home/nvidia/venvs/tanitad-edge/bin/python
mkdir -p "$C/publish"
exec 201>"$C/publish/loop.lock"
flock -n 201 || { echo "$(date -Is) another publish loop holds the lock - exiting"; exit 1; }
log() { echo "$(date -Is) $*" >> "$C/publish/loop.log"; }
log "publish loop start pid $$"
while true; do
  if [ -f "$C/STOP_PUBLISH" ]; then log "STOP_PUBLISH present - exiting"; break; fi
  final=""; [ -f "$C/DONE" ] && final="--final"
  env HF_HUB_DISABLE_PROGRESS_BARS=1 "$PY" /home/nvidia/sam3map/eval/corpus_publisher.py $final >> "$C/publish/publisher.out" 2>&1 < /dev/null 201>&-
  rc=$?; log "publisher exit $rc ${final:-(incremental)}"
  if [ -n "$final" ] && [ "$rc" -eq 0 ]; then log "final publish done - exiting"; break; fi
  sleep 21600 201>&-
done

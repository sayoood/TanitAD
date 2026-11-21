#!/bin/bash
# TIER 4 — the data that is NOT already on pod3. Chained to start after the checkpoint
# rescue (pid passed as $1) finishes, so the two never compete for the link.
#
# DELIBERATELY EXCLUDED: /workspace/data/physicalai_phase0 (302 G). It carries a PARITY_OK
# marker over the SAME build that pod3 holds intact as
# pai_epcache/physicalai-train-e438721ae894 (2376 episodes verified). Copying it would spend
# ~2 h and ~302 G to duplicate a verified-good corpus. If pod3 is ever lost this decision
# must be revisited — that is why it is written down rather than silently skipped.
#
# NOTHING IS DELETED ON THE SOURCE.

set -u
WAITPID="${1:-}"
OLDH="root@38.147.83.15"
SSHO="-n -p 39198 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20 -i /root/.ssh/id_ed25519"
DEST=/workspace/rescue
LOG=$DEST/rescue.log

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ -n "$WAITPID" ]; then
  log "TIER 4 waiting for checkpoint rescue (pid $WAITPID) to finish ..."
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 20; done
  log "TIER 4 starting."
fi

# tar-stream a whole directory: far better than file-by-file for many small files.
pulldir() {
  local rel="$1"
  local src="/workspace/data/$rel"
  mkdir -p "$DEST/data"
  if [ -d "$DEST/data/$rel" ] && [ -f "$DEST/data/.$rel.done" ]; then
    log "OK (already complete): data/$rel"; return
  fi
  local rs
  rs=$(ssh $SSHO "$OLDH" "du -sb '$src' 2>/dev/null | cut -f1"); rs=${rs//[^0-9]/}
  [ -z "$rs" ] && rs=0
  if [ "$rs" = "0" ]; then log "SKIP (absent on source): data/$rel"; return; fi
  log "PULL data/$rel  $((rs/1048576)) MiB ..."
  local t0=$(date +%s)
  ssh $SSHO "$OLDH" "tar cf - -C /workspace/data '$rel'" | tar xf - -C "$DEST/data"
  local rc=$?
  local t1=$(date +%s); local d=$((t1-t0)); [ $d -lt 1 ] && d=1
  local ls
  ls=$(du -sb "$DEST/data/$rel" 2>/dev/null | cut -f1); ls=${ls:-0}
  if [ "$rc" = "0" ] && [ "$ls" -ge $((rs*97/100)) ]; then
    touch "$DEST/data/.$rel.done"
    log "  DONE data/$rel  $((ls/1048576)) MiB in ${d}s = $((ls/1048576/d)) MB/s"
  else
    log "  !! INCOMPLETE data/$rel  got $((ls/1048576)) of $((rs/1048576)) MiB (rc=$rc) — retry next run"
  fi
}

log "=== TIER 4: data unique to the old pod (physicalai_phase0 excluded by design) ==="
pulldir cosmos
pulldir physicalai_v2

log "=== TIER 4 COMPLETE — rescue total $(du -sh $DEST | cut -f1) ==="

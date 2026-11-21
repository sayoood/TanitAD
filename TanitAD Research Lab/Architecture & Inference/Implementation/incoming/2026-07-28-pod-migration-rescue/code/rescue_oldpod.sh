#!/bin/bash
# Rescue the US-TX-1 CPU-only pod (aa88adadb582) -> ca-mtl-1 GPU pod (00b05f408e35).
# Direct pod->pod over the old pod's public TCP mapping. Resumable at FILE granularity:
# a file whose local size already equals the remote size is skipped, so a re-run costs nothing.
# NOTHING IS DELETED ON THE SOURCE — deletion is not authorized.

set -u
OLDH="root@38.147.83.15"
OLDP=39198
SSHO="-n -p $OLDP -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20 -i /root/.ssh/id_ed25519"
DEST=/workspace/rescue
LOG=/workspace/rescue/rescue.log

mkdir -p "$DEST"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

rsize() { ssh $SSHO "$OLDH" "stat -c %s '$1' 2>/dev/null || echo 0"; }

pull() {
  local src="$1" rel="$2"
  local dst="$DEST/$rel"
  mkdir -p "$(dirname "$dst")"
  local rs ls
  rs=$(rsize "$src"); rs=${rs//[^0-9]/}
  [ -z "$rs" ] && rs=0
  if [ "$rs" = "0" ]; then log "SKIP (absent on source): $rel"; return; fi
  ls=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  if [ "$ls" = "$rs" ]; then log "OK (already complete): $rel  $((rs/1048576)) MiB"; return; fi
  log "PULL $rel  $((rs/1048576)) MiB ..."
  local t0=$(date +%s)
  ssh $SSHO "$OLDH" "cat '$src'" > "$dst"
  local t1=$(date +%s); local d=$((t1-t0)); [ $d -lt 1 ] && d=1
  ls=$(stat -c %s "$dst" 2>/dev/null || echo 0)
  if [ "$ls" = "$rs" ]; then
    log "  DONE $rel  $((ls/1048576)) MiB in ${d}s = $((ls/1048576/d)) MB/s"
  else
    log "  !! SHORT $rel  got $ls of $rs — will retry on next run"
  fi
}

log "=== RESCUE START  old=aa88adadb582(US-TX-1)  new=$(hostname)(ca-mtl-1) ==="

# --- TIER 1: makes flagship-v2corpus-30k RESUMABLE (highest value per byte) ---
log "--- TIER 1: flagship-v2corpus-30k resume set ---"
for f in config.json supervisor.log train_log.jsonl ckpt.pt; do
  pull "/workspace/experiments/flagship-v2corpus-30k/$f" "experiments/flagship-v2corpus-30k/$f"
done

# --- TIER 2: the remaining unique checkpoints ---
log "--- TIER 2: remaining experiments ---"
ssh $SSHO "$OLDH" "cd /workspace/experiments && find . -type f -printf '%s\t%p\n'" > /tmp/oldfiles.txt 2>/dev/null
awk -F'\t' '{print $2}' /tmp/oldfiles.txt | sed 's|^\./||' | sort | while read -r rel; do
  case "$rel" in
    flagship-v2corpus-30k/*) continue ;;
  esac
  pull "/workspace/experiments/$rel" "experiments/$rel"
done

# --- TIER 3: loose root checkpoints ---
log "--- TIER 3: root checkpoints ---"
pull /workspace/ckpt27k_flagship.pt ckpt27k_flagship.pt
pull /workspace/ckpt14k_frozen.pt   ckpt14k_frozen.pt

log "=== RESCUE COMPLETE — $(du -sh $DEST | cut -f1) in $DEST ==="

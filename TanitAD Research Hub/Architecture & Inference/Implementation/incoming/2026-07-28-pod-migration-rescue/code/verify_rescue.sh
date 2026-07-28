#!/bin/bash
# md5-VERIFY the rescue, source vs destination.
# Size-match is NOT evidence of an intact checkpoint — a truncated-then-padded file, a partial
# write, or a silently dropped block all survive a size check. CLAUDE.md's rule is "verify md5
# either way", and the old pod cannot be released on anything weaker.
#
# Chained after the transfer (pid $1) so it never competes with it for the link.
# READ-ONLY on both sides. Nothing is deleted anywhere.

set -u
WAITPID="${1:-}"
OLDH="root@38.147.83.15"
SSHO="-n -p 39198 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20 -i /root/.ssh/id_ed25519"
DEST=/workspace/rescue
LOG=$DEST/verify.log

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ -n "$WAITPID" ]; then
  log "VERIFY waiting for transfer (pid $WAITPID) ..."
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 20; done
  log "VERIFY starting."
fi

log "=== md5 VERIFY: source(US-TX-1) vs destination(ca-mtl-1) ==="

pass=0; fail=0; missing=0

verify_one() {
  local src="$1" dst="$2" label="$3"
  local rm lm
  rm=$(ssh $SSHO "$OLDH" "md5sum '$src' 2>/dev/null | cut -d' ' -f1")
  rm=$(echo "$rm" | tr -dc 'a-f0-9')
  lm=$(md5sum "$dst" 2>/dev/null | cut -d' ' -f1)
  if [ -z "$rm" ] || [ -z "$lm" ]; then
    log "  MISSING $label (src='${rm:0:8}' dst='${lm:0:8}')"; missing=$((missing+1)); return
  fi
  if [ "$rm" = "$lm" ]; then
    log "  PASS $label  $rm"; pass=$((pass+1))
  else
    log "  !! MISMATCH $label  src=$rm  dst=$lm"; fail=$((fail+1))
  fi
}

# --- every .pt checkpoint under experiments/ (the irreplaceable material) ---
log "--- checkpoints under experiments/ ---"
find "$DEST/experiments" -type f -name "*.pt" 2>/dev/null | sort | while read -r f; do
  rel="${f#$DEST/}"
  echo "$rel"
done > /tmp/ckpt_rel.txt
while read -r rel; do
  verify_one "/workspace/$rel" "$DEST/$rel" "$rel"
done < /tmp/ckpt_rel.txt

# --- root checkpoints ---
log "--- root checkpoints ---"
for f in ckpt27k_flagship.pt ckpt14k_frozen.pt; do
  [ -f "$DEST/$f" ] && verify_one "/workspace/$f" "$DEST/$f" "$f"
done

log "=== VERIFY DONE — PASS=$pass  MISMATCH=$fail  MISSING=$missing ==="
if [ "$fail" = "0" ] && [ "$missing" = "0" ]; then
  log "✅ every checkpoint md5-identical to source. Safe to consider releasing the old pod (PI's call)."
else
  log "🔴 NOT clean — do NOT release the old pod. Re-run /workspace/rescue_oldpod.sh (it re-pulls only what differs)."
fi

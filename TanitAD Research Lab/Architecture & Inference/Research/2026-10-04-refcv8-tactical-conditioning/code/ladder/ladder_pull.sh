#!/usr/bin/env bash
# Dev-box side of the ladder chain (SPEC_WPB_LADDER; MM 2026-10-04 "copy the ckpt to the dev box (md5) and delete it
# on Thor"). Polls Thor for arms whose eval is done (`run/PULL_READY`), pulls the model-only checkpoint + the run
# record + the eval passes over the LAN, verifies the md5 ON THE DEV-BOX BYTES, and only then deletes the checkpoint
# on Thor by its explicit path. Never deletes anything it has not verified. Exits on ZZALLDONEZZ in chain.log.
# Usage (dev box, Git Bash): nohup bash ladder_pull.sh > <dest>/pull.out 2>&1 &
set -u
HOST=${HOST:-tanitad-thor}
W=${W:-/home/nvidia/refcv8_ladder/W}
DEST=${DEST:-D:/refcv8_ladder_pull}
mkdir -p "$DEST"
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$DEST/pull.log"; }
while :; do
  ready=$(timeout 60 ssh -n -o ConnectTimeout=10 "$HOST" "ls $W/arms/*/run/PULL_READY 2>/dev/null" | tr -d '\r')
  for f in $ready; do
    arm=$(basename "$(dirname "$(dirname "$f")")")
    [ -f "$DEST/$arm/PULLED" ] && continue
    mkdir -p "$DEST/$arm/run" "$DEST/$arm/eval"
    ok=1
    for x in model_final.pt model_final.pt.md5 config.json metrics.jsonl summary.json; do
      timeout 900 scp -q "$HOST:$W/arms/$arm/run/$x" "$DEST/$arm/run/$x" || ok=0
    done
    timeout 900 scp -q "$HOST:$W/eval/$arm/*" "$DEST/$arm/eval/" || ok=0
    want=$(tr -dc 0-9a-f < "$DEST/$arm/run/model_final.pt.md5" 2>/dev/null)
    got=$(md5sum "$DEST/$arm/run/model_final.pt" 2>/dev/null | cut -c1-32)
    if [ "$ok" = 1 ] && [ ${#want} -eq 32 ] && [ "$want" = "$got" ]; then
      timeout 60 ssh -n "$HOST" "rm -f $W/arms/$arm/run/model_final.pt && touch $W/arms/$arm/run/PULLED" \
        && { echo "$got" > "$DEST/$arm/PULLED"; log "PULLED $arm md5 $got (Thor copy deleted)"; }
    else
      log "INCOMPLETE $arm (ok=$ok want=${want:-?} got=${got:-?}) -- Thor copy KEPT, retry next poll"
    fi
  done
  timeout 60 ssh -n "$HOST" "grep -q ZZALLDONEZZ $W/chain.log 2>/dev/null && echo ZZFIN" | grep -q ZZFIN && {
    timeout 300 scp -q "$HOST:$W/{LADDER_SCORE.json,I0.json,S0.json,SIZE_K.json,N.txt,chain.log}" "$DEST/" 2>/dev/null
    log "chain done -- final records pulled"; exit 0; }
  sleep 300
done

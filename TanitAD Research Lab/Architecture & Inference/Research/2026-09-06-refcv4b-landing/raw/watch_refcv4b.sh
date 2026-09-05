#!/bin/bash
# P0 watcher for refcv4b-b1-v72-40k. CANNOT go silent on failure:
# it exits on COMPLETION *and* on FAILURE, and it also exits if the probe
# channel itself dies (consecutive ssh failures), because "no news" from a
# broken channel is not evidence of health.
#
# Parses ONLY the opaque marker ZZ<d>-<d>-<d>-<d>-<d>-<d>ZZ, which is disjoint
# from every token this script or the pod-side probe searches for.
set -uo pipefail

SD="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f407bc82-7969-457c-a947-6be2014fee89/scratchpad"
STATUS="$SD/refcv4b_watch.status"
LOG="$SD/refcv4b_watch.log"
TARGET=40284
POLL=300
MAX_ITERS=170          # ~14.2 h ceiling
sshfail=0
supgone=0
trngone=0
i=0

say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
finish() { echo "$1" > "$STATUS"; say "EXIT $1"; exit "$2"; }

say "watcher up; target=$TARGET poll=${POLL}s"
echo "RUNNING" > "$STATUS"

while [ "$i" -lt "$MAX_ITERS" ]; do
  i=$((i + 1))
  raw=$(timeout 150 ssh -n -o ConnectTimeout=25 -o ServerAliveInterval=15 \
        tanitad-refcv3 'bash /workspace/refcv4b_probe.sh' 2>&1)
  mark=$(printf '%s\n' "$raw" | grep -oE 'ZZ[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+ZZ' | tail -1)

  if [ -z "$mark" ]; then
    sshfail=$((sshfail + 1))
    say "probe returned no marker (consecutive=$sshfail): $(printf '%s' "$raw" | tr '\n' ' ' | cut -c1-200)"
    if [ "$sshfail" -ge 8 ]; then
      finish "FAIL_CHANNEL_DEAD after $sshfail consecutive probe failures" 20
    fi
    sleep 60
    continue
  fi
  sshfail=0

  body=${mark#ZZ}; body=${body%ZZ}
  IFS='-' read -r STEP SUM ERR SUP TRN CKPT <<< "$body"
  say "step=$STEP summary=$SUM errs=$ERR sup=$SUP trainer_procs=$TRN ckpts=$CKPT"
  printf 'RUNNING step=%s summary=%s errs=%s sup=%s trn=%s ckpts=%s at %s\n' \
    "$STEP" "$SUM" "$ERR" "$SUP" "$TRN" "$CKPT" "$(date -u +%FT%TZ)" > "$STATUS"

  if [ "$SUM" -ge 1 ]; then
    finish "DONE_MARKER_PRESENT step=$STEP ckpts=$CKPT" 0
  fi
  if [ "$ERR" -ge 1 ]; then
    finish "FAIL_ERRORS_IN_LOG step=$STEP errcount=$ERR" 21
  fi
  if [ "$STEP" -ge "$TARGET" ]; then
    finish "REACHED_TARGET_NO_MARKER step=$STEP" 22
  fi
  if [ "$SUP" -eq 0 ]; then
    supgone=$((supgone + 1))
    [ "$supgone" -ge 2 ] && finish "FAIL_SUPERVISOR_GONE step=$STEP" 23
  else
    supgone=0
  fi
  if [ "$TRN" -eq 0 ]; then
    trngone=$((trngone + 1))
    [ "$trngone" -ge 3 ] && finish "FAIL_TRAINER_GONE step=$STEP" 24
  else
    trngone=0
  fi

  sleep "$POLL"
done
finish "FAIL_WATCHER_TIMEOUT after $i iterations" 25

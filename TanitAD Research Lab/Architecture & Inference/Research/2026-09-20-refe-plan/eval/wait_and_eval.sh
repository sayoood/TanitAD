#!/bin/bash
# Banked copy of the dev-box snapshot waiter (the working copy lives in the session scratchpad as wait_and_eval2.sh).
# Wake on the EVENT (the snapshot file appearing on the pod -- atomic_save makes present == complete), then run the
# whole eval at once instead of waiting for the next hourly heartbeat. Read-only on the pod.
# v2 (2026-09-26 ~22:00 Berlin): IDEMPOTENT -- a waiter claims its snapshot with an atomic `mkdir` lock before it
# launches, so a second waiter armed for the same snapshot (e.g. by a heartbeat) exits instead of running a duplicate
# eval into the same directories. (v1 is still running for snapshot 014 and is left untouched: bash reads a running
# script lazily by byte offset, so it is never edited in place.)
# usage: wait_and_eval2.sh NNN DEADLINE_EPOCH_SECONDS [WORKERS]
# markers: $S/eval_NNN.lockdir (the claim) and $S/eval_NNN.started (the launch time)
# prints: ZZWAIT_TIMEOUT | ZZWAIT_SSH_DOWN | ZZWAIT_ALREADY_CLAIMED | the eval's own ZZSNAP_DONE / ZZSNAP_FAIL line
N="$1"; DL="$2"; W="${3:-2}"
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/bd7d00af-b98e-42f1-a53c-cb4113059b0f/scratchpad"
R="/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3"
K="-i $HOME/.ssh/tanitad_pod"
if [ -e "$S/eval_$N.lockdir" ] || [ -e "$S/eval_$N.started" ]; then echo "ZZWAIT_ALREADY_CLAIMED $N"; exit 4; fi
misses=0
while true; do
  out=$(timeout 60 ssh -n $K -p <port> -o ConnectTimeout=20 -o BatchMode=yes root@<redacted-ip> \
        "test -f $R/snap_epoch$N.pt && echo ZZPRESENT || echo ZZABSENT" 2>/dev/null)
  case "$out" in
    *ZZPRESENT*) break ;;
    *ZZABSENT*) misses=0 ;;
    *) misses=$((misses + 1)); if [ $misses -ge 10 ]; then echo "ZZWAIT_SSH_DOWN $N $(date -u +%FT%TZ)"; exit 3; fi ;;
  esac
  if [ "$(date +%s)" -gt "$DL" ]; then echo "ZZWAIT_TIMEOUT $N $(date -u +%FT%TZ)"; exit 2; fi
  sleep 120
done
if ! mkdir "$S/eval_$N.lockdir" 2>/dev/null; then echo "ZZWAIT_ALREADY_CLAIMED $N"; exit 4; fi
date -u +%FT%TZ > "$S/eval_$N.started"
echo "snapshot $N present $(date -u +%FT%TZ); eval starts"
bash "$S/eval_snap.sh" "$N" "$W" > "$S/eval_snap_$N.log" 2>&1
grep -E "ZZSNAP_|ZZA4|ZZREADOUT|ZZPROPTABLE_|G3" "$S/eval_snap_$N.log" | tail -8

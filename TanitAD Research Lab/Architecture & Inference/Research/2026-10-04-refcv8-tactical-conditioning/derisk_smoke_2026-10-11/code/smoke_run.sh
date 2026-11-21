#!/usr/bin/env bash
# refcv8 full-size de-risk smoke (non-binding): the launch argv, 100 steps, ONE gradient-share reading at step 50.
# Serialised behind the R2 ladder through the SAME GPU lock the ladder chain takes (flock is blocking, not FIFO: this
# waiter simply waits for the lock and runs between ladder chunks). Cap 3600 s, counted from the lock being held.
set -u
S=/home/nvidia/refcv8_smoke
TREE=/home/nvidia/refcv6_run/1df35031db
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
CAP_S=3600
MEM_MIN_GB=40
export PYTHONPATH="$TREE/stack:$TREE/taniteval" REFCV6_REPO="$TREE" REFCV6_KIT=/home/nvidia OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8
cd "$S" || exit 3
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$S/smoke.log"; }
mapfile -d '' ARGS < <("$PY" -c 'import json,sys; sys.stdout.write(chr(0).join(json.load(open(sys.argv[1]))["argv"]))' "$S/smoke.argv.json")
[ "${#ARGS[@]}" -ge 200 ] || { log "ZZSMOKE-STOPZZ argv has ${#ARGS[@]} tokens (expected 210)"; exit 3; }
[ ! -e "$S/run/metrics.jsonl" ] || { log "ZZSMOKE-STOPZZ $S/run/metrics.jsonl already exists (a smoke is one run)"; exit 3; }
log "ZZSMOKE-QUEUEDZZ ${#ARGS[@]} tokens; waiting on $LOCK (the ladder holds it between chunks)"
while :; do
  flock "$LOCK" bash "$S/smoke_inlock.sh" "$MEM_MIN_GB" timeout "$CAP_S" "$PY" "$TREE/stack/scripts/refc_v3_train.py" "${ARGS[@]}" >> "$S/train.log" 2>&1 200>&- 9>&-
  rc=$?
  [ "$rc" != 97 ] && break
  log "MemAvailable < ${MEM_MIN_GB} GB at lock time -- lock released, retry in 5 min"
  sleep 300 200>&- 9>&-
done
log "ZZSMOKE-ENDZZ trainer-rc=$rc (artifact check: $S/run/metrics.jsonl, summary.json)"
if [ -s "$S/run/metrics.jsonl" ]; then
  "$PY" "$TREE/stack/scripts/refcv8_assemble_argv.py" --read-gs-measure "$S/run/metrics.jsonl" --out-json "$S/gs_measure.json" >> "$S/smoke.log" 2>&1 200>&- 9>&-
fi
log "ZZSMOKE-DONEZZ"

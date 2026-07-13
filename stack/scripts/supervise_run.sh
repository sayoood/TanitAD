#!/usr/bin/env bash
# =============================================================================
# TanitAD supervised trainer — boot-persistent auto-resume + external heartbeat
# =============================================================================
# Relaunches a trainer that AUTO-RESUMES from ckpt.pt whenever it dies, forever,
# until the run's DONE sentinel (summary.json "done": true) appears. Emits a
# heartbeat JSON to a known path on the persistent /workspace volume every
# HB_PERIOD seconds so a stall (step not advancing) or a death (heartbeat mtime
# frozen) is externally detectable WITHOUT an interactive session.
#
# This closes the exact gap that killed the dataset build: a bare detached tmux
# does not come back after a pod restart. Paired with pod_boot_hook.sh (invoked
# by the /pre_start.sh stub that RunPod's /start.sh runs on every container
# start), the supervisor is relaunched on reboot with zero human intervention.
#
# Usage:   bash supervise_run.sh <runs.d/<run>.env>
# The manifest (sourced) must define: RUN_ID OUT WORKDIR TRAIN_CMD
# and may override: DONE_TOKEN HEARTBEAT HB_PERIOD MAX_BACKOFF
# -----------------------------------------------------------------------------
set -u
RUN_ENV="${1:?usage: supervise_run.sh <run.env>}"
# shellcheck disable=SC1090
source "$RUN_ENV"
: "${RUN_ID:?manifest must set RUN_ID}"
: "${OUT:?manifest must set OUT}"
: "${TRAIN_CMD:?manifest must set TRAIN_CMD}"
: "${WORKDIR:=.}"
: "${DONE_TOKEN:=FLAGSHIP4B_DONE}"
: "${HB_PERIOD:=30}"
: "${INIT_BACKOFF:=10}"
: "${MAX_BACKOFF:=120}"
: "${OPS_DIR:=/workspace/ops}"
: "${HEARTBEAT:=${OPS_DIR}/heartbeats/${RUN_ID}.json}"

SUP_LOG="${OUT}/supervisor.log"
TRAIN_OUT="${OUT}/train.out"          # combined stdout/stderr of the trainer
TRAIN_LOG="${OUT}/train_log.jsonl"    # the trainer's own json-lines (has "step")
SUMMARY="${OUT}/summary.json"
LOCK="${OPS_DIR}/locks/${RUN_ID}.lock"
POD="$(hostname)"
RESTARTS=0
CHILD_PID=""

mkdir -p "$OUT" "$(dirname "$HEARTBEAT")" "$(dirname "$LOCK")" 2>/dev/null || true

log(){ echo "[$(date -u +%FT%TZ)] $*" >>"$SUP_LOG" 2>/dev/null; echo "[superv:$RUN_ID] $*" >&2; }

# --- single-instance guard: never let two supervisors fight over ckpt.pt -----
# Primary: flock on a RUN_ID-keyed lock file (airtight, released on death).
# Fallback (no flock): match the manifest path in our own argv via pgrep.
SELF_MATCH="supervise_run.sh.*$(basename "$RUN_ENV")"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK" || true
  if ! flock -n 9; then log "another supervisor holds $LOCK — exiting"; exit 0; fi
elif command -v pgrep >/dev/null 2>&1; then
  if [ "$(pgrep -fc "$SELF_MATCH" 2>/dev/null || echo 1)" -gt 1 ]; then
    log "another supervisor for $(basename "$RUN_ENV") is alive — exiting"; exit 0
  fi
fi

is_done(){
  { [ -f "$SUMMARY" ] && grep -q '"done"[[:space:]]*:[[:space:]]*true' "$SUMMARY" 2>/dev/null; } && return 0
  { [ -f "$TRAIN_OUT" ] && grep -q "$DONE_TOKEN" "$TRAIN_OUT" 2>/dev/null; } && return 0
  return 1
}

last_step(){
  tail -n 60 "$TRAIN_LOG" 2>/dev/null \
    | grep -o '"step"[[:space:]]*:[[:space:]]*[0-9]\+' | tail -1 \
    | grep -o '[0-9]\+'
}

write_hb(){   # $1 = status
  local step mt now
  step="$(last_step)"; step="${step:-null}"
  mt="$(stat -c %Y "$TRAIN_LOG" 2>/dev/null || echo 0)"
  now="$(date +%s)"
  cat > "${HEARTBEAT}.tmp" 2>/dev/null <<JSON
{"run":"${RUN_ID}","pod":"${POD}","status":"${1}","last_step":${step},"train_log_mtime":${mt},"hb_epoch":${now},"hb_iso":"$(date -u +%FT%TZ)","restarts":${RESTARTS},"pid":${CHILD_PID:-null},"out":"${OUT}"}
JSON
  mv -f "${HEARTBEAT}.tmp" "$HEARTBEAT" 2>/dev/null || true
}

if is_done; then log "run already DONE (summary.json) — nothing to do"; write_hb done; exit 0; fi
log "supervisor UP on ${POD}; OUT=${OUT}; heartbeat=${HEARTBEAT}"
write_hb starting

backoff="$INIT_BACKOFF"
while :; do
  if is_done; then log "DONE detected — stopping"; write_hb done; break; fi
  if [ ! -d "$WORKDIR" ]; then log "WORKDIR $WORKDIR missing — retry in ${backoff}s"; write_hb blocked; sleep "$backoff"; continue; fi
  log "launch attempt (restarts=${RESTARTS}) in ${WORKDIR}"
  ( cd "$WORKDIR" && exec bash -c "$TRAIN_CMD" ) >>"$TRAIN_OUT" 2>&1 &
  CHILD_PID=$!
  log "trainer pid=${CHILD_PID}"
  # inline heartbeat while the child lives (no extra helper process to lose)
  while kill -0 "$CHILD_PID" 2>/dev/null; do write_hb running; sleep "$HB_PERIOD"; done
  wait "$CHILD_PID"; rc=$?
  CHILD_PID=""
  log "trainer exited rc=${rc}"
  if is_done; then log "clean finish (summary.json done)"; write_hb done; break; fi
  RESTARTS=$((RESTARTS+1))
  write_hb relaunching
  log "not done — relaunch #${RESTARTS} in ${backoff}s (resumes from ckpt.pt)"
  sleep "$backoff"
  backoff=$(( backoff * 2 )); [ "$backoff" -gt "$MAX_BACKOFF" ] && backoff="$MAX_BACKOFF"
done
log "supervisor exiting (run complete)"

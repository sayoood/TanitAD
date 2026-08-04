#!/usr/bin/env bash
# =============================================================================
# Change a supervised run's TRAIN_CMD without losing training — safely.
# =============================================================================
# Written 2026-08-04 after a hand-rolled cutover cost ~40 min of the v5f
# headline run. It encodes the four mechanisms that each burned a debugging
# round, in the order they bite:
#
#  1. supervise_run.sh SOURCES ITS MANIFEST ONCE, at supervisor startup — not
#     per relaunch. Editing runs.d/<run>.env under a live supervisor changes
#     NOTHING: it replays the command captured at boot and the relaunch LOOKS
#     successful. => the SUPERVISOR must die first, then the trainer.
#  2. Restarting the supervisor immediately after killing the old one RACES ITS
#     flock: the new one exits "another supervisor holds ...lock" and NOTHING
#     RUNS, while the log reads like a normal startup. => poll until both are
#     gone, and verify the lock has no holder.
#  3. `pkill -f <trainer>` SELF-MATCHES the ssh command running this script and
#     kills the session, returning empty output so it looks like a no-op.
#     => every kill here is by EXPLICIT PID, and every state check filters out
#     this script's own pid.
#  4. Killing mid-save-interval discards up to --save-every steps. => this waits
#     for a checkpoint at or beyond a step you name before touching anything.
#
# And the rule that makes it verifiable: CONFIRM THE NEW FLAGS BY READING THE
# RUNNING PROCESS (/proc/<pid>/cmdline), never by reading the manifest back.
#
# Usage:
#   supervised_cutover.sh --run <run-id> --min-step <N> [--timeout-min 180] \
#                         [--expect-flag '--batch 8'] [--dry-run]
#
# The manifest must ALREADY contain the new TRAIN_CMD — edit it first, then run
# this. On failure to bring the new config up, it RESTORES the manifest backup
# it took and restarts the supervisor on the old command, so a bad cutover
# degrades to the status quo instead of to a dead run.
# -----------------------------------------------------------------------------
set -u

RUN_ID=""; MIN_STEP=""; TIMEOUT_MIN=180; EXPECT_FLAG=""; DRY=0
OPS_DIR="${OPS_DIR:-/workspace/ops}"
STACK="${STACK:-/workspace/TanitAD/stack}"

while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN_ID="$2"; shift 2 ;;
    --min-step) MIN_STEP="$2"; shift 2 ;;
    --timeout-min) TIMEOUT_MIN="$2"; shift 2 ;;
    --expect-flag) EXPECT_FLAG="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$RUN_ID" ] || { echo "usage: --run <run-id> --min-step <N>" >&2; exit 2; }
[ -n "$MIN_STEP" ] || { echo "usage: --run <run-id> --min-step <N>" >&2; exit 2; }

ENVF="${OPS_DIR}/runs.d/${RUN_ID}.env"
LOCK="${OPS_DIR}/locks/${RUN_ID}.lock"
[ -r "$ENVF" ] || { echo "no manifest at $ENVF" >&2; exit 2; }
# shellcheck disable=SC1090
OUT="$(. "$ENVF" >/dev/null 2>&1; echo "${OUT:-}")"
[ -n "$OUT" ] || { echo "manifest defines no OUT" >&2; exit 2; }
CK="${OUT}/ckpt.pt"

log(){ echo "[$(date -u +%FT%TZ)] $*"; }

# --- state helpers. NEVER pgrep -f: it matches this very script. -------------
SELF=$$
# supervisor pid(s) for this run, excluding ourselves and our children
sup_pids(){
  ps -eo pid,cmd 2>/dev/null | awk -v self="$SELF" -v run="$RUN_ID" \
    '$1 != self && /supervise_run\.sh/ && index($0, run) > 0 && !/awk/ {print $1}'
}
# trainer pid(s): a python process whose cmdline names this run's OUT
trainer_pids(){
  ps -eo pid,cmd 2>/dev/null | awk -v self="$SELF" -v run="$RUN_ID" \
    '$1 != self && /python/ && index($0, run) > 0 && !/awk/ && !/supervise_run/ {print $1}'
}
alive(){ kill -0 "$1" 2>/dev/null; }

ckpt_step(){
  python3 - "$CK" <<'PY' 2>/dev/null
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step", -1)))
except Exception:
    print(-1)
PY
}

# --- 0. show what we are about to do ----------------------------------------
log "run=$RUN_ID OUT=$OUT"
log "supervisor pids: $(sup_pids | tr '\n' ' ')"
log "trainer pids:    $(trainer_pids | tr '\n' ' ')"
log "manifest TRAIN_CMD now:"; grep -c . "$ENVF" >/dev/null && sed -n 's/^TRAIN_CMD=//p' "$ENVF" | head -1
if [ "$DRY" = "1" ]; then log "DRY RUN — stopping here"; exit 0; fi

# --- 1. wait for a checkpoint so the cutover costs ~0 steps ------------------
DEADLINE=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
while :; do
  S="$(ckpt_step)"
  log "ckpt step=$S (need >= $MIN_STEP)"
  [ -n "$S" ] && [ "$S" -ge "$MIN_STEP" ] 2>/dev/null && break
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "TIMEOUT waiting for step $MIN_STEP — nothing killed, run untouched"; exit 1; }
  sleep 60
done
log "checkpoint $S banked — proceeding"

BACKUP="${ENVF}.pre-cutover.$(date -u +%Y%m%dT%H%M%SZ)"
cp -p "$ENVF" "$BACKUP"; log "manifest backed up -> $BACKUP"

# --- 2. SUPERVISOR FIRST, then the trainer (mechanism 1) --------------------
for p in $(sup_pids); do log "killing SUPERVISOR pid $p"; kill "$p" 2>/dev/null; done
for p in $(trainer_pids); do log "killing TRAINER pid $p"; kill "$p" 2>/dev/null; done

# --- 3. poll until BOTH are gone (mechanism 2) ------------------------------
for _ in $(seq 1 90); do
  [ -z "$(sup_pids)" ] && [ -z "$(trainer_pids)" ] && break
  sleep 2
done
for p in $(sup_pids) $(trainer_pids); do
  log "pid $p survived SIGTERM — escalating to SIGKILL (explicit pid, never pkill -f)"
  kill -9 "$p" 2>/dev/null
done
for _ in $(seq 1 30); do
  [ -z "$(sup_pids)" ] && [ -z "$(trainer_pids)" ] && break
  sleep 2
done
if [ -n "$(sup_pids)$(trainer_pids)" ]; then
  log "FAILED: processes still alive: $(sup_pids) $(trainer_pids) — NOT relaunching"; exit 1
fi
log "supervisor and trainer both down"

# A lock file with no holder in /proc/*/fd is debris, not contention.
if [ -e "$LOCK" ]; then
  HOLDER=""
  for fd in /proc/[0-9]*/fd/*; do
    [ "$(readlink -f "$fd" 2>/dev/null)" = "$(readlink -f "$LOCK")" ] && { HOLDER="$fd"; break; }
  done
  [ -z "$HOLDER" ] && log "lock $LOCK has NO holder — debris, safe to proceed" \
                   || log "WARNING: lock still held via $HOLDER"
fi

# --- 4. fresh supervisor, which re-sources the manifest ---------------------
log "starting a fresh supervisor (re-sources the manifest)"
setsid nohup bash "${STACK}/scripts/supervise_run.sh" "$ENVF" \
  >"/tmp/superv_${RUN_ID}.log" 2>&1 < /dev/null &
sleep 25

# --- 5. VERIFY FROM THE RUNNING PROCESS, never from the manifest ------------
NEWSUP="$(sup_pids | head -1)"; NEWTRAIN="$(trainer_pids | head -1)"
log "new supervisor pid=${NEWSUP:-NONE} trainer pid=${NEWTRAIN:-NONE}"
OK=1
[ -n "$NEWSUP" ] || { log "NO SUPERVISOR came up"; OK=0; }
if [ -n "$NEWTRAIN" ]; then
  CMD="$(tr '\0' ' ' < "/proc/${NEWTRAIN}/cmdline" 2>/dev/null)"
  log "RUNNING trainer cmdline: $CMD"
  if [ -n "$EXPECT_FLAG" ]; then
    case "$CMD" in
      *"$EXPECT_FLAG"*) log "VERIFIED: running process carries '$EXPECT_FLAG'" ;;
      *) log "MISMATCH: running process does NOT carry '$EXPECT_FLAG'"; OK=0 ;;
    esac
  fi
else
  log "NO TRAINER came up"; OK=0
fi

# --- 6. degrade to the status quo rather than to a dead run -----------------
if [ "$OK" = "0" ]; then
  log "CUTOVER FAILED — restoring $BACKUP and restarting on the old command"
  for p in $(sup_pids) $(trainer_pids); do kill -9 "$p" 2>/dev/null; done
  sleep 5
  cp -p "$BACKUP" "$ENVF"
  setsid nohup bash "${STACK}/scripts/supervise_run.sh" "$ENVF" \
    >"/tmp/superv_${RUN_ID}.log" 2>&1 < /dev/null &
  sleep 20
  log "rollback supervisor=$(sup_pids | head -1) trainer=$(trainer_pids | head -1)"
  exit 1
fi

log "CUTOVER OK. Watch: rss+shmem (NOT usage_in_bytes) and the first logged step."
log "  python3 ${STACK}/scripts/pod_kill_forensics.py --pids ${NEWTRAIN} --gpu-samples 15"
exit 0

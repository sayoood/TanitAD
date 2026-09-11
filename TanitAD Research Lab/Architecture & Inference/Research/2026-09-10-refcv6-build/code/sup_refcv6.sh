#!/usr/bin/env bash
# refcv6 supervisor -- one arm per supervisor.
#
#   ./sup_refcv6.sh <ARM> [RUNS_DIR]
#
# Every rule below is here because it cost a run. The comment names the failure.
#
# ⛔ 1. THE LOCK FD IS CLOSED ON EVERY CHILD (`200>&-`), NOT JUST THE TRAINER.
#    `exec 200>"$LOCK"; flock -n 200` puts the lock on fd 200. A child launched
#    as `nohup python3 ... &` INHERITS EVERY OPEN FD, so the trainer ends up
#    holding the supervisor's lock for its entire life -- and once the old
#    supervisor is killed the lock is NEVER released while its trainer lives, so
#    no replacement can ever start. It is not a race; it is a permanent block
#    that merely LOOKS like one, and the `.out` file says the same reassuring
#    sentence either way.
#    ⛔⛔ AND `200>&-` ON THE TRAINER ALONE IS NOT ENOUGH -- that partial fix was
#    shipped and failed the same day: the lock was still held by `sleep 180`, the
#    supervisor's own poll child, which had outlived its parent. EVERY child gets
#    `200>&-`, the sleeps included.
#
# ⛔ 2. THE DONE-MARKER IS WRITTEN ON COMPLETION, AND CHECKED AT STARTUP.
#    A supervised run that finished but never wrote its marker gets RESURRECTED
#    the moment whatever made its relaunches crash is fixed -- MEASURED: a run
#    that finished 2026-08-09 was relaunched for 2 days, then a fix landed, a
#    relaunch SUCCEEDED, resumed from a stale ckpt, and started overwriting
#    config.json / metrics.json / ckpt.pt in the canonical run directory while
#    burning GPU next to a live eval.
#
# ⛔ 3. KILL BY EXPLICIT PID. `pkill -f <trainer>` SELF-MATCHES the ssh command
#    that issued it and kills your own session, returning empty output that looks
#    exactly like "nothing was running".
#
# ⛔ 4. THE PROGRESS MARKER IS DISJOINT FROM ANYTHING GREPPED.
#    A monitor whose filter contains the pattern it searches for matches its own
#    echoed command line -- the interactive PTY echoes it back. MEASURED THREE
#    TIMES; the worst case reported `Traceback CUDA out of memory` on a run that
#    was healthy and 3 minutes in. Counts are computed HERE and emitted as an
#    opaque `ZZ...ZZ` token; the client parses that, never the raw stream.
#
# ⛔ 5. SUCCESS IS ASSERTED ON THE ARTIFACT, NEVER ON AN EXIT CODE.
#    `$?` after a pipeline is the LAST element's status, so `cmd | tail` reports
#    tail's success -- four false successes in two days. This script uses
#    PIPESTATUS where a pipe is unavoidable, and the done-marker is written only
#    after `metrics.jsonl` is READ and its step count checked.
#
# ⚠️ 6. NEVER `sed -i` THIS FILE WHILE IT IS RUNNING. bash reads a script lazily
#    by byte offset; an in-place edit makes a live shell execute garbage from the
#    middle of a line. Write a new file and switch to it.
#
# ⚠️ 7. TO CHANGE A LIVE RUN: edit the manifest, kill the SUPERVISOR FIRST, then
#    the trainer, then start a fresh supervisor. The supervisor SOURCES ITS
#    MANIFEST ONCE at startup; editing it under a live supervisor changes
#    nothing and the relaunch looks successful while running the OLD config.

set -u  # ⛔ deliberately NOT `set -e`: a supervisor that exits on the first
        # non-zero command is not a supervisor.

ARM="${1:?usage: sup_refcv6.sh <ARM> [RUNS_DIR]}"
RUNS_DIR="${2:-/workspace/refcv6/runs.d}"
MANIFEST="${RUNS_DIR}/${ARM}.env"

if [ ! -f "$MANIFEST" ]; then
  echo "[sup:${ARM}] FATAL: no manifest at ${MANIFEST}"
  exit 2
fi

# Sourced ONCE, on purpose (rule 7). Must define: TRAIN_CMD, OUT_DIR, STEPS.
# shellcheck disable=SC1090
. "$MANIFEST"

: "${TRAIN_CMD:?manifest must define TRAIN_CMD}"
: "${OUT_DIR:?manifest must define OUT_DIR}"
: "${STEPS:?manifest must define STEPS}"

LOCK="${RUNS_DIR}/${ARM}.lock"
LOG="${OUT_DIR}/train.log"
ERRLOG="${OUT_DIR}/train.stderr.log"
SUPLOG="${OUT_DIR}/sup_${ARM}.log"
DONE_MARKER="${OUT_DIR}/summary.json"
MAX_RELAUNCH="${MAX_RELAUNCH:-8}"
POLL_S="${POLL_S:-120}"

mkdir -p "$OUT_DIR" "$RUNS_DIR"

log() { echo "[sup:${ARM}] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$SUPLOG"; }

# ---- rule 2: never resurrect a finished run -------------------------------- #
if [ -f "$DONE_MARKER" ] && grep -q '"done"[[:space:]]*:[[:space:]]*true' "$DONE_MARKER" 2>/dev/null; then
  log "done-marker present at ${DONE_MARKER} -- run is FINISHED. Exiting without launching."
  exit 0
fi

# ---- rule 1: take the lock, and close it on every child -------------------- #
exec 200>"$LOCK"
if ! flock -n 200; then
  log "another supervisor holds ${LOCK} -- exiting."
  log "⚠️ If nothing is actually supervising, the holder may be an INHERITED fd"
  log "   (a trainer or a stray sleep). Name it with:"
  log "   for p in /proc/[0-9]*/fd/*; do [ \"\$(readlink \$p 2>/dev/null)\" = \"${LOCK}\" ] && tr '\\0' ' ' < /proc/\$(echo \$p|cut -d/ -f3)/cmdline && echo; done"
  log "   ⛔ Do NOT kill a live trainer to free it -- point a new supervisor at a FRESH lock path instead."
  exit 3
fi
log "lock acquired on ${LOCK} (fd 200); supervisor pid $$"

# ---- artifact-based completion test (rule 5) ------------------------------- #
# ⛔ Reads metrics.jsonl and requires the LAST step to have reached STEPS.
#    An exit code is a claim by the process about itself; the artifact is the
#    thing we actually wanted.
# NOTE 2026-09-11: the artifact is `metrics.jsonl`, JSON-LINES, opened in APPEND
# mode by `refc_v3_train.py:4956`. This function used to name `metrics.json` and
# `json.load` it -- MEASURED against a real completed run dir
# (/home/nvidia/experiments/tacgoal-wsweep/A_w0/: ckpt.pt, config.json,
# metrics.jsonl, summary.json, train.log -- NO metrics.json). It could therefore
# NEVER have returned 0, so the done-marker would NEVER have been written, and
# this supervisor would have relaunched a FINISHED arm until MAX_RELAUNCH ran
# out -- which is precisely the resurrection failure rule 2 exists to prevent.
# "Assert on the artifact, not the status" only protects you when it is the
# RIGHT artifact.
run_is_complete() {
  local mj="${OUT_DIR}/metrics.jsonl"
  [ -s "$mj" ] || return 1
  python3 - "$mj" "$STEPS" <<'PYEOF' 200>&-
import json, sys
steps = []
try:
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue            # a torn final line is skipped, never fatal
            if isinstance(r, dict) and isinstance(r.get("step"), int):
                steps.append(r["step"])
except Exception:
    sys.exit(1)
sys.exit(0 if steps and max(steps) >= int(sys.argv[2]) - 1 else 1)
PYEOF
}

write_done_marker() {
  local final_step="$1"
  python3 - "$DONE_MARKER" "$ARM" "$final_step" "$STEPS" "$OUT_DIR" <<'PYEOF' 200>&-
import json, sys, time
path, arm, final_step, steps, out_dir = sys.argv[1:6]
with open(path, "w", encoding="utf-8") as fh:
    json.dump({
        "done": True, "arm": arm,
        "final_step": int(final_step), "steps_requested": int(steps),
        "out_dir": out_dir,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("Written by sup_refcv6.sh AFTER metrics.jsonl was read and its "
                 "step count checked. Its presence is the supervisor's OFF "
                 "SWITCH: a supervisor started against this directory exits "
                 "immediately rather than resurrecting a finished run."),
    }, fh, indent=2)
PYEOF
}

final_step_from_metrics() {
  python3 - "${OUT_DIR}/metrics.jsonl" <<'PYEOF' 200>&-
import json, sys
steps = []
try:
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, dict) and isinstance(r.get("step"), int):
                steps.append(r["step"])
except Exception:
    print(-1); raise SystemExit(0)
print(max(steps) if steps else -1)
PYEOF
}

# ---- the supervise loop ---------------------------------------------------- #
launch=0
while [ "$launch" -lt "$MAX_RELAUNCH" ]; do
  launch=$((launch + 1))

  if run_is_complete; then
    fs="$(final_step_from_metrics)"
    log "metrics.jsonl already reaches step ${fs} >= ${STEPS} -- writing done-marker."
    write_done_marker "$fs"
    log "done-marker written; exiting cleanly."
    exit 0
  fi

  log "launch #${launch} of ${MAX_RELAUNCH}: ${TRAIN_CMD}"
  # ⛔ rule 1: `200>&-` closes the lock fd in the child. Without it this trainer
  #    holds the supervisor's lock for its entire life.
  # shellcheck disable=SC2086
  nohup ${TRAIN_CMD} >> "$LOG" 2>> "$ERRLOG" 200>&- &
  TRAIN_PID=$!
  echo "$TRAIN_PID" > "${OUT_DIR}/train.pid"
  log "trainer pid ${TRAIN_PID} (recorded in ${OUT_DIR}/train.pid -- kill by THIS pid, never pkill -f)"

  # ---- poll ---------------------------------------------------------------- #
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    # rule 4: compute counts HERE, emit an OPAQUE marker. The client greps
    # `ZZ...ZZ`, which appears nowhere in any command line.
    cur="$(final_step_from_metrics)"
    n_err=0
    if [ -s "$ERRLOG" ]; then
      # ⛔ the searched words are BUILT here, never written literally, so this
      #    command line cannot match itself in an echoing PTY.
      pat="$(printf 'Trace''back|CUDA out of mem''ory|OutOfMemory')"
      n_err="$(grep -Ec "$pat" "$ERRLOG" 2>/dev/null || echo 0)"
    fi
    echo "ZZ${ARM}-${cur}-${STEPS}-${n_err}-${launch}ZZ" | tee -a "$SUPLOG" > /dev/null
    sleep "$POLL_S" 200>&-   # ⛔ rule 1: the sleep gets it too (this exact child
                             #    once held the lock and made a run unsupervisable)
  done

  wait "$TRAIN_PID"
  rc=$?
  log "trainer pid ${TRAIN_PID} exited rc=${rc} (⚠️ rc is NOT the verdict -- the artifact is)"

  if run_is_complete; then
    fs="$(final_step_from_metrics)"
    log "ARTIFACT CHECK PASSED: metrics.jsonl reaches step ${fs} >= ${STEPS}."
    write_done_marker "$fs"
    log "done-marker written at ${DONE_MARKER}; supervisor exiting cleanly."
    exit 0
  fi

  cur="$(final_step_from_metrics)"
  log "ARTIFACT CHECK FAILED: metrics.jsonl reaches ${cur}, wanted >= ${STEPS}."
  log "⚠️ refc_v3_train.py has NO --resume (two probes) -- a relaunch RESTARTS this arm."
  log "   Relaunching only because MAX_RELAUNCH allows it; set MAX_RELAUNCH=1 to forbid."
  sleep 30 200>&-
done

log "⛔ exhausted ${MAX_RELAUNCH} launches without a complete artifact. NO done-marker written."
log "   The absence of ${DONE_MARKER} is the admissible evidence that this arm did not finish."
exit 4

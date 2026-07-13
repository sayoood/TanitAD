#!/usr/bin/env bash
# =============================================================================
# TanitAD boot-persistent relauncher (idempotent)
# =============================================================================
# Invoked at container start by the /pre_start.sh stub that RunPod's /start.sh
# runs, AND safe to run by hand at any time. For every ACTIVE run manifest in
# RUNS_DIR it (re)launches a detached supervisor UNLESS one is already alive or
# the run is already DONE.
#
# Contract with /start.sh: /start.sh runs under `set -e` and calls /pre_start.sh
# SYNCHRONOUSLY *before* sshd starts, so this hook must (a) return fast and
# (b) never exit non-zero — otherwise the pod fails to boot and locks us out.
# The /pre_start.sh stub therefore backgrounds this script; this script itself
# also ends in `exit 0` and wraps its body so a bad manifest can't abort it.
# All durable logic lives here on the /workspace volume so a plain "Restart Pod"
# needs zero intervention.
# -----------------------------------------------------------------------------
RUNS_DIR="${RUNS_DIR:-/workspace/ops/runs.d}"
HOOK_LOG="${HOOK_LOG:-/workspace/ops/boot.log}"
# Resolve the supervisor next to THIS script so the bundle is relocatable
# (deployed to /workspace/ops/bin, outside the git checkout -> survives git clean).
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo /workspace/ops/bin)"
SUPERVISOR="${SUPERVISOR:-${SELF_DIR}/supervise_run.sh}"

mkdir -p "$(dirname "$HOOK_LOG")" "$RUNS_DIR" 2>/dev/null || true

{
  echo "[$(date -u +%FT%TZ)] boot hook on $(hostname); runs_dir=${RUNS_DIR}"
  if [ ! -x "$SUPERVISOR" ] && [ ! -f "$SUPERVISOR" ]; then
    echo "  ERROR: supervisor not found at ${SUPERVISOR} — is /workspace/TanitAD checked out?"
  fi
  shopt -s nullglob
  for envf in "$RUNS_DIR"/*.env; do
    (
      # subshell: a malformed manifest can't take down the loop
      RUN_ID=""; OUT=""; ENABLED=1
      # shellcheck disable=SC1090
      source "$envf" 2>/dev/null || { echo "  skip ${envf} (source failed)"; exit 0; }
      [ -n "$RUN_ID" ] && [ -n "$OUT" ] || { echo "  skip ${envf} (no RUN_ID/OUT)"; exit 0; }
      if [ "${ENABLED:-1}" != "1" ]; then echo "  skip ${RUN_ID} (ENABLED=${ENABLED})"; exit 0; fi
      if [ -f "${OUT}/summary.json" ] && grep -q '"done"[[:space:]]*:[[:space:]]*true' "${OUT}/summary.json" 2>/dev/null; then
        echo "  ${RUN_ID}: already DONE"; exit 0
      fi
      if command -v pgrep >/dev/null 2>&1 && pgrep -f "supervise_run.sh.*$(basename "$envf")" >/dev/null 2>&1; then
        echo "  ${RUN_ID}: supervisor already running"; exit 0
      fi
      mkdir -p "$OUT" 2>/dev/null || true
      echo "  ${RUN_ID}: launching detached supervisor"
      setsid nohup bash "$SUPERVISOR" "$envf" </dev/null >>"${OUT}/supervisor.out" 2>&1 &
    )
  done
  echo "[$(date -u +%FT%TZ)] boot hook finished"
} >>"$HOOK_LOG" 2>&1

exit 0

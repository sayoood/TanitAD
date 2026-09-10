#!/usr/bin/env bash
# refcv6 -- ⛔ ASSERT A SUPERVISOR IS ACTUALLY RUNNING, VIA /proc/<pid>/cmdline.
#
#   ./assert_supervisor.sh <ARM> [RUNS_DIR]
#
# Why not `grep -c`:
#
#   ⛔ A POLLING MONITOR WHOSE FILTER CONTAINS THE PATTERN IT SEARCHES FOR WILL
#   MATCH ITS OWN ECHOED COMMAND. The interactive PTY echoes the command line
#   back, so `ps -eo args | grep -c sup_refcv6` counts the grep itself, the ssh
#   command that carried it, and any tee of the transcript. MEASURED THREE TIMES;
#   the worst instance invented a FAILURE that never happened on a healthy run.
#
#   `pgrep -f` / `pkill -f` are the same trap with teeth: they self-match the ssh
#   command and kill your own session, returning empty output that reads exactly
#   like "nothing was running".
#
# ⭐ The check that settles it: read the PID the supervisor RECORDED, then read
#    THAT pid's own `/proc/<pid>/cmdline` and require the arm name in it. A pid
#    file alone is not enough -- pids are reused, and a stale pid file after a
#    reboot points at something else entirely.
#
# ⛔ Every failure in this family REPORTS SUCCESS AND LEAVES NOTHING RUNNING, so
#    this must be run AFTER every supervisor start, never assumed.

set -u

ARM="${1:?usage: assert_supervisor.sh <ARM> [RUNS_DIR]}"
RUNS_DIR="${2:-/workspace/refcv6/runs.d}"
PIDFILE="${RUNS_DIR}/${ARM}.sup.pid"

fail() { echo "SUPERVISOR_ASSERT=FAIL arm=${ARM} reason=$*"; exit 1; }

[ -f "$PIDFILE" ] || fail "no pid file at ${PIDFILE}"

SUP_PID="$(cat "$PIDFILE" 2>/dev/null | tr -dc '0-9')"
[ -n "$SUP_PID" ] || fail "pid file ${PIDFILE} is empty or unreadable"

CMDFILE="/proc/${SUP_PID}/cmdline"
[ -r "$CMDFILE" ] || fail "no /proc/${SUP_PID}/cmdline -- pid ${SUP_PID} is NOT alive"

# NUL-separated; turn it into spaces.
CMD="$(tr '\0' ' ' < "$CMDFILE")"

# ⭐ Positive assertion on CONTENT: the live process must be THIS supervisor for
#    THIS arm. A pid that is alive but belongs to something else is a stale pid
#    file, which is exactly the case a `kill -0` check would pass.
case "$CMD" in
  *sup_refcv6*)  : ;;
  *) fail "pid ${SUP_PID} is alive but is not sup_refcv6 (cmdline: ${CMD})" ;;
esac
case "$CMD" in
  *" ${ARM} "*|*" ${ARM}") : ;;
  *) fail "pid ${SUP_PID} is a sup_refcv6 but not for arm ${ARM} (cmdline: ${CMD})" ;;
esac

# The trainer, if one is running, reported separately -- its absence is NOT a
# supervisor failure (the supervisor may be between relaunches).
OUT_DIR_HINT=""
if [ -f "${RUNS_DIR}/${ARM}.env" ]; then
  OUT_DIR_HINT="$(sed -n 's/^OUT_DIR=//p' "${RUNS_DIR}/${ARM}.env" | tr -d '"'"'" | head -1)"
fi
TRAIN_STATE="none"
if [ -n "$OUT_DIR_HINT" ] && [ -f "${OUT_DIR_HINT}/train.pid" ]; then
  TP="$(cat "${OUT_DIR_HINT}/train.pid" 2>/dev/null | tr -dc '0-9')"
  if [ -n "$TP" ] && [ -r "/proc/${TP}/cmdline" ]; then
    TRAIN_STATE="alive:${TP}"
  else
    TRAIN_STATE="dead-or-stale:${TP:-?}"
  fi
fi

# ⛔ Opaque marker, disjoint from every token this script's command line carries,
#    so a client-side filter cannot match the command that produced it.
echo "QQ${ARM}-${SUP_PID}-${TRAIN_STATE}QQ"
echo "SUPERVISOR_ASSERT=OK arm=${ARM} sup_pid=${SUP_PID} trainer=${TRAIN_STATE}"
exit 0

#!/usr/bin/env bash
# Advisory GPU lock for shared pods — WHY THIS EXISTS.
#
# On 2026-07-20 three VLM jobs landed on tanitad-eval while a latency benchmark
# was running. The benchmark did not crash; it produced CLEAN-LOOKING numbers
# that were ~2x wrong, and 9 arms had to be discarded. A contaminated latency
# measurement is worse than a missing one, because it publishes.
#
# The trap it closes: "wait for ALLDONE in the log" is NOT the same as "the GPU
# is free". A log token marks one script finishing, not the end of an agent's
# campaign. Lock on the DEVICE, not on a log line.
#
# Usage:
#   gpu_lock.sh acquire <owner> [timeout_s] [--adopt] [--pid N] [--ttl S]
#       --adopt   I am ALREADY the GPU occupant — record the lock, skip the
#                 busy-check. Use when a campaign is in flight.
#       --pid N   liveness is tied to PID N (your long-running job). Without it
#                 the lock holds on a TTL instead (default 4 h).
#   gpu_lock.sh release <owner>
#   gpu_lock.sh status
#
# Typical: launch the job, then `acquire mine --pid <job_pid>`; or, if you are
# about to start, `acquire mine` (TTL-held) and re-acquire with --pid once known.
#
# The lock is ADVISORY: it only works if every job takes it. It is also
# self-healing — a lock whose PID is gone is treated as stale and broken.

set -uo pipefail
LOCK=/tmp/tanitad_gpu.lock
DEFAULT_TTL=14400                            # 4 h — a lock older than this is stale

# --- v2 (2026-07-21). v1 had TWO defects, both found in use, both fixed here:
#   (a) `acquire` recorded pid=$$ — the SCRIPT's own PID — then exited. The owner
#       was dead the instant the lock existed, so `status` said STALE and the very
#       next caller broke it. The documented usage could not hold a lock at all.
#       FIX: liveness = (a live --pid, if given) OR (still inside its TTL). A
#       standalone acquire now holds for real, and you may pass the long-running
#       job's PID once you have it.
#   (b) `_gpu_busy` counted EVERY compute process, including the caller's own
#       already-running job, so an agent with a campaign in flight could never
#       acquire — exactly the case the lock exists to cover. (`grep -v "^$$\$"`
#       filtered the shell PID, which never appears in nvidia-smi.)
#       FIX: --adopt skips the occupancy check, and --pid values are excluded.

_lock_get() { sed -n "s/^$1=//p" "$LOCK" 2>/dev/null | head -1; }

_owner_alive() {
  [ -f "$LOCK" ] || return 1
  local pid ttl since since_s now
  pid=$(_lock_get pid); ttl=$(_lock_get ttl); since=$(_lock_get since)
  # A live recorded PID is authoritative when one was supplied.
  if [ -n "${pid:-}" ] && [ "$pid" != "none" ] && kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  # Otherwise fall back to the TTL so a standalone acquire still holds.
  since_s=$(date -u -d "${since:-1970-01-01}" +%s 2>/dev/null || echo 0)
  now=$(date -u +%s)
  [ $((now - since_s)) -lt "${ttl:-$DEFAULT_TTL}" ]
}

# Real occupancy, independent of the lockfile — a job that ignored the lock still
# shows up here. Excludes PIDs the caller declares as its own (--pid).
_gpu_busy() {
  local mine="${1:-}" pids n
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  [ -z "$pids" ] && return 1
  n=0
  for p in $pids; do
    case " $mine " in *" $p "*) continue ;; esac
    n=$((n+1))
  done
  [ "$n" -gt 0 ]
}

case "${1:-}" in
  acquire)
    owner="${2:?owner required}"; shift 2
    timeout=7200; adopt=0; mypid="none"; ttl="$DEFAULT_TTL"
    while [ $# -gt 0 ]; do
      case "$1" in
        --adopt) adopt=1 ;;
        --pid) shift; mypid="${1:?--pid needs a value}" ;;
        --ttl) shift; ttl="${1:?--ttl needs a value}" ;;
        *[0-9]*) timeout="$1" ;;
      esac
      shift
    done
    waited=0
    while :; do
      if [ -f "$LOCK" ] && _owner_alive && [ "$(_lock_get owner)" != "$owner" ]; then
        :                                   # held by someone else, still valid
      elif [ "$adopt" -eq 0 ] && _gpu_busy "$mypid"; then
        :                                   # unlocked but occupied anyway
      else
        if [ -f "$LOCK" ] && [ "$(_lock_get owner)" != "$owner" ]; then
          echo "gpu_lock: breaking stale lock ($(head -c 200 "$LOCK" | tr '\n' ' '))" >&2
        fi
        printf 'owner=%s\npid=%s\nsince=%s\nttl=%s\n' \
               "$owner" "$mypid" "$(date -u +%FT%TZ)" "$ttl" > "$LOCK"
        echo "gpu_lock: ACQUIRED by $owner (pid=$mypid ttl=${ttl}s adopt=$adopt)"; exit 0
      fi
      if [ "$waited" -ge "$timeout" ]; then
        echo "gpu_lock: TIMEOUT after ${timeout}s waiting for the GPU" >&2
        echo "gpu_lock: current holder: $(cat "$LOCK" 2>/dev/null | tr '\n' ' ')" >&2
        exit 2
      fi
      sleep 30; waited=$((waited+30))
    done
    ;;
  release)
    owner="${2:?owner required}"
    if [ -f "$LOCK" ] && grep -q "^owner=${owner}\$" "$LOCK"; then
      rm -f "$LOCK"; echo "gpu_lock: released by $owner"
    else
      echo "gpu_lock: NOT released — $owner does not hold it" >&2
      echo "gpu_lock: holder: $(cat "$LOCK" 2>/dev/null | tr '\n' ' ')" >&2; exit 1
    fi
    ;;
  status)
    if [ -f "$LOCK" ]; then
      cat "$LOCK"; _owner_alive && echo "state=HELD" || echo "state=STALE(owner pid gone)"
    else
      echo "state=FREE"
    fi
    echo "gpu_compute_procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')"
    ;;
  *) echo "usage: $0 {acquire <owner> [timeout_s]|release <owner>|status}" >&2; exit 64 ;;
esac

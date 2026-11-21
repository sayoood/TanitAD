#!/usr/bin/env bash
# refcv6 -- run the ordered arm list ONE AT A TIME on a single-GPU box.
#
#   ./chain_refcv6.sh V0 V0b D
#
# ⛔ WHY A CHAIN EXISTS AT ALL, AND WHY IT IS NOT JUST A `for` LOOP.
#    Thor has ONE GPU and its 20 SMs saturate at batch 8. MEASURED: SEVEN
#    concurrent arms sat at 0-6 % GPU for FIFTY MINUTES with zero progress --
#    which looks exactly like a hang, not like contention. ⇒ arms are strictly
#    SEQUENTIAL, and "sequential" has to be ENFORCED, not intended:
#    `launch_refcv6.sh` deliberately refuses to chain ("a chain that launches arm
#    2 while arm 1 is still training is how a box ends up with two trainers on
#    one card").
#
# ⛔ THE HAND-OFF IS AN ARTIFACT, NEVER A STATUS.
#    Arm N+1 starts only when arm N's `summary.json` carries `"done": true` --
#    the marker `sup_refcv6.sh` writes AFTER it has READ `metrics.jsonl` and
#    checked its step count. An exit code is a claim a process makes about
#    itself; the done-marker is the thing we actually wanted.
#    ⛔ And a TIMEOUT is not a completion: if the wait expires with no marker,
#    this script STOPS. It never "assumes it probably finished".
#
# ⛔ AND AN ARTIFACT ALONE IS STILL NOT ENOUGH -- THE BOX IS ALSO ASSERTED EMPTY.
#    Before every launch this script scans `/proc/*/cmdline` for a live trainer.
#    ⚠️ NOT `pgrep -f` and NOT `ps | grep`: both SELF-MATCH the command line that
#    carried them (the PTY echoes it back), which has three times produced a
#    count that was pure artifact -- once inventing a failure on a healthy run,
#    once reporting "1 supervisor" on an EMPTY box. The search token here is
#    assembled at runtime so this script's own argv cannot contain it.
#
# ⛔ EVERY CHILD GETS `200>&-`, THE SLEEPS INCLUDED.
#    A child inherits every open fd. `200>&-` on the interesting child alone was
#    shipped once and failed the same day -- the lock was held by a `sleep 180`
#    that had outlived its parent, and no replacement chain could ever start.
#
# ⚠️ NOTHING HERE RE-DECIDES AN ARM. Flags come from `arms.py` via
#    `launch_refcv6.sh`, which runs the FULL pre-launch gate for each arm --
#    including a 20-step smoke of THAT arm's own configuration -- and refuses on
#    anything but `"verdict": "PASS"` in the gate's JSON.

set -u

ARMS_LIST=("$@")
[ "${#ARMS_LIST[@]}" -gt 0 ] || { echo "usage: chain_refcv6.sh <ARM> [<ARM> ...]"; exit 2; }

ROOT="${REFCV6_ROOT:-/workspace/refcv6}"
OUT_ROOT="${REFCV6_OUT_ROOT:-/workspace/experiments}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="${ROOT}/runs.d"
CHAIN_LOG="${ROOT}/chain_refcv6.log"
LOCK="${ROOT}/chain.lock"
# ~60 h at the default poll: an arm is ~46 h, so this bounds a hang without
# cutting a healthy run short. Override for a short-budget chain.
WAIT_MAX_S="${REFCV6_CHAIN_WAIT_S:-216000}"
POLL_S="${REFCV6_CHAIN_POLL_S:-300}"

mkdir -p "$ROOT" "$RUNS_DIR"
log() { echo "[chain] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$CHAIN_LOG"; }

# ---- one chain at a time --------------------------------------------------- #
exec 200>"$LOCK"
if ! flock -n 200; then
  log "another chain holds ${LOCK} -- exiting."
  log "   If nothing is actually chaining, the holder may be an INHERITED fd."
  log "   Name it: for p in /proc/[0-9]*/fd/*; do [ \"\$(readlink \$p 2>/dev/null)\" = \"${LOCK}\" ] && tr '\\0' ' ' < /proc/\$(echo \$p|cut -d/ -f3)/cmdline && echo; done"
  exit 3
fi
log "lock acquired on ${LOCK} (fd 200); chain pid $$; arms: ${ARMS_LIST[*]}"

# ---- is any trainer alive? read /proc, never `ps | grep` -------------------- #
# Returns the pid of a live trainer, or nothing. The token is assembled here so
# this script's own command line cannot match itself.
live_trainer_pid() {
  local tok
  tok="$(printf 'refc_v3')_$(printf 'train.py')"
  local p pid cmd
  for p in /proc/[0-9]*/cmdline; do
    pid="${p#/proc/}"; pid="${pid%/cmdline}"
    [ "$pid" = "$$" ] && continue
    cmd="$(tr '\0' ' ' < "$p" 2>/dev/null)" || continue
    case "$cmd" in
      *"$tok"*)
        # ⛔ the GATE's own argv contains the trainer path too (--trainer <path>).
        #    A gate is not a training run; excluding it here is why this check
        #    does not block on its own pre-flight.
        case "$cmd" in *prelaunch_gate*) continue ;; esac
        echo "$pid"; return 0 ;;
    esac
  done
  return 1
}

done_marker_ok() {                       # $1 = arm
  local m="${OUT_ROOT}/refcv6-${1}/summary.json"
  [ -s "$m" ] || return 1
  grep -q '"done"[[:space:]]*:[[:space:]]*true' "$m" 2>/dev/null
}

# ---- the chain ------------------------------------------------------------- #
for ARM in "${ARMS_LIST[@]}"; do
  if done_marker_ok "$ARM"; then
    log "ZZCHAIN-${ARM}-ALREADY_DONEZZ  (done-marker present; skipping)"
    continue
  fi

  # 1. the box must be EMPTY before we start anything.
  tp="$(live_trainer_pid || true)"
  if [ -n "$tp" ]; then
    log "⛔ REFUSING to launch ${ARM}: a trainer is ALIVE at pid ${tp}."
    log "   One GPU, one arm. MEASURED: 7 concurrent arms = 0-6 % GPU for 50 min."
    log "   ⛔ Kill by EXPLICIT pid if that process is stale; never pkill -f."
    exit 4
  fi

  log "ZZCHAIN-${ARM}-LAUNCHINGZZ"
  # ⛔ No extra flags are forwarded. Every per-arm value that is not BASE is a PI
  #    decision (`--w-agent`, `--w-tac-goal`) and arms.py REFUSES to invent one,
  #    so an arm needing one cannot be chained silently -- it fails loudly at the
  #    argv build. V0 / V0b / D need none.
  bash "${HERE}/launch_refcv6.sh" --arm "$ARM" >> "$CHAIN_LOG" 2>&1 200>&-
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "⛔ launch_refcv6.sh returned ${rc} for ${ARM}. ⚠️ The rc is a HINT; the"
    log "   admissible evidence is ${ROOT}/raw/prelaunch_gate_${ARM}.json --"
    log "   read its 'verdict' field. A missing file means the gate never ran."
    log "ZZCHAIN-${ARM}-LAUNCH_FAILED-${rc}ZZ"
    exit 5
  fi

  # 2. wait for the ARTIFACT. ⛔ A timeout STOPS the chain; it never advances.
  waited=0
  while ! done_marker_ok "$ARM"; do
    if [ "$waited" -ge "$WAIT_MAX_S" ]; then
      log "⛔ ${ARM} produced NO done-marker after ${waited}s. STOPPING."
      log "   The ABSENCE of ${OUT_ROOT}/refcv6-${ARM}/summary.json is the"
      log "   admissible evidence that this arm did not finish. Not advancing."
      log "ZZCHAIN-${ARM}-TIMEOUTZZ"
      exit 6
    fi
    # progress marker: counts computed HERE, emitted as an opaque token.
    step="-1"
    mj="${OUT_ROOT}/refcv6-${ARM}/metrics.jsonl"
    if [ -s "$mj" ]; then
      step="$(python3 - "$mj" <<'PYEOF' 200>&-
import json, sys
s = []
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
                s.append(r["step"])
except Exception:
    pass
print(max(s) if s else -1)
PYEOF
)"
    fi
    log "ZZCHAIN-${ARM}-${step}-${waited}ZZ"
    sleep "$POLL_S" 200>&-
    waited=$((waited + POLL_S))
  done
  log "ZZCHAIN-${ARM}-DONEZZ  (done-marker present after ${waited}s)"

  # 3. the box must be empty again before the next arm.
  tp="$(live_trainer_pid || true)"
  if [ -n "$tp" ]; then
    log "⚠️ ${ARM} is marked done but a trainer is still alive at pid ${tp}."
    log "   Waiting for it rather than starting a second arm beside it."
    while [ -n "$tp" ] && [ "$waited" -lt "$WAIT_MAX_S" ]; do
      sleep 60 200>&-
      waited=$((waited + 60))
      tp="$(live_trainer_pid || true)"
    done
    [ -n "$tp" ] && { log "⛔ trainer ${tp} still alive; STOPPING."; exit 7; }
  fi
done

log "ZZCHAIN-ALL_DONE-${#ARMS_LIST[@]}ZZ"
exit 0

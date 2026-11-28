#!/bin/bash
# Supervisor / WATCHDOG for refcv5-v2 (P1-OUT arm, TACGOAL=1).
#
# ⛔ THIS DOES NOT PERFORM THE FIRST LAUNCH. The arm was launched by the
# prepared `LAUNCH_refcv5_v2.sh` (P1=out TACGOAL=1) at 2026-09-06T23:10:12Z as
# pid 2559052. This process ADOPTS that trainer -- `trainer_pid` finds it and
# the loop only polls -- and relaunches ONLY if it dies.
#
# The four measured supervisor traps from sup_refcv5.sh are kept UNCHANGED:
#  1. SOURCES ITS MANIFEST ONCE -- but stronger here: the relaunch argv is read
#     from THE RUN'S OWN config.json, so this file cannot drift from what was
#     actually launched. There is no second copy of the command to go stale.
#  2. RESURRECTS A FINISHED RUN -- done condition derived from metrics.jsonl.
#  3. RACES / INHERITS ITS OWN flock -- `200>&-` on the trainer AND every sleep.
#  4. CRASH LOOP READS LIKE SUPERVISION -- 3 relaunches with no step progress
#     and this stops.
#
# ⛔ The trainer AUTO-RESUMES from $OUT/ckpt.pt with a STRICT load
# (refc_v3_train.py:3634-3640), so a relaunch continues the same run rather
# than restarting it.
set -uo pipefail

OUT=/workspace/experiments/refcv5-v2-noagents-b1-v72-40k
STACK=/workspace/TanitAD/stack
TARGET=40284
MAX_RELAUNCH=40
POLL=120
LOCK=/workspace/.sup_refcv5_v2.lock
SLOG=$OUT/supervisor.log

mkdir -p "$OUT"
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING: another supervisor holds $LOCK -- not starting a second" >&2
  exit 3
fi

log() { echo "[sup $(date -u +%FT%TZ)] $*" >> "$SLOG"; }

last_step() {
  # MUST RETURN EXACTLY ONE INTEGER -- see sup_refcv5.sh's note on pipefail.
  s=$( { cat "$OUT/metrics.jsonl" 2>/dev/null || true; } | tail -n 200 | python3 -c '
import sys, json
s = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        s = max(s, int(json.loads(line).get("step", 0)))
    except Exception:
        pass
print(s)
' 2>/dev/null | tail -n 1 )
  case "$s" in
    ""|*[!0-9]*) s=0 ;;
  esac
  echo "$s"
}

trainer_pid() {
  # character-class regex so grep cannot match its OWN argv, keyed on THIS
  # run's out-dir so a sibling refc_v3_train is never adopted.
  ps -eo pid=,args= | grep 'refcv5[-]v2-noagents-b1-v72-40k' | grep 'refc[_]v3[_]train' \
    | grep -v ' grep ' | awk '{print $1}' | head -1
}

launch() {
  cd "$OUT" || return 1
  # THE RELAUNCH USES THE RUN'S OWN RECORDED argv. One source of truth.
  mapfile -t A < <(python3 -c 'import json,sys
print("\n".join(json.load(open(sys.argv[1]))["argv"]))' "$OUT/config.json" 2>/dev/null)
  if [ "${#A[@]}" -lt 20 ]; then
    log "REFUSING relaunch: argv read from config.json has ${#A[@]} entries (expected >= 20) -- INCONCLUSIVE, not a short command"
    return 1
  fi
  PYTHONPATH="$STACK" nohup python3 -u "$STACK/scripts/refc_v3_train.py" "${A[@]}" \
    >> "$OUT/train.log" 2>> "$OUT/train.stderr.log" 200>&- &
  echo $!
}

# REFUSE TO SUPERVISE WITHOUT THE VOCABULARY (the refcv3 fall-back defect).
if [ ! -s "$OUT/anchors.pt" ]; then
  echo "REFUSING: $OUT/anchors.pt missing -- that is the refcv3 defect" >&2
  exit 5
fi
if ! PYTHONPATH="$STACK" python3 -c '
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
c = d.get("controls") if isinstance(d, dict) else None
assert c is not None and c.ndim == 2 and c.shape[1] == 2, "no controls [N, 2]"
assert bool(((c[:, 0] == 0) & (c[:, 1] == 0)).any()), "no {a=0, kappa=0}"
print("anchors OK: %s controls, straight-ahead present" % (tuple(c.shape),))
' "$OUT/anchors.pt"; then
  echo "REFUSING: $OUT/anchors.pt is not a v0-conditioned vocabulary" >&2
  exit 6
fi

log "watchdog up for refcv5-v2 (P1 OUT, --agents off; P14 --sel-refined --sel-score-emitted => sampler_ranks_the_fan True; --tac-goal-tok-head ON, rollability MEASURED 0 missing / 0 unexpected; anchors n=117 units=alat); target $TARGET; adopted pid $(trainer_pid); resume point $(last_step)"
n=0
stall=0
prev_st=-1
while true; do
  st=$(last_step)
  if [ "${st:-0}" -ge "$TARGET" ]; then
    log "DONE: metrics.jsonl reports step $st >= $TARGET -- exiting cleanly"
    echo "{\"done\": true, \"final_step\": $st, \"target\": $TARGET}" > "$OUT/summary.json"
    exit 0
  fi
  pid=$(trainer_pid)
  if [ -z "$pid" ]; then
    if [ "$st" = "$prev_st" ]; then
      stall=$((stall + 1))
    else
      stall=0
      prev_st=$st
    fi
    if [ "$stall" -ge 3 ]; then
      log "REFUSING: 3 consecutive relaunches with no step progress (stuck at $st) -- crash loop, not a flaky run"
      echo "{\"done\": false, \"crash_loop\": true, \"stuck_at_step\": $st}" > "$OUT/summary.json"
      exit 7
    fi
    if [ "$n" -ge "$MAX_RELAUNCH" ]; then
      log "REFUSING further relaunches: $n reached MAX_RELAUNCH at step $st"
      exit 4
    fi
    n=$((n + 1))
    newpid=$(launch)
    log "relaunch #$n from step $st -> pid $newpid (stall=$stall)"
    sleep 90 200>&-
  else
    stall=0
    prev_st=$st
    sleep "$POLL" 200>&-
  fi
done

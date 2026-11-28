#!/bin/bash
# Supervisor for REF-C v4b — the v0-CONDITIONED-vocabulary relaunch.
# (refcv4b-b1-v72-40k). Adapted from sup_refcv4.sh, which is kept verbatim
# except where marked; the three measured supervisor traps are UNCHANGED:
#  1. "SOURCES ITS MANIFEST ONCE" — the command is EMBEDDED, so what you read
#     is what runs; to change it, stop this supervisor and edit this file.
#  2. "RESURRECTS A FINISHED RUN" — the done condition is DERIVED FROM THE DATA
#     (last step in metrics.jsonl >= TARGET), and the done-marker is written in
#     the same breath as the decision.
#  3. "RACES / INHERITS ITS OWN flock" — `200>&-` on the trainer AND on every
#     `sleep`, because EVERY child inherits the lock fd, not just the
#     interesting one (measured 2026-09-02: a `sleep 180` held the lock after
#     its parent died and made the run permanently unsupervisable).
#
# WHAT IS NEW IN v4b vs sup_refcv4.sh — TWO changes, both PI-directed:
#   --anchor-v0-conditioned + --n-anchors 117
#                        ⛔ THE REASON FOR THE RESTART. refcv4's vocabulary was
#                        128 FIXED paths in absolute metres; on the banked
#                        4,823-window surface its oracle-in-vocabulary reads
#                        0.3773 m — +0.0777 [+0.0528, +0.1044] SEPARATED WORSE
#                        than hold-action (ha = 0.2996), i.e. its CEILING was
#                        above the bar. This file's vocabulary is 117
#                        v0-CONDITIONED, SPEED-CLAMPED constant-(a_lon, a_lat)
#                        controls: CURVATURE IS DERIVED per window as
#                        a_lat / max(v0, 4.0)^2, so the control space IS the
#                        Kamm-circle space and a bound on the grid is a bound on
#                        the friction circle. Reads 0.1987 m — -0.1009
#                        [-0.1213, -0.0813], BEATS ha, at a SMALLER budget, and
#                        0 of 117 anchors break a mu = 0.7 circle at v0 = 27 m/s
#                        (peak 0.68 g). ⚠️ A FLAT-CURVATURE family of the same
#                        size reads 0.2610 (still beats ha) but breaks 104 of
#                        117 at 3.96 g, because a_lat = v^2 * kappa; it was
#                        built, measured and REJECTED for that reason.
#                        Paired episode-cluster bootstrap, n_boot 2000, seed 0.
#   Defect A resolved    `refc_v3.py` no longer feeds the 8-wide v7 heads to
#                        `derive_man5_logprobs` (a [B,3]x[B,3] positional
#                        contract). H19's anchor prior is now driven by the
#                        core's own 3-wide kin3 heads instead of a scrambled
#                        distribution. Resolution (a) of three, pre-registered.
# Everything else is refcv4's own command, byte for byte.
#
# ⚠️ ATTRIBUTION IS FORFEITED BY DESIGN (PI decision): five-plus levers move at
# once against refcv3. This arm answers "does the stack work", not "which lever".
set -uo pipefail

OUT=/workspace/experiments/refcv4b-b1-v72-40k
STACK=/workspace/TanitAD/stack
TARGET=40284
MAX_RELAUNCH=40
POLL=120
LOCK=/workspace/.sup_refcv4b_v3.lock
SLOG=$OUT/supervisor.log

mkdir -p "$OUT"
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING: another supervisor holds $LOCK — not starting a second" >&2
  exit 3
fi

log() { echo "[sup $(date -u +%FT%TZ)] $*" >> "$SLOG"; }

last_step() {
  # ⛔ MUST RETURN EXACTLY ONE INTEGER. The v1 form was
  # `tail ... | python3 ... || echo 0`. Under `set -o pipefail` a MISSING
  # metrics.jsonl makes `tail` exit 1, so the PIPELINE reports failure even
  # though python already printed a clean 0 -- and the `|| echo 0` fallback
  # then appends a SECOND line. The value becomes two lines, every
  # `[ "$st" -ge "$TARGET" ]` dies with "integer expression expected", and
  # THE DONE-MARKER BRANCH IS NEVER REACHED. That is the missing-done-marker
  # failure (a finished run relaunched 80 times over 2.7 h), reintroduced
  # through a fallback that looks like belt-and-braces. MEASURED here at
  # 2026-09-04T11:24:44Z on the first start of this very run.
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
  # character-class regex so grep cannot match its OWN argv (measured trap: a
  # filter containing the pattern it searches for reports a phantom result),
  # and keyed on THIS run's out-dir so a sibling refc_v3_train is not adopted.
  ps -eo pid=,args= | grep 'refcv4b[-]b1-v72-40k' | grep 'refc[_]v3[_]train' \
    | grep -v ' grep ' | awk '{print $1}' | head -1
}

launch() {
  cd "$OUT" || return 1
  PYTHONPATH="$STACK" nohup python3 -u "$STACK/scripts/refc_v3_train.py" \
    --arm hier --size base \
    --v2-cache /root/data/train \
    --v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz \
    --eval-cache /root/data/eval \
    --eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
    --eval-every 500 --eval-batches 8 \
    --image-hw 256 640 \
    --steps "$TARGET" --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24 \
    --lr 1e-4 --warmup 2000 --seed 0 \
    --log-every 50 --save-every 500 \
    --nav-from-v7 \
    --u8-batches \
    --anchors "$OUT/anchors.pt" \
    --n-anchors 117 \
    --anchor-v0-conditioned \
    --anchor-control-units alat \
    --sel-accel-max 2.0 \
    --goal-str \
    --ego-state-inject --ego-dropout 0.5 \
    --out "$OUT" >> "$OUT/train.log" 2>> "$OUT/train.stderr.log" 200>&- &
  echo $!
}

# ⛔ REFUSE TO LAUNCH WITHOUT THE VOCABULARY. A missing --anchors file is
# exactly how refcv3 spent 53 h on the synthetic set: the trainer would not
# fail, it would fall back. This makes the fallback impossible here.
if [ ! -s "$OUT/anchors.pt" ]; then
  echo "REFUSING: $OUT/anchors.pt missing — that is the refcv3 defect" >&2
  exit 5
fi
# ⛔ AND REFUSE IF IT IS NOT THE v0-CONDITIONED FILE. `--anchor-v0-conditioned`
# with a controls-free file is a SystemExit in the trainer, but a supervisor
# that relaunches into that 40 times is 2 h of thrash the log makes look like
# supervision. Answer it once, here, before the loop.
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

log "supervisor refcv4b up (v0-CONDITIONED anchors=$OUT/anchors.pt, n=117, sel-accel-max 2.0, --goal-str, ego-inject dropout 0.5, NO --echo-base, Defect A resolved); target $TARGET; resume point $(last_step)"
n=0
stall=0
prev_st=-1
while true; do
  st=$(last_step)
  if [ "${st:-0}" -ge "$TARGET" ]; then
    log "DONE: metrics.jsonl reports step $st >= $TARGET — exiting cleanly"
    echo "{\"done\": true, \"final_step\": $st, \"target\": $TARGET}" > "$OUT/summary.json"
    exit 0
  fi
  pid=$(trainer_pid)
  if [ -z "$pid" ]; then
    # ⭐ NEW IN v4b — CRASH-LOOP GUARD. A trainer that dies before writing a
    # single new metrics row will die the same way every time; relaunching it
    # 40 more times produces a log that reads like supervision and a GPU that
    # does nothing. Three relaunches with NO step progress and this stops.
    if [ "$st" = "$prev_st" ]; then
      stall=$((stall + 1))
    else
      stall=0
      prev_st=$st
    fi
    if [ "$stall" -ge 3 ]; then
      log "REFUSING: 3 consecutive relaunches with no step progress (stuck at $st) — this is a crash loop, not a flaky run"
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

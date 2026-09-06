#!/bin/bash
# Supervisor for REF-C v5 — the WP-4 DIFFUSION-SAMPLER rung.
# (refcv5-ddim-b1-v72-40k). Adapted from sup_refcv4b_v3.sh, which is kept
# verbatim except where marked; the four measured supervisor traps are
# UNCHANGED:
#  1. "SOURCES ITS MANIFEST ONCE" — the command is EMBEDDED, so what you read
#     is what runs; to change it, stop this supervisor and edit this file.
#  2. "RESURRECTS A FINISHED RUN" — the done condition is DERIVED FROM THE DATA
#     (last step in metrics.jsonl >= TARGET), and the done-marker is written in
#     the same breath as the decision. The TRAINER also writes its own
#     summary.json on a clean exit (verified in the 3-step smoke:
#     "[v3:hier] DONE at 3 — summary.json written"), so the marker has TWO
#     independent writers and neither depends on the other.
#  3. "RACES / INHERITS ITS OWN flock" — `200>&-` on the trainer AND on every
#     `sleep`, because EVERY child inherits the lock fd, not just the
#     interesting one (measured 2026-09-02: a `sleep 180` held the lock after
#     its parent died and made the run permanently unsupervisable).
#  4. "CRASH LOOP READS LIKE SUPERVISION" — three relaunches with no step
#     progress and this stops, rather than burning 40 relaunches.
#
# WHAT IS NEW IN v5 vs sup_refcv4b_v3.sh — ONE lever moves, deliberately:
#   --sampler ddim --w-u0 0.5
#                        ⭐ THE REASON FOR THIS ARM (WP-4). The metre-space
#                        truncated denoise is replaced by an anchored Gaussian
#                        in CONTROL space, so every sample re-rolls through the
#                        kinematic model and is FLYABLE BY CONSTRUCTION rather
#                        than by penalty. `--w-u0` is the only term that
#                        supervises the sampler in the space it samples in; the
#                        trainer REFUSES `--sampler ddim --w-u0 0` (validation
#                        arm R4), so a non-zero weight is not optional.
#                        0.5 is the value exercised as the honest CONTROL in
#                        the WP-4/WP-6 guard-mutation panel; it is not invented
#                        here.
#   --agents off         ⛔ NOT a preference — a MEASURED blocker. `--agents
#                        oracle` was the intended first rung and it does NOT
#                        run without agent ground truth: the preflight's 2-step
#                        arm died with "this build is `--agents oracle` but no
#                        agent_gt reached the forward". The oracle's tokens ARE
#                        the GT boxes, so it needs --agent-join exactly as
#                        --agents head does; the brief's "oracle needs no
#                        detector" is true about the DETECTOR and false about
#                        the JOIN. No join exists on this pod, and the only one
#                        the programme holds is PARITY-scoped (train2400,
#                        2,308 clips) which covers ~4 % of this B1 corpus.
#                        ⇒ WP-6 is deferred to its own rung behind a B1-scoped
#                        join. Attribution is CLEANER this way, not worse: one
#                        lever moves against refcv4b.
#   --anchor-control-units alat
#                        the refcv4b bank carries `controls` and declares NO
#                        `control_units`. That is the 396 g / 0.31 g incident's
#                        exact shape, and the trainer REFUSES it unless the
#                        units are passed BY NAME. They are passed here, and
#                        the run record says they came from the operator
#                        (`control_units_source: cli-override-legacy-file`),
#                        not from the artifact. The units are not guessed: the
#                        SAME tensor trained 40,284 steps as refcv4b under
#                        `alat` (its config.json records control_units "alat"
#                        with anchors sha256 51f930dc6f3564ff…).
#
# Everything else is refcv4b's own command, byte for byte, on the SAME corpus
# and the SAME label file — so refcv5 vs refcv4b is a matched-step, matched-
# corpus comparison in which ONE mechanism moved.
#
# ⛔ --sel-refined IS NOT PASSED, and cannot be: the flag does not exist in
# this trainer (grep "sel_refined" reads 0 against a control "sel_accel_max"
# reading 5). MEASURED on refcv4b it was 0.0259 m separated WORSE, because the
# ranking head was never trained to rank.
set -uo pipefail

OUT=/workspace/experiments/refcv5-ddim-b1-v72-40k
STACK=/workspace/TanitAD/stack
TARGET=40284
MAX_RELAUNCH=40
POLL=120
LOCK=/workspace/.sup_refcv5.lock
SLOG=$OUT/supervisor.log

mkdir -p "$OUT"
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING: another supervisor holds $LOCK — not starting a second" >&2
  exit 3
fi

log() { echo "[sup $(date -u +%FT%TZ)] $*" >> "$SLOG"; }

last_step() {
  # ⛔ MUST RETURN EXACTLY ONE INTEGER. Under `set -o pipefail` a MISSING
  # metrics.jsonl makes `tail` exit 1, so a `|| echo 0` fallback appends a
  # SECOND line, every `[ "$st" -ge "$TARGET" ]` dies with "integer expression
  # expected", and THE DONE-MARKER BRANCH IS NEVER REACHED. Kept in the fixed
  # form from sup_refcv4b_v3.sh.
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
  ps -eo pid=,args= | grep 'refcv5[-]ddim-b1-v72-40k' | grep 'refc[_]v3[_]train' \
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
    --sampler ddim \
    --w-u0 0.5 \
    --agents off \
    --out "$OUT" >> "$OUT/train.log" 2>> "$OUT/train.stderr.log" 200>&- &
  echo $!
}

# ⛔ REFUSE TO LAUNCH WITHOUT THE VOCABULARY. A missing --anchors file is
# exactly how refcv3 spent 53 h on the synthetic set: the trainer would not
# fail, it would fall back (oracle-in-vocabulary 1.0882 m vs 0.3796 m).
if [ ! -s "$OUT/anchors.pt" ]; then
  echo "REFUSING: $OUT/anchors.pt missing — that is the refcv3 defect" >&2
  exit 5
fi
# ⛔ AND REFUSE IF IT IS NOT THE v0-CONDITIONED FILE.
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
# ⭐ NEW IN v5 — REFUSE IF THE WP-4 SEAM IS NOT IN THE SHIPPED STACK. Pods have
# no git credentials and their checkouts drift silently; a launch from a stale
# tree would run refcv4b's code under refcv5's name and the config.json would
# still say `sampler: ddim`. This is a POSITIVE content assertion with a
# same-breath control that must read non-zero.
if ! PYTHONPATH="$STACK" python3 -c '
import sys
sys.path.insert(0, "'"$STACK"'")
from tanitad.refs import refc_sampler, refc_agents
for n in ("DDIMSchedule", "roll_controls"):
    assert hasattr(refc_sampler, n), "refc_sampler lacks " + n
assert hasattr(refc_agents, "build_agent_head"), "refc_agents lacks build_agent_head"
print("WP-4/WP-6 seam present: DDIMSchedule, roll_controls, build_agent_head")
'; then
  echo "REFUSING: the WP-4 sampler seam is absent from $STACK — stale pod checkout" >&2
  exit 8
fi

log "supervisor refcv5 up (WP-4 DDIM in CONTROL space, w_u0 0.5, agents OFF pending a B1-scoped join, v0-conditioned anchors=$OUT/anchors.pt n=117 units=alat via cli-override-legacy-file, sel-accel-max 2.0, --goal-str, ego-inject dropout 0.5, NO --sel-refined); target $TARGET; resume point $(last_step)"
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
    # CRASH-LOOP GUARD. A trainer that dies before writing a single new
    # metrics row will die the same way every time; relaunching it 40 more
    # times produces a log that reads like supervision and a GPU that does
    # nothing. Three relaunches with NO step progress and this stops.
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

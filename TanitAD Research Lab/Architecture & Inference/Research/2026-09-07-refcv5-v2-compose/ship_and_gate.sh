#!/usr/bin/env bash
# ============================================================================ #
# ship_and_gate.sh — THE PRE-LAUNCH GATE for refcv5-v2.  RUN THIS FIRST.
#
# ⛔⛔ THIS IS THE STEP THAT WAS MISSING, AND IT IS THE REAL BLOCKER.
# MEASURED 2026-09-06 22:0xZ on tanitad-a40:
#
#   pod  /workspace/TanitAD/stack/scripts/refc_v3_train.py
#        3,634 lines · md5 acad9ae6e336e52d1b61a314f93a39fa · 2026-09-05 23:38Z
#        add_argument 79 · sel_refined 0 · sel_score_emitted 0 · CONTROL sel_ 8
#   repo HEAD (52ca682) same path
#        4,479 lines · add_argument 88 · sel_refined 6 · sel_score_emitted 16
#        CONTROL sel_ 33
#
# ⇒ the pod is POD-BEHIND by 845 lines and NINE flags, and every one of them is
#   a lever this composition depends on:
#       --sel-refined --sel-score-emitted --sel-score-emitted-t   (P14)
#       --no-strategic                                            (P8)
#       --goal-point-inject/-geo-prior/-t/-w                      (P2)
#       --nav-args                                                (P11)
#
# ⚠️ AND IT EXPLAINS THE PLAN'S OWN NUMBER. REFCV5_MISSING_PIECES_PLAN.md §8
# P14 records `sel_refined`/`sel_score_emitted` at ZERO with a control of
# `sel_` = 8. **8 is the POD's control value, not HEAD's (33).** That census
# was taken against the deployed copy. The fix LANDED in the repo at c8601d0
# and was never shipped to the box — exactly the repo→box gap
# AGENT_OPERATING_STANDARD.md names as C99/C102/C105.
#
# ⇒ Launching without this step would silently reproduce refcv5-v1's defect:
#   argparse would REJECT --sel-refined outright (loud, good), or — if someone
#   dropped the flag to "make it start" — the arm would ship
#   `sampler_ranks_the_fan: False` again.
#
# ⛔ DO NOT RUN THIS UNTIL THE SIBLING'S TRAINER EDITS HAVE LANDED IN HEAD.
#    As of 2026-09-06 late, `stack/scripts/refc_v3_train.py` is `MM` in the
#    worktree (4,576 lines) and `--tac-goal-tok-head` exists ONLY there.
#    Shipping mid-edit is how you get a half-landed trainer.
# ============================================================================ #
set -euo pipefail

HOST="${HOST:-tanitad-a40}"
REPO="${REPO:-G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD}"
POD_STACK=/workspace/TanitAD/stack

say() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------------------
# 0. REFUSE to ship a dirty trainer. The whole point is shipping a KNOWN ref.
# ---------------------------------------------------------------------------
say "0. repo state"
cd "$REPO"
git --no-optional-locks rev-parse --short HEAD
if ! git --no-optional-locks diff --quiet -- stack/scripts/refc_v3_train.py; then
  echo "⛔ REFUSING: stack/scripts/refc_v3_train.py has UNSTAGED edits." >&2
  echo "   A sibling is mid-edit. Ship a committed ref, never a worktree." >&2
  exit 3
fi

# ---------------------------------------------------------------------------
# 1. SHIP the python surface from the REF (not the worktree).
#    ⚠️ `git archive HEAD` reads blobs, so a flapping G: mount cannot hand us a
#    half-written file — the failure mode that truncated a file to 0 bytes.
# ---------------------------------------------------------------------------
say "1. ship stack/ from HEAD"
TGZ=/tmp/stack_refcv5_v2.tgz
# ⛔ MEASURED 2026-09-06: `core.autocrlf=true` on this dev box and `git archive`
#    HONOURS it. The packed trainer came out 270,331 B / md5 30889783... against the
#    blob's 265,852 B / cee9196b... -- exactly +4,479 bytes, ONE CR PER LINE. Step 2
#    then compares the pod against `git show | md5sum` (an LF blob) and FAILS, with a
#    signature that reads like a transfer fault. Without this pin the script cannot
#    pass its own gate on this box.
git --no-optional-locks -c core.autocrlf=false -c core.eol=lf archive HEAD stack/tanitad stack/scripts -o "$TGZ"
ls -la "$TGZ"
scp -o Compression=no "$TGZ" "$HOST":/workspace/stack_refcv5_v2.tgz
ssh -n "$HOST" 'cd /workspace/TanitAD && tar -xzf /workspace/stack_refcv5_v2.tgz && echo EXTRACT-OK'

# ---------------------------------------------------------------------------
# 2. VERIFY BY CONTENT, with a same-breath control that must read non-zero.
#    ⛔ "presence proves transfer, md5 proves bytes, a successful import proves
#    loading — none of them proves currency" (AGENT_OPERATING_STANDARD.md).
# ---------------------------------------------------------------------------
say "2. verify by md5 against the REF's blob"
WANT=$(git --no-optional-locks show HEAD:stack/scripts/refc_v3_train.py | md5sum | cut -d' ' -f1)
GOT=$(ssh -n "$HOST" "md5sum $POD_STACK/scripts/refc_v3_train.py | cut -d' ' -f1")
echo "repo HEAD md5: $WANT"
echo "pod      md5: $GOT"
[ "$WANT" = "$GOT" ] || { echo "⛔ MD5 MISMATCH — the box is not running the ref." >&2; exit 4; }

say "2b. P14 markers present on the pod, WITH the non-zero control"
ssh -n "$HOST" "F=$POD_STACK/scripts/refc_v3_train.py; \
  echo \"sel_refined        : \$(grep -c sel_refined \$F)        (must be > 0)\"; \
  echo \"sel_score_emitted  : \$(grep -c sel_score_emitted \$F)  (must be > 0)\"; \
  echo \"CONTROL sel_       : \$(grep -c sel_ \$F)               (must be ~33, NOT 8)\"; \
  echo \"add_argument       : \$(grep -c add_argument \$F)       (must be >= 88)\""

say "2c. BEHAVIOUR: --help ON THE POD -- the step that actually catches a dead trainer"
# ⛔⛔ THIS STEP EXISTS BECAUSE 2 AND 2b PASSED ON A TRAINER THAT COULD NOT START.
# MEASURED 2026-09-06: the shipped md5 matched HEAD's blob EXACTLY and the marker
# census read sel_ 33 / sel_refined 6 / add_argument 88 -- and `--help` raised
#     ImportError: cannot import name 'goal_point' from 'tanitad.refs'
# because a HEAD file imported a module that was STAGED AND NEVER COMMITTED.
# Presence proves transfer; md5 proves bytes; a grep census proves text. NONE of
# them proves the program RUNS.
#
# ⚠️ THE must-read-NON-ZERO CONTROLS BELOW ARE LOAD-BEARING. When the trainer
# raises, --help writes nothing and EVERY grep reads 0 -- so nine zeros would be
# filed as "nine flags ABSENT", the exact inversion of the truth. If any control
# reads 0 the flag census is INCONCLUSIVE, not negative. Ship it back to a human.
cat > /tmp/refcv5_help_gate.sh <<'GATE'
set -u
STACK=/workspace/TanitAD/stack
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=$STACK
python3 $STACK/scripts/refc_v3_train.py --help > /tmp/pod_help.txt 2>/tmp/pod_help.err
RC=$?
echo "HELP_RC=$RC   (must be 0)"
echo "help bytes: $(wc -c < /tmp/pod_help.txt)   (0 means it did NOT parse)"
echo '--- stderr (must be empty) ---'
head -20 /tmp/pod_help.err
echo '--- NINE FLAGS (each must be >= 1) ---'
for f in --sel-refined --sel-score-emitted --sel-score-emitted-t --no-strategic \
         --goal-point-inject --goal-point-geo-prior --goal-point-t \
         --goal-point-w --nav-args; do
  printf '  %-26s %s\n' "$f" "$(grep -c -- "$f" /tmp/pod_help.txt)"
done
echo '--- NEGATIVE CONTROLS (each must be 0) ---'
for f in --max-speed-input --tac-goal-tok-head --str-goal-tok-head; do
  printf '  %-26s %s\n' "$f" "$(grep -c -- "$f" /tmp/pod_help.txt)"
done
echo '--- PROBE-IS-WORKING CONTROLS (each must be > 0, else INCONCLUSIVE) ---'
for f in usage: --agents --anchors --steps; do
  printf '  %-26s %s\n' "$f" "$(grep -c -- "$f" /tmp/pod_help.txt)"
done
test $RC -eq 0
GATE
scp -q /tmp/refcv5_help_gate.sh "$HOST":/tmp/refcv5_help_gate.sh
ssh -n "$HOST" 'bash /tmp/refcv5_help_gate.sh' || {
  echo "⛔ THE TRAINER DID NOT PARSE ON THE POD. The bytes arrived; the program is dead." >&2
  exit 6; }

# ---------------------------------------------------------------------------
# 3. THE WHOLE-SUBTREE CURRENCY AUDIT — the rung md5-on-one-file cannot reach.
#    MEASURED 2026-09-04: a .py sweep read 738/830 IDENTICAL and looked fine
#    while a DATA sidecar a safety oracle loads was absent from the box.
# ---------------------------------------------------------------------------
say "3. pod_currency_audit (repo -> box, whole subtree, by CONTENT)"
python stack/scripts/pod_currency_audit.py --host "$HOST" || {
  echo "⛔ CURRENCY AUDIT FAILED — resolve before launching." >&2; exit 5; }

# ---------------------------------------------------------------------------
# 4. THE ZERO-GPU PREFLIGHT. Every weight that reaches config.json must reach a
#    loss with a NON-ZERO GRADIENT. An ABSENT input is INCONCLUSIVE = FAILURE.
# ---------------------------------------------------------------------------
say "4. refcv5_preflight on the pod"
ssh -n "$HOST" "cd /workspace && PYTHONPATH=$POD_STACK python3 $POD_STACK/scripts/refcv5_preflight.py \
  --v2-cache /root/data/train \
  --agent-join /workspace/TanitAD/data/joins/b1train_agents.jsonl.xz \
  --v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz \
  --anchors /workspace/anchors_117_alat_declared.pt \
  --json /workspace/refcv5_v2_preflight.json"

say "GATE PASSED — refcv5-v2 may launch."

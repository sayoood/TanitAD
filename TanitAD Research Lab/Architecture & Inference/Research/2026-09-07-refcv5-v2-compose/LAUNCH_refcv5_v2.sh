#!/usr/bin/env bash
# ============================================================================ #
# LAUNCH_refcv5_v2.sh — the composed arm.  ⛔ NOT LAUNCHED BY ITS AUTHOR.
#
# ⛔⛔ THE GATE (REFCV5_MISSING_PIECES_PLAN.md §4, restated §8.2):
#     the composed arm DOES NOT LAUNCH until P1 (agent conditioning) is either
#     IN or explicitly declared OUT by the PI. A sibling owns that gate.
#
# ⛔ AND ship_and_gate.sh MUST HAVE PASSED. The pod was measured 845 lines and
#    nine flags BEHIND HEAD on 2026-09-06; without the ship step argparse will
#    reject --sel-refined and the arm cannot carry P14 at all.
#
# USAGE (pick exactly one):
#     P1=in  bash LAUNCH_refcv5_v2.sh     # agent conditioning IN
#     P1=out bash LAUNCH_refcv5_v2.sh     # agent conditioning explicitly OUT
# ============================================================================ #
set -euo pipefail

P1="${P1:?set P1=in or P1=out — this arm has no default, by design}"
STACK=/workspace/TanitAD/stack
TARGET=40284
DATA=/workspace/TanitAD/data

case "$P1" in
  in)  RUN=refcv5-v2-agents-b1-v72-40k ;;
  out) RUN=refcv5-v2-noagents-b1-v72-40k ;;
  *)   echo "P1 must be 'in' or 'out', got '$P1'" >&2; exit 2 ;;
esac
OUT=/workspace/experiments/$RUN

# --------------------------------------------------------------------------- #
# REFUSALS — every one of these has cost the programme a run.
# --------------------------------------------------------------------------- #
[ -d "$OUT" ] && { echo "⛔ $OUT exists. Refusing to overwrite a run's ckpt/config/metrics." >&2; exit 3; }
mkdir -p "$OUT"

# ⛔ REFUSE WITHOUT THE VOCABULARY. A missing --anchors is exactly how refcv3
#    spent 53 h on the synthetic set: the trainer does not fail, it FALLS BACK
#    (oracle-in-vocabulary 1.0882 m vs 0.3796 m).
# ⭐ anchors_117_alat_declared.pt is refcv4b's OWN bank, restamped with
#    control_units + sha256 provenance. Same 117 anchors as the BASELINE, so
#    the anchor vocabulary is HELD CONSTANT across the comparison.
cp /workspace/anchors_117_alat_declared.pt "$OUT/anchors.pt"
PYTHONPATH="$STACK" python3 -c "
import torch,sys
d=torch.load('$OUT/anchors.pt',map_location='cpu',weights_only=True)
assert d['controls'].shape==(117,2), d['controls'].shape
assert d['control_units']=='alat', d['control_units']
assert d['straight_ahead_control_present'] is True
print('[gate] anchors OK:',d['controls'].shape,'units',d['control_units'])"

if [ "$P1" = in ]; then
  JOIN=$DATA/joins/b1train_agents.jsonl.xz
  [ -s "$JOIN" ] || { echo "⛔ agent join missing: $JOIN" >&2; exit 4; }
  # md5 is the artifact's identity; the join was built dev-box CPU, zero GPU.
  echo "1c985e6d6ad34e605c4ebd30cb353558  $JOIN" | md5sum -c - \
    || { echo "⛔ agent join md5 MISMATCH" >&2; exit 5; }
fi

# --------------------------------------------------------------------------- #
# THE ARM
# --------------------------------------------------------------------------- #
cd "$OUT"
ARGS=(
  # ---- identity: UNCHANGED from refcv5-v1, so the delta is the levers ------
  --arm hier --size base
  --v2-cache /root/data/train
  --v7-labels "$DATA/s2_labels_v7.2_train.jsonl.gz"
  --eval-cache /root/data/eval
  --eval-labels "$DATA/s2_labels_v7.2_eval.jsonl.gz"
  --eval-every 500 --eval-batches 8
  --image-hw 256 640
  --steps "$TARGET" --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24
  --lr 1e-4 --warmup 2000 --seed 0
  --log-every 50 --save-every 500
  --u8-batches
  --out "$OUT"

  # ---- INPUT 1/3: the nav command ----------------------------------------- #
  # ⚠️ PROVENANCE WARNING, and it is NOT cosmetic. All 4,719 v7.2 nav records
  # carry provenance "ego-future" (vocab_v7.py:314-327 calls that
  # "USABLE FOR TRAINING ONLY"); the loader is called with
  # allow_oracle_nav=True (refc_v3_train.py:3461, :3544). ⇒ this arm is an
  # ORACLE-NAV arm, not the production-nav arm the vocabulary rule describes.
  # Kept because it is refcv4b's OWN input (comparability), and STAMPED.
  --nav-from-v7

  # ---- INPUT 2/3: measured v0 at t0 --------------------------------------- #
  # E11': the MEASURED t0 ego block. --ego-valid-channel is set as a
  # PRECONDITION, not an option (refc_v3_train.py:234) — X15: v0 is exactly
  # 0.0 on 4.4531 % of windows, so without the bit "stopped" and "dropped out"
  # are the same token, 23.5x apart between train and eval.
  --ego-state-inject --ego-dropout 0.5

  # ---- the v0-conditioned vocabulary (refcv4b's, held constant) ----------- #
  --anchors "$OUT/anchors.pt"
  --n-anchors 117
  --anchor-v0-conditioned
  --anchor-control-units alat     # file declares 'alat' too => source 'file+cli'
  --sel-accel-max 2.0

  # ---- the sampler -------------------------------------------------------- #
  --sampler ddim
  --w-u0 0.5                      # 'ddim' needs a v0-conditioned vocab AND w_u0>0

  # ---- ⭐ P14 — THE ONE VALIDATED NEW LEVER ------------------------------- #
  # Ceiling ADE 0.4728 -> 0.1914 m, +0.2813 [+0.2127, +0.3543] separated,
  # n = 881 windows / 40 episodes, paired episode-cluster bootstrap.
  # ⛔ THE TWO ARE A PAIR AND THE TRAINER REFUSES THEM SPLIT: --sel-refined
  # ALONE is the MEASURED-HARMFUL lever (0.0259 m separated WORSE, 29.82 % of
  # picks flipped). On a ddim arm sel_score_emitted_t auto-corrects -1 -> 0 and
  # STAMPS the correction as 'auto-zero-on-ddim'.
  # ⚠️ 98.9 % OF THE GAIN IS ALONG-TRACK. This is a LONGITUDINAL lever.
  # Curvature is NOT separated. It must never be sold as a lateral fix.
  --sel-refined --sel-score-emitted

  # ---- the strategic goal head (geometric), as in refcv5-v1 --------------- #
  # ⚠️ This is the 3-unit BEARING head str_goal_head, NOT the 15-token
  # vocabulary. The 15-token head (P4) is NOT WIRED — see README §4.
  --goal-str
)

# ⭐ THE STRATEGIC LAYER'S CONDITIONING STAYS ON. --no-strategic is the
# ABLATION arm and is deliberately absent here: the PI asked for the vocabulary
# to be USED; the ablation answers a different question (plan §0.1).

if [ "$P1" = in ]; then
  ARGS+=(
    # ---- ⭐ P1 — agent / environment conditioning ------------------------- #
    # 'head' = the LEARNED monocular 3D head: vision-only at inference,
    # obstacle.offline as TRAIN-TIME labels. ⛔ NOT 'oracle' — that feeds GT
    # boxes at inference and is INADMISSIBLE as a capability claim.
    --agents head
    --w-agent 1.0                 # ⚠️ CONVENTION, not a validated value
    --agent-join "$DATA/joins/b1train_agents.jsonl.xz"
    --agent-queries 100           # M17: train max 94 over 12,122,129 boxes
    # --agent-w-project / --agent-w-ground stay 0.0: the preflight MEASURED
    # w-ground computing a TAUTOLOGY (2.6e-08, gradient 8.7e-11).
  )
else
  ARGS+=(--agents off)
fi

# ⛔ --require-parity is DELIBERATELY ABSENT: B1 is an unregistered corpus key,
#    so the flag would REFUSE. Expect the loud NON-PARITY line — that is
#    correct behaviour, not a warning to silence.

PYTHONPATH="$STACK" nohup python3 -u "$STACK/scripts/refc_v3_train.py" "${ARGS[@]}" \
  >> "$OUT/train.log" 2>> "$OUT/train.stderr.log" 200>&- &
echo "launched pid $! -> $OUT"

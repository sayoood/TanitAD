#!/usr/bin/env bash
# A8 (PREREG_REFCV6_DEVBOX_PREPARATION §3.1): A3's EXACT configuration run to 5,000 steps, held-out
# halfB read (500 batches = 1,000 windows) every 1,000 steps -> a CURVE with its step index. One config,
# no replicate; never to be turned into "5,000 beats 2,000". Run BEFORE A7 because A7 needs its BN
# amendment first (a priority order, not a dependency chain). Otherwise a byte-for-byte copy of A3.
# A3 (prerequisite) — a map head trained on halfA ONLY, so halfB is genuinely HELD OUT.
#
# ⛔⛔ WHY THIS RUN EXISTS AT ALL — A DEVIATION FROM THE PRE-REGISTRATION, STATED OPENLY.
# The prereg's A3 says: "score the EXISTING 2,000-step map-head checkpoint on 1,000 HELD-OUT
# windows of the other half", ≈1.8 h, EVAL ONLY. MEASURED 2026-09-19: that checkpoint was
# trained on `v2ep-eval139-416x1024cyl` — the FULL 139-clip cache — and 62/62 of BOTH clean
# halves are inside it, with ZERO clips outside. ⇒ there is NO held-out half for it. Scoring
# it on either half would be a TRAIN-side read wearing a "held-out" label, which is exactly
# the error `PREREG_S1` §8's blocking dependency exists to avoid.
# ⇒ A3 becomes: TRAIN a map head on halfA (2,000 steps), then score halfB. Cost at the
# MEASURED 4.55 s/step: 2.53 h + ~1.8 h eval ≈ 4.3 h, against the prereg's 1.8 h.
#
# ⚠️ AND IT IS A DIFFERENT ARM FROM THE BANKED ONE, DECLARED HERE: the levers
# (`--trunk-chunk-ckpt 1 --trunk-frozen-bn`) are ON, because without them this configuration
# allocates 14.6 GB on an 8 GiB card and PAGES — 29 s/step instead of 4.55, i.e. 16 h instead
# of 2.5. Frozen BN CHANGES the arm, so this checkpoint is NOT bit-comparable with the banked
# 2,000-step maphead1k. A3 is an INSTRUMENT READING of ONE configuration, so that is
# admissible — provided the configuration is stated, which it is here and in the stamp.
#
# ⛔ NOT A CAPABILITY CLAIM: 2,000 steps x batch 2 = 4,000 windows of a 5,302-step epoch
# over 62 clips = 0.38 of ONE epoch.
set -u
WT=/c/Users/Admin/tanitad-wt-bevtac
OUT=/c/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=/d/Projects/TanitAD-artifacts
LAB=/c/Users/Admin/tanitad-wt/_s2build/release/v8/s2_labels_v8_eval.jsonl.gz
A="$ART/v2ep-eval124clean-416x1024cyl-halfA"
B="$ART/v2ep-eval124clean-416x1024cyl-halfB"
MAPS="$ART/sam3-maps-eval"
JOIN="$ART/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
EXTR="$ART/refcv5v2_final/extrinsics141.json"
for p in "$A" "$B" "$MAPS" "$JOIN" "$EXTR" "$LAB"; do
  [ -e "$p" ] || { echo "ZZA3-ABORT-MISSING ${p}ZZ"; exit 2; }
done

WTW='C:\Users\Admin\tanitad-wt-bevtac'
export PYTHONPATH="$WTW\\stack;$WTW;$WTW\\taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
mkdir -p "$OUT"

read -r GU HF < <("$PY" /c/Users/Admin/qland/boxstat.py) || {
  echo "ZZA3-ABORT-PROBE-FAILED -- INCONCLUSIVE, never clearZZ"; exit 3; }
case "$GU$HF" in *[!0-9]*) echo "ZZA3-ABORT-PROBE-SHAPE '$GU $HF'ZZ"; exit 3 ;; esac
[ "$GU" -le 2500 ] || { echo "ZZA3-ABORT-GPU-BUSY used=${GU}MiBZZ"; exit 3; }
[ "$HF" -ge 8 ]    || { echo "ZZA3-ABORT-HOST-TIGHT free=${HF}GBZZ"; exit 3; }
echo "== A3 train: halfA ONLY (halfB held out)  gpu=${GU}MiB host=${HF}GB  $(date -u +%H:%M:%SZ)"

T0=$(date +%s)
( cd "$WT/stack" && "$PY" -u scripts/refc_v3_train.py \
    --arm hier --size tiny --out "$OUT/run" \
    --device cuda --seed 0 --steps 5000 --batch 2 --workers 0 \
    --log-every 10 --save-every 250 \
    --v2-cache "$A" --image-hw 416 1024 --v2-lru 4 \
    --trunk timm --trunk-name resnet34.a1_in1k --trunk-in-channels 9 \
    --trunk-pretrained \
    --trunk-chunk-ckpt 1 --trunk-frozen-bn \
    --v7-labels "$LAB" \
    --agent-join "$JOIN" --agent-join-verify off --agents head --w-agent 1.0 \
    --agent-queries 16 --agent-pad 32 \
    --agent-rig-camera extrinsics --agent-rig-extrinsics "$EXTR" \
    --map-gt-root "$MAPS" --map-lru 4 --map-min-coverage 0.90 \
    --w-map 1.0 --join3d "$JOIN" --w-box3d 1.0 \
    --tac-decoder-v6 --w-tac-v6 1.0 --tac-decoder-d-bev 96 \
    --conflict-detector off --eval-cache "$B" --eval-labels "$LAB" --eval-every 1000 --eval-batches 500 ) > "$OUT/run.log" 2>&1
RC=$?; T1=$(date +%s)
echo "ZZA3-TRAIN-RC ${RC} wall=$((T1-T0))s = $(( (T1-T0)/60 )) minZZ"
ls -l "$OUT/run/ckpt.pt" 2>/dev/null | awk '{print "  ckpt bytes:", $5}'

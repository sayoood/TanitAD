#!/usr/bin/env bash
# A9 — conflict-detector COST at 416x1024, re-measured rather than scaled.
#
# PREREG_REFCV6_DEVBOX_PREPARATION §3.1/§7: 200 steps `--conflict-detector on` vs 200 steps
# `off`, otherwise IDENTICAL (A3's configuration: resnet34, halfA, levers on, batch 2).
# SUPPORTS if the +1/-1 controls read EXACTLY 1.0 / -1.0 and the detached case NaN, with the
# overhead reported as a % with its n. The banked +71.7...+104.2 % was measured on the SMALLER
# grid and the ruling says "re-measure rather than scale".
#
# ⛔ NOT A CAPABILITY CLAIM: 200 steps x batch 2 = 400 windows = 0.075 of one halfA epoch.
# ⛔ The rate comes from `elapsed_s` MARGINAL deltas, warm-up excluded — never cumulative/n.
set -u
WT=/c/Users/Admin/tanitad-wt-bevtac
OUT=/c/Users/Admin/tanitad-caches/a9-conflict-20260919
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=/d/Projects/TanitAD-artifacts
LAB=/c/Users/Admin/tanitad-wt/_s2build/release/v8/s2_labels_v8_eval.jsonl.gz
A="$ART/v2ep-eval124clean-416x1024cyl-halfA"
MAPS="$ART/sam3-maps-eval"
JOIN="$ART/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
EXTR="$ART/refcv5v2_final/extrinsics141.json"
for p in "$A" "$MAPS" "$JOIN" "$EXTR" "$LAB"; do
  [ -e "$p" ] || { echo "ZZA9-ABORT-MISSING ${p}ZZ"; exit 2; }
done

WTW='C:\Users\Admin\tanitad-wt-bevtac'
export PYTHONPATH="$WTW\\stack;$WTW;$WTW\\taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
mkdir -p "$OUT"

run_arm () {
  local mode="$1"
  read -r GU HF < <("$PY" /c/Users/Admin/qland/boxstat.py) || {
    echo "ZZA9-ABORT-PROBE-FAILED -- INCONCLUSIVE, never clearZZ"; exit 3; }
  case "$GU$HF" in *[!0-9]*) echo "ZZA9-ABORT-PROBE-SHAPE '$GU $HF'ZZ"; exit 3 ;; esac
  [ "$GU" -le 2500 ] || { echo "ZZA9-ABORT-GPU-BUSY used=${GU}MiBZZ"; exit 3; }
  [ "$HF" -ge 8 ]    || { echo "ZZA9-ABORT-HOST-TIGHT free=${HF}GBZZ"; exit 3; }
  echo "== A9 conflict-detector=${mode}  gpu=${GU}MiB host=${HF}GB  $(date -u +%H:%M:%SZ)"
  local T0 T1 RC
  T0=$(date +%s)
  ( cd "$WT/stack" && "$PY" -u scripts/refc_v3_train.py \
      --arm hier --size tiny --out "$OUT/$mode" \
      --device cuda --seed 0 --steps 200 --batch 2 --workers 0 \
      --log-every 10 --save-every 100000 \
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
      --conflict-detector "$mode" ) > "$OUT/$mode.log" 2>&1
  RC=$?; T1=$(date +%s)
  echo "ZZA9-${mode}-RC ${RC} wall=$((T1-T0))sZZ"
}

run_arm off
run_arm on

# ⛔ ASSERT ON THE ARTIFACTS, NEVER THE EXIT CODES.
"$PY" /c/Users/Admin/qland/a9_read.py "$OUT"

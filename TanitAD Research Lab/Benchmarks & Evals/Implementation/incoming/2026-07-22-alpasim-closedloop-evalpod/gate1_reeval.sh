#!/bin/bash
# gate1_reeval.sh -- PAIRED closed-loop re-eval of REF-C baseline vs Gate-1 fine-tune
# over the SAME 15 junction scenes, SAME renderer/session (controls run-to-run
# variance), SAME 480x854 as the baseline collection. Reuses the validated M2
# topology via gate1_run.sh. Reads AlpaSim's own offroad/collision metrics.
#
# Usage: bash gate1_reeval.sh <tag> <ckpt> [preset]
#   creates /workspace/gate1_reeval_<tag>, copies the junction config from
#   gate1_junc (NOT the rollouts), runs the pipeline, leaves parquets in place.
set -uo pipefail
TAG="$1"; CKPT="$2"; PRESET="${3:-base}"
SRC=/workspace/gate1_junc
DST=/workspace/gate1_reeval_${TAG}
GLOB_SS=$(grep -m1 data_dir "$SRC/generated-user-config-0.yaml" | awk '{print $NF}' | tr -d '"')
GLOB="$GLOB_SS/**/*.usdz"
ML=/workspace/gate1_reeval_${TAG}.log
mlog(){ echo "[$(date -u +%H:%M:%S)] REEVAL[$TAG] $*" | tee -a "$ML"; }

mlog "=== START tag=$TAG ckpt=$CKPT preset=$PRESET ==="
rm -rf "$DST"; mkdir -p "$DST"
# copy config only (NOT rollouts/aggregate/controller/txt-logs/preds)
for f in generated-user-config-0.yaml generated-network-config.yaml \
         controller-config.yaml driver-config.yaml eval-config.yaml \
         trafficsim-config.yaml run_metadata.yaml wizard-config.yaml \
         wizard-config-loadable.yaml; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/$f"
done
mlog "config copied: $(ls "$DST" | tr '\n' ' ')"

# renderer up on :6011 (shared; leave running across tags)
if ! ss -ltn | grep -q ':6011 '; then
  mlog "launch renderer"
  setsid bash /workspace/renderer_serve.sh 6011 "$GLOB" </dev/null >/workspace/reeval_renderer.log 2>&1 &
  for i in $(seq 1 120); do ss -ltn | grep -q ':6011 ' && grep -q 'Serving on 0.0.0.0:6011' /workspace/reeval_renderer.log && break; sleep 5; done
fi
ss -ltn | grep -q ':6011 ' && mlog "renderer up" || { mlog "renderer FAIL"; exit 4; }

# run the closed-loop pipeline (gate1_run.sh: controller+driver+physics+simulate)
bash /workspace/gate1_run.sh "$DST" /workspace/refc_driver.py "$CKPT" "$PRESET"
mlog "run DONE: $(cat "$DST/DONE" 2>/dev/null)"
NROLL=$(ls -1 "$DST"/rollouts/clipgt-*/*/metrics.parquet 2>/dev/null | wc -l)
mlog "rollouts_with_parquet=$NROLL"
mlog "=== DONE tag=$TAG ==="

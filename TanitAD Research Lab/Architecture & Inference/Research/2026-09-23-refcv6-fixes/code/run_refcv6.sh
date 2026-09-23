#!/usr/bin/env bash
# refcv6 (SPEC_REFCV6_V2) on Thor -- THE launch line, parameterised ONLY by what the smoke measures.
#
#   CODE=<shipped tree> OUT=<run dir> BATCH=<B> STEPS=<S> CONFLICT_EVERY=<N> [WORKERS=..]
#   [EVAL=1 AGENT_JOIN=<train+eval 2-D join> JOIN3D=<train+eval 3-D join>] ./run_refcv6.sh
#   [TRUNK_BF16=1] [TRUNK_CL=1]  the backbone in bf16 / NHWC (speed; outputs back to fp32)
#   [CUDNN_BENCH=1]  cuDNN autotuning   [TRUNK_FROZEN_BN=1]  frozen BN without chunking
#   [TRUNK_FOLD_BN=1]  fold each FROZEN BN into its conv (same function, no BN pass);
#   needs frozen BN, i.e. TRUNK_CHUNK>0 or TRUNK_FROZEN_BN=1 -- the trainer refuses otherwise
#   [TRUNK_CHUNK=<N>]  gradient-checkpoint the backbone in slices of N images; N > 0 also
#   pins BatchNorm to its ImageNet statistics (--trunk-frozen-bn, which the chunking
#   REQUIRES to stay exact). MEASURED 2026-09-23 on Thor: unchunked resnet101 at 416x1024
#   needs ~22 GB per SAMPLE (8 window steps x 3 frames through the trunk), so batch 4 OOMs.
#
# Every flag below is pinned by a review finding, a PI ruling or SPEC v2 -- see
# `TanitAD Research Lab/Architecture & Inference/Research/2026-09-23-refcv6-fixes/LAUNCH_READINESS_FIXES.md`.
# ⛔ `exec`: the supervisor records THIS pid as the trainer's, and kills by explicit pid.
# ⛔ EVAL=1 needs joins that ALSO cover the eval split: the trainer builds its eval join reader
#    from the same --agent-join, and `enable_agent_join` REFUSES a join that covers ZERO of the
#    split's episodes. The train-only joins below therefore run with EVAL=0.
set -u
: "${CODE:?}" "${OUT:?}" "${BATCH:?}" "${STEPS:?}" "${CONFLICT_EVERY:?}"
D=/home/nvidia/data
EVAL="${EVAL:-0}"; EVAL_EVERY="${EVAL_EVERY:-500}"; EVAL_BATCHES="${EVAL_BATCHES:-8}"
SAVE_EVERY="${SAVE_EVERY:-500}"; LOG_EVERY="${LOG_EVERY:-50}"; WORKERS="${WORKERS:-0}"
AGENT_JOIN="${AGENT_JOIN:-/home/nvidia/percprobe/raw/b1train_agents.jsonl.xz}"
JOIN3D="${JOIN3D:-$D/join3d/b1train_agents_3d.jsonl.xz}"
TRUNK_CHUNK="${TRUNK_CHUNK:-0}"
CKPT_ARGS=()
if [ "$TRUNK_CHUNK" -gt 0 ]; then CKPT_ARGS=(--trunk-chunk-ckpt "$TRUNK_CHUNK" --trunk-frozen-bn)
elif [ "${TRUNK_FROZEN_BN:-0}" = "1" ]; then CKPT_ARGS=(--trunk-frozen-bn); fi
SPEED_ARGS=()   # the backbone in bf16 and/or NHWC -- see --trunk-bf16 / --trunk-channels-last
if [ "${TRUNK_BF16:-0}" = "1" ]; then SPEED_ARGS+=(--trunk-bf16); fi
if [ "${TRUNK_CL:-0}" = "1" ]; then SPEED_ARGS+=(--trunk-channels-last); fi
if [ "${CUDNN_BENCH:-0}" = "1" ]; then SPEED_ARGS+=(--cudnn-benchmark); fi
if [ "${TRUNK_FOLD_BN:-0}" = "1" ]; then SPEED_ARGS+=(--trunk-fold-bn); fi
EVAL_ARGS=()
if [ "$EVAL" = "1" ]; then
  EVAL_ARGS=(--eval-cache "$D/refcv6-b1-416x1024-eval139"
             --eval-labels "$D/v8labels/labels/s2_labels_v8_eval.jsonl.gz"
             --eval-every "$EVAL_EVERY" --eval-batches "$EVAL_BATCHES"
             --speed-max-sidecar-v6-eval "$D/refcv6_speed_max_v8_eval.jsonl")
fi
export PYTHONPATH="$CODE/stack:$CODE/taniteval" PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec /home/nvidia/venvs/tanitad-train/bin/python "$CODE/stack/scripts/refc_v3_train.py" \
  --arm hier --size small --trunk timm --trunk-name resnet101.a1_in1k \
  --trunk-mode shared --trunk-fuse concat1x1 --ego-history --no-strategic --image-hw 416 1024 \
  --sampler ddim --f1-random-t --f2-dd-step --f3-per-layer --f4-adaln --f5-focal \
  --f5-emitting-conf --f6-w-u0-zero --f9-assert-vocab \
  --anchors "$D/anchors/refc_anchors_6s_v0cond_alat_117.pt" --anchor-v0-conditioned --n-anchors 117 \
  --v2-cache "$D/refcv6-b1-416x1024-train" \
  --v7-labels "$D/v8labels/labels/s2_labels_v8_train.jsonl.gz" --nav-from-v7 \
  ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
  --tac-decoder-v6 --w-tac-v6 1.0 --tac-decoder-d-bev 96 --graft-behaviour-sel \
  --max-speed-input-v6 --speed-max-sidecar-v6 "$D/refcv6_speed_max_v8_train.jsonl" \
  --agents head --agent-join "$AGENT_JOIN" \
  --agent-rig-camera extrinsics --agent-rig-extrinsics "$D/refcv6_train_eval139_extrinsics.json" \
  --agent-cls-weight b1 --w-agent 1.0 \
  --w-map 1.0 --map-gt-root "$D/sam3_corpus" \
  --w-box3d 1.0 --join3d "$JOIN3D" --bev-coupling \
  --equalize-bottom-rows 43 --opt dd --lr 1e-4 --warmup 2000 --seed 0 --u8-batches \
  ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} \
  ${SPEED_ARGS[@]+"${SPEED_ARGS[@]}"} \
  --workers "$WORKERS" --conflict-every "$CONFLICT_EVERY" \
  --save-every "$SAVE_EVERY" --log-every "$LOG_EVERY" --batch "$BATCH" --steps "$STEPS" \
  --out "$OUT"

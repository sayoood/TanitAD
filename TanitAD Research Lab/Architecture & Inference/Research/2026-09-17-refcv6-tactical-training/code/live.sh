#!/usr/bin/env bash
# The LIVE refcv6 §4/§5 tactical run, on the REAL artifacts, CPU-only.
#
# ⛔ EVERY PATH IS AN ENV VAR WITH NO BAKED-IN DEFAULT FOR THE SCRATCH ROOT.
# `tests/test_refcv6_no_session_paths.py` forbids a session scratchpad path in
# a repo artifact, for the stated reason that nothing can tell a session UUID
# from a clip UUID by looking at it. So the caller supplies it:
#
#   TAC_OUT=<some scratch dir> TAC_WT=<worktree> bash live.sh <tag> [extra args]
set -u
WT="${TAC_WT:?set TAC_WT to the worktree root}"
OUT="${TAC_OUT:?set TAC_OUT to a scratch output dir}"
PY="${TAC_PY:?set TAC_PY to the venv python}"
ART="${TAC_ART:?set TAC_ART to the artifacts root}"
LAB="${TAC_LABELS:?set TAC_LABELS to s2_labels_v8.0_eval.jsonl.gz}"
SIDE="${TAC_SIDECAR:?set TAC_SIDECAR to the speed-max sidecar}"

CACHE="$ART/v2ep-eval139-256x1024cyl"
JOIN="$ART/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
EXTR="$ART/refcv5v2_final/extrinsics141.json"

export PYTHONPATH="$WT/stack;$WT;$WT/taniteval"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=""
# ⛔ torch spawns ~113 threads PER PROCESS and concurrent arms then make NO
# progress -- it looks exactly like a hang (7 arms at sm 0-6 % for 50 min).
export OMP_NUM_THREADS=6

TAG="${1:-live}"
shift || true
mkdir -p "$OUT"
( cd "$WT/stack" && "$PY" -u scripts/refc_v3_train.py \
    --arm hier --size tiny --out "$OUT/$TAG" \
    --device cpu --seed 0 --steps 3 --batch 2 --workers 0 \
    --log-every 1 --save-every 3 \
    --v2-cache "$CACHE" --image-hw 256 1024 --v2-lru 4 \
    --trunk timm --trunk-name resnet34.a1_in1k --trunk-in-channels 9 \
    --v7-labels "$LAB" \
    --agent-join "$JOIN" --agent-join-verify off --agents head --w-agent 1.0 \
    --agent-queries 16 --agent-pad 32 \
    --agent-rig-camera extrinsics --agent-rig-extrinsics "$EXTR" \
    --tac-decoder-v6 --w-tac-v6 1.0 \
    --max-speed-input-v6 --speed-max-sidecar-v6 "$SIDE" \
    --graft-behaviour-sel \
    --grad-probe-modules tac_decoder_v6,tac_decoder_v6.validity_head,tac_decoder_v6.conf_head,tac_decoder_v6.lat_head,tac_decoder_v6.lon_head,tac_decoder_v6.queries,tac_decoder_v6.layers,tac_decoder_v6.agent_in \
    "$@" ) > "$OUT/$TAG.log" 2>&1
# ⛔ NEVER read `$?` through a pipe -- it reports the LAST element's status.
echo "$TAG exit=$?"

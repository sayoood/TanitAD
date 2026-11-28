#!/usr/bin/env bash
# The LIVE perception run, on the REAL artifacts, CPU-only.
#   256x1024 cache (139 clips) · SAM3 maps (135) · 3-D agent join (905,512
#   agent-frames) · per-clip extrinsics (141 clips).
set -u
WT="C:/Users/Admin/tanitad-wt-perctrain"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/<session-id>/scratchpad"
CACHE="D:/Projects/TanitAD-artifacts/v2ep-eval139-256x1024cyl"
MAPS="D:/Projects/TanitAD-artifacts/sam3-maps-eval"
JOIN="D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
EXTR="D:/Projects/TanitAD-artifacts/refcv5v2_final/extrinsics141.json"

export PYTHONPATH="$WT/stack;$WT;$WT/taniteval"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=6

TAG="${1:-live}"
shift || true
mkdir -p "$SP/live"
( cd "$WT/stack" && "$PY" -u scripts/refc_v3_train.py \
    --arm hier --size tiny --out "$SP/live/$TAG" \
    --device cpu --seed 0 --steps 3 --batch 2 --workers 0 \
    --log-every 1 --save-every 3 \
    --v2-cache "$CACHE" --image-hw 256 1024 --v2-lru 4 \
    --trunk timm --trunk-name resnet34.a1_in1k --trunk-in-channels 9 \
    --agent-join "$JOIN" --agent-join-verify off --agents head --w-agent 1.0 \
    --agent-queries 16 --agent-pad 32 \
    --agent-rig-camera extrinsics --agent-rig-extrinsics "$EXTR" \
    --map-gt-root "$MAPS" --map-lru 2 --map-min-coverage 0.90 \
    --join3d "$JOIN" \
    "$@" ) > "$SP/live/$TAG.log" 2>&1
echo "$TAG exit=$?"

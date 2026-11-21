#!/usr/bin/env bash
# THE LIVE BEV->TACTICAL ARM, on the REAL artifacts, CPU-only.
#
#   256x1024 v2 cache (139 clips) · REAL SAM3 map GT (135 npz) · 3-D agent join
#   · per-clip extrinsics (141 clips) · v8.0 tactical labels
#
# ⭐ WHY THIS EXISTS AND THE SYNTHETIC RIG DOES NOT REPLACE IT. `grad_reach_bev.py`
# proves the gradient reaches the trunk on a hand-built geometry; it does NOT
# prove the branch survives a real SAM3 label grid, a real lift geometry, or the
# trainer's own loss assembly. This does, end to end, on the path an operator
# would actually launch.
#
# ⛔ EVERY PATH IS AN ENV VAR WITH NO BAKED-IN DEFAULT.
# `tests/test_refcv6_no_session_paths.py` forbids a session scratchpad path and
# an absolute home path in a repo artifact — nothing can tell a session UUID
# from a clip UUID by looking at it.
#
#   TAC_WT=<worktree> TAC_OUT=<scratch> TAC_PY=<venv python> \
#   TAC_ART=<artifacts root> TAC_LABELS=<s2_labels_v8.0_eval.jsonl.gz> \
#   bash live.sh <tag> [extra args]
set -u
WT="${TAC_WT:?set TAC_WT to the worktree root}"
OUT="${TAC_OUT:?set TAC_OUT to a scratch output dir}"
PY="${TAC_PY:?set TAC_PY to the venv python}"
ART="${TAC_ART:?set TAC_ART to the artifacts root}"
LAB="${TAC_LABELS:?set TAC_LABELS to s2_labels_v8.0_eval.jsonl.gz}"

CACHE="$ART/v2ep-eval139-256x1024cyl"
MAPS="$ART/sam3-maps-eval"
JOIN="$ART/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
EXTR="$ART/refcv5v2_final/extrinsics141.json"

export PYTHONPATH="$WT/stack;$WT;$WT/taniteval"
export PYTHONIOENCODING=utf-8
# ⛔ The GPU is held by an RL arm. ⚠️ Under this setting torch reports
# `is_available()` True with `device_count()` 0, so anything touching cudnn
# (a GRU's `flatten_parameters()`) dies with a bare
# `ValueError: min() iterable argument is empty` — environmental, not a defect.
export CUDA_VISIBLE_DEVICES=""
# ⛔ torch spawns ~113 threads PER PROCESS and concurrent arms then make NO
# progress -- it looks exactly like a hang (7 arms at sm 0-6 % for 50 min).
export OMP_NUM_THREADS=6

TAG="${1:-bevtac}"
shift || true
mkdir -p "$OUT"
( cd "$WT/stack" && "$PY" -u scripts/refc_v3_train.py \
    --arm hier --size tiny --out "$OUT/$TAG" \
    --device cpu --seed 0 --steps 2 --batch 2 --workers 0 \
    --log-every 1 --save-every 2 \
    --v2-cache "$CACHE" --image-hw 256 1024 --v2-lru 4 \
    --trunk timm --trunk-name resnet34.a1_in1k --trunk-in-channels 9 \
    --no-trunk-pretrained \
    --v7-labels "$LAB" \
    --agent-join "$JOIN" --agent-join-verify off --agents head --w-agent 1.0 \
    --agent-queries 16 --agent-pad 32 \
    --agent-rig-camera extrinsics --agent-rig-extrinsics "$EXTR" \
    --map-gt-root "$MAPS" --map-lru 2 --map-min-coverage 0.90 \
    --w-map 1.0 --join3d "$JOIN" --w-box3d 1.0 \
    --tac-decoder-v6 --w-tac-v6 1.0 \
    --tac-decoder-d-bev 96 \
    --conflict-detector on \
    "$@" ) > "$OUT/$TAG.log" 2>&1
# ⛔ NEVER `$?` THROUGH A PIPE -- it reports the LAST element's status.
echo "$TAG exit=$?"

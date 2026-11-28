#!/usr/bin/env bash
# SPEC_REFCV6_V2 §10.6 — PIPELINE VALIDATION, on the GPU, at the PRIMARY trunk.
#
# §10.6 verbatim: "The 139 B1 eval clips are rebuilt at 256 x 1024 and carry the whole
# chain end to end -- trunk -> lift -> map and box heads -> planner -- BEFORE ANY POD
# HOUR IS SPENT. Their SAM3 maps (135 of 139) are already on the dev box."
#
# WHY THIS AND NOT THE AGENT'S live.sh. That run was CPU-only, resnet34, no pretrained
# init, 2 steps -- it proved the chain ASSEMBLES. This runs the arm that will actually
# train: section 10.2's PRIMARY trunk (resnet101), ImageNet init, on the GPU, for
# enough steps that the losses have to move. A chain that assembles on CPU at batch 2
# is not a chain that trains.
#
# Section 8's two prerequisites are DONE: the D3 proof package (48 files landed) and
# the D9 RL lever L1 (Stage A closed, H-DDV2RL-2 = FAIL-HARM). This is the remaining
# dev-box gate before a pod hour is justified.
#
# 256x1024 and not 408x1024: 408 is REFUSED by timm_trunk.py:216 (408 % 32 = 24) --
# PI_DECISION_QUEUE item 20. 256x1024 is section 10.1's standing amendment, is legal
# at both strides, and its cache is already built.
set -u
WT=/c/Users/Admin/tanitad-wt-bevtac
OUT=/c/Users/Admin/tanitad-caches/refcv6-pipeval-20260917
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=/d/Projects/TanitAD-artifacts
LAB=/c/Users/Admin/tanitad-wt/_s2build/release/v8/s2_labels_v8_eval.jsonl.gz

CACHE="$ART/v2ep-eval139-256x1024cyl"
MAPS="$ART/sam3-maps-eval"
JOIN="$ART/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
EXTR="$ART/refcv5v2_final/extrinsics141.json"

for p in "$CACHE" "$MAPS" "$JOIN" "$EXTR" "$LAB"; do
  [ -e "$p" ] || { echo "ZZABORT-MISSING-INPUT ${p}ZZ"; exit 2; }
done

# WINDOWS-FORM PATHS, NOT MSYS. A "/c/Users/..." entry handed to a *Windows* Python is
# resolved against the current drive and lands as "G:\c\Users\..." -- garbage that
# silently does nothing. The import then falls through to the venv's PEP 660 EDITABLE
# FINDER, which points at the G: mount, and `tanitad` resolves to a tree that is
# dehydrated. That is how this run first died with an ImportError on a module that is
# present in the worktree. VERIFIED: with the form below, tanitad.__file__ reads
# C:\Users\Admin\tanitad-wt-bevtac\stack\tanitad\__init__.py -- the PYTHONPATH entry
# wins over the finder hook.
WTW='C:\Users\Admin\tanitad-wt-bevtac'
export PYTHONPATH="$WTW\\stack;$WTW;$WTW\\taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6          # torch spawns ~113 threads per process

# Never add GPU load to a box that is training -- but measure FREE MEMORY, not a name.
# boxstat.py writes BYTES rather than print(): Python's Windows text mode would append
# a carriage return, the second field would arrive as "18" plus CR, and the digit test
# below would reject a perfectly valid reading. Fixed at the producer so no caller
# needs a `tr -d` it might forget.
read -r GU HF < <("$PY" /c/Users/Admin/qland/boxstat.py) || {
  echo "ZZABORT-PROBE-FAILED box state unreadable -- INCONCLUSIVE, never clearZZ"; exit 3; }
case "$GU$HF" in *[!0-9]*) echo "ZZABORT-PROBE-SHAPE '$GU $HF'ZZ"; exit 3 ;; esac
[ "$GU" -le 2500 ] || { echo "ZZABORT-GPU-BUSY used=${GU}MiBZZ"; exit 3; }
[ "$HF" -ge 8 ]    || { echo "ZZABORT-HOST-TIGHT free=${HF}GBZZ"; exit 3; }
echo "   box ok: gpu_used=${GU}MiB host_free=${HF}GB"

TAG="${1:-r101}"; shift || true
mkdir -p "$OUT"
echo "== pipeline validation $TAG  $(date -u +%H:%M:%SZ)"
( cd "$WT/stack" && "$PY" -u scripts/refc_v3_train.py \
    --arm hier --size tiny --out "$OUT/$TAG" \
    --device cuda --seed 0 --steps 40 --batch 2 --workers 0 \
    --log-every 5 --save-every 40 \
    --v2-cache "$CACHE" --image-hw 256 1024 --v2-lru 4 \
    --trunk timm --trunk-name resnet101.a1_in1k --trunk-in-channels 9 \
    --trunk-pretrained \
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
RC=$?
# THE ARTIFACT IS THE EVIDENCE, NEVER THE EXIT CODE.
ROWS=$(wc -l < "$OUT/$TAG/metrics.jsonl" 2>/dev/null || echo 0)
CFG=$(stat -c %s "$OUT/$TAG/config.json" 2>/dev/null || echo 0)
echo "ZZPIPEVAL $TAG rc=$RC rows=${ROWS} config_bytes=${CFG}ZZ"
if [ "${ROWS:-0}" -lt 5 ]; then
  echo "ZZPIPEVAL-INCOMPLETE ${TAG}ZZ"; tail -25 "$OUT/$TAG.log"; exit 5
fi
echo "ZZPIPEVAL-OK ${TAG}ZZ"

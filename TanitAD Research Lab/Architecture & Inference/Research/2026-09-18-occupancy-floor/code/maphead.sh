#!/usr/bin/env bash
# Does the refcv6 MAP HEAD learn at all? -- the instrument S1-GATE-PRED is blocked on.
#
# ⛔ THIS IS AN INSTRUMENT RUN, NOT AN ARM. One configuration, no replicate, no
# comparison. It CANNOT make a capability claim about refcv6 -- that needs the
# pre-registered panel in PREREG_REFCV6.md with its one-variable arms and replicates.
# What it CAN answer is the question `PREREG_S1` §8 names as the first work item:
# is the predicted occupancy good enough to gate on, and does it move at all?
#
# WHAT WOULD COUNT AS MOVING (both metrics land every step, so it is readable live):
#   map_iou_drivable            0.0 -> must clear 0.3412 merely to MATCH A CONSTANT
#                               (the no-information floor, MEASURED over 10,068,274
#                               seen cells, …/2026-09-18-occupancy-floor/)
#   map_pred_drivable_prob_mean 0.078 -> toward the GT's ~0.35; threshold-free, so it
#                               shows progress the IoU cannot while under-confident
#
# MEASURED 2026-09-18: 28.30 s/step at 416x1024, resnet34, batch 2 on this box
# (30 steps, wallclock 850.2 s). 1,000 steps therefore costs ~7.9 h.
# ⛔ resnet34 and not resnet101: §10.2's PRIMARY trunk OOMs here at batch 1.
# Every head stays ON -- the question is whether the map head learns INSIDE the refcv6
# chain, not whether one can learn in isolation, and those are different questions.
set -u
WT=/c/Users/Admin/tanitad-wt-bevtac
OUT=/c/Users/Admin/tanitad-caches/refcv6-maphead-20260918
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=/d/Projects/TanitAD-artifacts
LAB=/c/Users/Admin/tanitad-wt/_s2build/release/v8/s2_labels_v8_eval.jsonl.gz
CACHE="$ART/v2ep-eval139-416x1024cyl"
MAPS="$ART/sam3-maps-eval"
JOIN="$ART/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
EXTR="$ART/refcv5v2_final/extrinsics141.json"

for p in "$CACHE" "$MAPS" "$JOIN" "$EXTR" "$LAB"; do
  [ -e "$p" ] || { echo "ZZABORT-MISSING-INPUT ${p}ZZ"; exit 2; }
done

WTW='C:\Users\Admin\tanitad-wt-bevtac'
export PYTHONPATH="$WTW\\stack;$WTW;$WTW\\taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6

read -r GU HF < <("$PY" /c/Users/Admin/qland/boxstat.py) || {
  echo "ZZABORT-PROBE-FAILED -- INCONCLUSIVE, never clearZZ"; exit 3; }
case "$GU$HF" in *[!0-9]*) echo "ZZABORT-PROBE-SHAPE '$GU $HF'ZZ"; exit 3 ;; esac
[ "$GU" -le 2500 ] || { echo "ZZABORT-GPU-BUSY used=${GU}MiBZZ"; exit 3; }
[ "$HF" -ge 8 ]    || { echo "ZZABORT-HOST-TIGHT free=${HF}GBZZ"; exit 3; }
echo "   box ok: gpu_used=${GU}MiB host_free=${HF}GB"

TAG=maphead1k
mkdir -p "$OUT"
echo "== map-head instrument run, 1000 steps  $(date -u +%H:%M:%SZ)  (~7.9 h at 28.30 s/step)"
( cd "$WT/stack" && "$PY" -u scripts/refc_v3_train.py \
    --arm hier --size tiny --out "$OUT/$TAG" \
    --device cuda --seed 0 --steps 1000 --batch 2 --workers 0 \
    --log-every 10 --save-every 250 \
    --v2-cache "$CACHE" --image-hw 416 1024 --v2-lru 4 \
    --trunk timm --trunk-name resnet34.a1_in1k --trunk-in-channels 9 --trunk-pretrained \
    --v7-labels "$LAB" \
    --agent-join "$JOIN" --agent-join-verify off --agents head --w-agent 1.0 \
    --agent-queries 16 --agent-pad 32 \
    --agent-rig-camera extrinsics --agent-rig-extrinsics "$EXTR" \
    --map-gt-root "$MAPS" --map-lru 4 --map-min-coverage 0.90 \
    --w-map 1.0 --join3d "$JOIN" --w-box3d 1.0 \
    --tac-decoder-v6 --w-tac-v6 1.0 --tac-decoder-d-bev 96 \
    --conflict-detector off ) > "$OUT/$TAG.log" 2>&1
RC=$?
# THE ARTIFACT IS THE EVIDENCE, NEVER THE EXIT CODE.
ROWS=$(wc -l < "$OUT/$TAG/metrics.jsonl" 2>/dev/null || echo 0)
echo "ZZMAPHEAD rc=$RC rows=${ROWS}ZZ"
[ "${ROWS:-0}" -ge 50 ] || { echo "ZZMAPHEAD-INCOMPLETEZZ"; tail -25 "$OUT/$TAG.log"; exit 5; }
"$PY" -c "
import json
rows=[json.loads(l) for l in open(r'C:/Users/Admin/tanitad-caches/refcv6-maphead-20260918/$TAG/metrics.jsonl',encoding='utf-8')]
f,l=rows[0],rows[-1]
print(f'  map        {f.get(\"map\"):.4f} -> {l.get(\"map\"):.4f}')
print(f'  map_iou    {f.get(\"map_iou_drivable\"):.4f} -> {l.get(\"map_iou_drivable\"):.4f}   (floor 0.3412)')
print(f'  prob_mean  {f.get(\"map_pred_drivable_prob_mean\"):.4f} -> {l.get(\"map_pred_drivable_prob_mean\"):.4f}   (GT ~0.35)')
print('  ZZMAPHEAD-MOVED' + ('YES' if (l.get('map_iou_drivable') or 0) > 0 else 'NO') + 'ZZ')
"

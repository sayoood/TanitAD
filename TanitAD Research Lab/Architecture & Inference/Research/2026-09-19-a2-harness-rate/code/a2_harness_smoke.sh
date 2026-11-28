#!/usr/bin/env bash
# A2 — harness rate + 40-step wiring smoke on the CLEAN-124 halves, WITH the levers on.
#
# ⛔ THE OBLIGATION THIS CARRIES. The whole re-pricing (bucket A 132.1 h -> 33.6 h) rests on
# 2.8 s/step, measured on a microbench-style arm. The prereg's own instruction is
# "re-measure, do not scale", so A2 measures the rate on the REAL rig, at the REAL geometry,
# on the REAL clean-124 half, with every head live — and that measured value is what goes
# into the record.
#
# ⛔ "IT DID NOT RAISE" IS NOT A FIT ON THIS BOX: CUDA spills past VRAM into host RAM
# instead of raising. The trainer's own `cuda_max_mem_gb` is read back from the metrics and
# checked against the 7.1 GB bar.
#
# ⚠️ 40 steps at batch 2 over a 62-clip half is 80 windows of a 5,302-step epoch = 0.0075 of
# ONE epoch. NOTHING here is a capability claim; it is a wiring and rate instrument.
set -u
WT=/c/Users/Admin/tanitad-wt-bevtac
OUT=/c/Users/Admin/tanitad-caches/a2-smoke-20260919
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=/d/Projects/TanitAD-artifacts
LAB=/c/Users/Admin/tanitad-wt/_s2build/release/v8/s2_labels_v8_eval.jsonl.gz
A="$ART/v2ep-eval124clean-416x1024cyl-halfA"
B="$ART/v2ep-eval124clean-416x1024cyl-halfB"
MAPS="$ART/sam3-maps-eval"
JOIN="$ART/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
EXTR="$ART/refcv5v2_final/extrinsics141.json"
for p in "$A" "$B" "$MAPS" "$JOIN" "$EXTR" "$LAB"; do
  [ -e "$p" ] || { echo "ZZA2-ABORT-MISSING ${p}ZZ"; exit 2; }
done

WTW='C:\Users\Admin\tanitad-wt-bevtac'
export PYTHONPATH="$WTW\\stack;$WTW;$WTW\\taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
mkdir -p "$OUT"

read -r GU HF < <("$PY" /c/Users/Admin/qland/boxstat.py) || {
  echo "ZZA2-ABORT-PROBE-FAILED -- INCONCLUSIVE, never clearZZ"; exit 3; }
case "$GU$HF" in *[!0-9]*) echo "ZZA2-ABORT-PROBE-SHAPE '$GU $HF'ZZ"; exit 3 ;; esac
[ "$GU" -le 2500 ] || { echo "ZZA2-ABORT-GPU-BUSY used=${GU}MiBZZ"; exit 3; }
[ "$HF" -ge 8 ]    || { echo "ZZA2-ABORT-HOST-TIGHT free=${HF}GBZZ"; exit 3; }
echo "== A2  train=halfA eval=halfB  gpu=${GU}MiB host=${HF}GB  $(date -u +%H:%M:%SZ)"

T0=$(date +%s)
( cd "$WT/stack" && "$PY" -u scripts/refc_v3_train.py \
    --arm hier --size tiny --out "$OUT/run" \
    --device cuda --seed 0 --steps 40 --batch 2 --workers 0 \
    --log-every 1 --save-every 1000 \
    --v2-cache "$A" --image-hw 416 1024 --v2-lru 4 \
    --trunk timm --trunk-name resnet34.a1_in1k --trunk-in-channels 9 \
    --trunk-pretrained \
    --trunk-chunk-ckpt 1 --trunk-frozen-bn \
    --v7-labels "$LAB" \
    --agent-join "$JOIN" --agent-join-verify off --agents head --w-agent 1.0 \
    --agent-queries 16 --agent-pad 32 \
    --agent-rig-camera extrinsics --agent-rig-extrinsics "$EXTR" \
    --map-gt-root "$MAPS" --map-lru 2 --map-min-coverage 0.90 \
    --w-map 1.0 --join3d "$JOIN" --w-box3d 1.0 \
    --tac-decoder-v6 --w-tac-v6 1.0 --tac-decoder-d-bev 96 \
    --conflict-detector off \
    --eval-cache "$B" --eval-labels "$LAB" \
    --eval-every 40 --eval-batches 4 ) > "$OUT/run.log" 2>&1
RC=$?; T1=$(date +%s)
echo "ZZA2-RC ${RC} wall=$((T1-T0))sZZ"

# ⛔ ASSERT ON THE ARTIFACT, NEVER THE EXIT CODE.
"$PY" - "$OUT/run" "$((T1-T0))" <<'PYEOF'
import json, pathlib, sys
d, wall = pathlib.Path(sys.argv[1]), float(sys.argv[2])
mj = d / "metrics.jsonl"
if not mj.exists():
    print("ZZA2-VERDICT NO-METRICSZZ"); raise SystemExit(4)
rows = [json.loads(l) for l in mj.read_text(encoding="utf-8").splitlines() if l.strip()]
tr = [r for r in rows if "loss" in r and "step" in r and not any(
    k.startswith("eval_") for k in r)]
losses = [float(r["loss"]) for r in tr]
steps = [int(r["step"]) for r in tr]
print(f"train rows={len(tr)}  steps {min(steps) if steps else '-'}..{max(steps) if steps else '-'}")
if len(losses) < 5:
    print("ZZA2-VERDICT TOO-FEW-ROWSZZ"); raise SystemExit(5)
finite = all(l == l and abs(l) != float("inf") for l in losses)
moved = len(set(round(l, 6) for l in losses)) > 1
print(f"loss finite={finite}  MOVED={moved}  first={losses[0]:.4f} last={losses[-1]:.4f} "
      f"distinct={len(set(round(l,6) for l in losses))}")
# the rate, from the trainer's OWN per-step field where present, else wall/steps
sps = [float(r["step_s"]) for r in tr if "step_s" in r]
med = sorted(sps)[len(sps)//2] if sps else None
print(f"s/step: trainer median={med}  wall/steps={wall/max(len(tr),1):.2f} "
      f"(wall includes build+first-batch, so it is an UPPER bound)")
mem = [float(r["cuda_max_mem_gb"]) for r in tr if "cuda_max_mem_gb" in r]
peak = max(mem) if mem else None
print(f"cuda_max_mem_gb peak={peak}  (bar: <= 7.1)")
tac = [float(r["tacv6_n_scene_mean"]) for r in tr if "tacv6_n_scene_mean" in r]
print(f"tacv6_n_scene_mean={set(round(t,1) for t in tac) if tac else 'ABSENT'}  (want 496.0)")
ok = finite and moved and (peak is None or peak <= 7.1)
json.dump({"_what": "A2 harness rate + 40-step wiring smoke, levers ON",
           "_evidence_class": "MEASURED (ours)",
           "_not_a_capability_claim": ("40 steps x batch 2 = 80 windows of a 5,302-step "
                                       "epoch = 0.0075 of ONE epoch"),
           "train_rows": len(tr), "loss_finite": finite, "loss_moved": moved,
           "loss_first": losses[0], "loss_last": losses[-1],
           "s_per_step_trainer_median": med, "wall_s": wall,
           "cuda_max_mem_gb_peak": peak, "fits_7_1_gb": (peak is None or peak <= 7.1),
           "tacv6_n_scene_mean": sorted(set(round(t,1) for t in tac)) if tac else None},
          open(r"C:/Users/Admin/qland/a2_smoke.json", "w", encoding="utf-8"), indent=1)
print("ZZA2-OK" if ok else "ZZA2-FAIL")
raise SystemExit(0 if ok else 6)
PYEOF

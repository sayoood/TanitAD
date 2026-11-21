#!/bin/sh
# CPU smoke of the per-frame battery path (run_battery_r8_pf.py -> roll_seed_r8_pf.py), 2 clips x 3 windows, both seeds, EMA-40k; no GPU, no lock.
# Then the independent clause-(1b) checker on each seed's dump against the CLIP-FED banked 40k EMA dump (subset allowed: smoke only).
HOLDW=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
OUT=${SMK_OUT:-C:/Users/Admin/refcv8_pf/smoke_batt}
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
TAG=${SMK_TAG:-smoke40kema_pf}
CK=${SMK_CK:-E:/refcv8_eval_kit/ckpt/ckpt_ema_40000.pt}
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 \
       CUDA_VISIBLE_DEVICES=-1 R8_NO_POSE_SYNC=1
echo "ZZSMKSTARTZZ $(date +%FT%T%z) tag $TAG" >> "$OUT/smoke.log"
"$PY" -X utf8 "$HOLDW/code/run_battery_r8_pf.py" --ckpt "$CK" --config E:/refcv8_eval_kit/ckpt/config.json --no-g0 --tag "$TAG" --infer-seeds 0,1 \
     --smoke-cpu --max-clips 2 --max-windows-per-clip 3 --out-root "$OUT" > "$OUT/battery_$TAG.log" 2>&1
echo "ZZSMKBATTDONEZZ $(date +%FT%T%z)" >> "$OUT/smoke.log"
for s in 0 1; do
  "$PY" -X utf8 "$HOLDW/code/feed_parity_pf.py" --pf-dump "$OUT/$TAG/dump_s$s" --clip-dump "C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc/dump_s$s" \
       --v9 E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz --v9-md5 6b5c7f207cffc3b7eb3cd527fd433599 \
       --out "$OUT/FEED_PARITY_smoke_s$s.json" --label "smoke battery s$s" --known-values 3 --subset-ok-smoke-only >> "$OUT/parity.log" 2>&1
done
echo "ZZSMKENDZZ $(date +%FT%T%z)" >> "$OUT/smoke.log"

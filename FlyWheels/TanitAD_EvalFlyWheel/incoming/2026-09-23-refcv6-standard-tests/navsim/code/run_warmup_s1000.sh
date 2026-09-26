#!/usr/bin/env bash
# Pipeline validation (SPEC §6) — every arm on warmup_two_stage with the step-1,000 kit checkpoint,
# CPU fp32, exact dedup, then the official scorer through the RAM-gated retrying queue (+ the
# model-free ECHO control). ⚠️ SECOND RUN: the first (2026-09-23 22:09Z) used a decoder rebuilt with
# the WRONG anchor units ('kappa' for an 'alat' vocabulary — KL, raw/controls/) and is quarantined
# in raw/VOID_wrong_anchor_units/.
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
E2RAW="$P/../../2026-09-19-navsim-refcv4b-bridge/raw"
OUT="$P/raw/bridge_warmup_s1000"
SC="$P/raw/scores_warmup_s1000"
mkdir -p "$OUT" "$SC"
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval" CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8
echo "BRIDGE_START $(date -u +%FT%TZ)" >> "$OUT/queue.log"
"$PY" "$P/code/run_bridge6.py" --split warmup_two_stage \
   --arms R6_A1,R6_A1_s1,R6_BLIND,R6_NAVOFF,R6_VMAXOFF,R6_A1NT \
   --inputs "$E2RAW/navsim_agent_inputs.json" --speed "$P/raw/inputs/speed_limits_warmup_two_stage.json" \
   --bank2 C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923/warmup_two_stage \
   --road-plane "$P/raw/inputs/road_plane_navhard_warmup_logs.json" \
   --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step1000.pt --config D:/refcv6_eval_kit/ckpt/config.json \
   --ckpt-md5 7c3ad3c1fbf3d30be5c7733e7b436c65 --device cpu --precision fp32 --exact-dedup \
   --threads 8 --out "$OUT" > "$OUT/bridge.log" 2>&1
echo "BRIDGE_RC=$? $(date -u +%FT%TZ)" >> "$OUT/queue.log"
S=""
for a in R6_A1 R6_A1_s1 R6_BLIND R6_NAVOFF R6_VMAXOFF R6_A1NT; do
  [ -f "$OUT/seam_$a.npz" ] && S="$S $a=$OUT/seam_$a.npz"
done
bash "$P/code/score_queue6.sh" warmup_two_stage "$SC" e6s1000b $S ECHO_ha0_ext="$E2RAW/seam_ECHO_ha0_ext.npz"
echo "ZZWARMUPV2DONEZZ $(date -u +%FT%TZ)" >> "$OUT/queue.log"

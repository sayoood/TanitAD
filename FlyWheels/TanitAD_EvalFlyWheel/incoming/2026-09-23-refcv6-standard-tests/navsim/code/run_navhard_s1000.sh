#!/usr/bin/env bash
# Pipeline validation (SPEC §6) on navhard_two_stage: R6_A1 with the step-1,000 kit checkpoint,
# BOTH stages (450 stage-1 ORIGINAL scenes + 5,462 synthetic), CPU fp32, exact dedup — then the
# official two-stage scorer and this package's navhard CV re-score (harness control KH-nav), both
# through the RAM-gated RETRYING queue (code/score_queue6.sh).
# ⚠️ RAM-GATED (MEASURED 2026-09-24 00:30/00:46: other sessions drove the box to 1.3 / 1.9 GB free
# and E1's RAM guard aborted two scorings): the bridge waits for >= RAM_MIN_GB free on 3
# consecutive 60 s samples. Opaque tokens only (never grep the words this script contains).
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
OUT="$P/raw/bridge_navhard_s1000"
SC="$P/raw/scores_navhard_s1000"
B=C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923
RAM_MIN_GB="${RAM_MIN_GB:-5}"
mkdir -p "$OUT" "$SC"
ram_gate() {   # $1 = label
  local ok=0 f=0
  while [ $ok -lt 3 ]; do
    f=$("$PY" -c "import psutil; print(int(psutil.virtual_memory().available/2**30))")
    if [ "$f" -ge "$RAM_MIN_GB" ]; then ok=$((ok+1)); else ok=0; fi
    [ $ok -lt 3 ] && sleep 60
  done
  echo "RAMGATE_PASS $1 free=${f}GB $(date -u +%FT%TZ)" >> "$OUT/queue.log"
}
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval" CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8
ram_gate bridge
echo "NAVHARD_BRIDGE_START $(date -u +%FT%TZ)" >> "$OUT/queue.log"
"$PY" "$P/code/run_bridge6.py" --split navhard_two_stage --arms R6_A1 \
   --inputs C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json \
   --speed "$P/raw/inputs/speed_limits_navhard_two_stage.json" \
   --bank2 "$B/navhard_s2" --bank1 "$B/navhard_s1" \
   --road-plane "$P/raw/inputs/road_plane_navhard_warmup_logs.json" \
   --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step1000.pt --config D:/refcv6_eval_kit/ckpt/config.json \
   --ckpt-md5 7c3ad3c1fbf3d30be5c7733e7b436c65 --device cpu --precision fp32 --exact-dedup \
   --threads 8 --out "$OUT" > "$OUT/bridge.log" 2>&1
echo "NAVHARD_BRIDGE_RC=$? $(date -u +%FT%TZ)" >> "$OUT/queue.log"
if [ -f "$OUT/seam_R6_A1.npz" ]; then
  bash "$P/code/score_queue6.sh" navhard_two_stage "$SC" e6s1000 R6_A1="$OUT/seam_R6_A1.npz"
fi
bash "$P/code/score_queue6.sh" navhard_two_stage "$P/raw/harness_repro_navhard" e6repro \
   CV_official=OFFICIAL:constant_velocity_agent
echo "ZZNAVHARDS1000DONEZZ $(date -u +%FT%TZ)" >> "$OUT/queue.log"

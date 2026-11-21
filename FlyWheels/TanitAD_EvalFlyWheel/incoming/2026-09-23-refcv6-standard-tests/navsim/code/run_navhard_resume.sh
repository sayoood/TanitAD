#!/usr/bin/env bash
# RESUME the navhard step-1,000 R6_A1 bridge (SPEC §6 pipeline validation, all 5,912 tokens) from its
# rows jsonl, then score it through the RAM-gated retrying queue.
# History: first paused by SPEC amendment A1 (dropped-frame histories, 2026-09-23); relaunched with the
# anchor-units fix (A2) at 23:34:13Z; PAUSED BY THE OPERATOR at 23:43:17Z with 138 corrected rows
# banked, because the box was at 1.7 GB free and E1's guard had just aborted the KH-nav CV scoring —
# this run is the lowest-value job on the box (pipeline validation, superseded by any milestone run).
# So the resume (a) waits for the warmup bridge to finish (two bridges only split the same CPU),
# (b) waits for >= RAM_MIN_GB free on 3 consecutive 60 s samples, and (c) runs at BELOW-NORMAL
# priority (children inherit it on Windows), so a milestone run at normal priority pre-empts it.
# The KH-nav CV control is NOT run here (its own queue owns raw/harness_repro_navhard/).
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
OUT="$P/raw/bridge_navhard_s1000"
SC="$P/raw/scores_navhard_s1000"
B=C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923
RAM_MIN_GB="${RAM_MIN_GB:-5}"
mkdir -p "$OUT" "$SC"
until grep -q "BRIDGE_RC" "$P/raw/bridge_warmup_s1000/queue.log" 2>/dev/null; do sleep 60; done
ok=0
while [ $ok -lt 3 ]; do
  f=$("$PY" -c "import psutil; print(int(psutil.virtual_memory().available/2**30))")
  if [ "$f" -ge "$RAM_MIN_GB" ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -lt 3 ] && sleep 60
done
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval" CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8
echo "NAVHARD_BRIDGE_RESUME free=${f}GB rows=$(wc -l < "$OUT/rows_R6_A1.jsonl") prio=BelowNormal $(date -u +%FT%TZ)" >> "$OUT/queue.log"
"$PY" -c "import psutil, subprocess, sys; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS); sys.exit(subprocess.call(sys.argv[1:]))" \
   "$PY" "$P/code/run_bridge6.py" --split navhard_two_stage --arms R6_A1 \
   --inputs C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json \
   --speed "$P/raw/inputs/speed_limits_navhard_two_stage.json" \
   --bank2 "$B/navhard_s2" --bank1 "$B/navhard_s1" \
   --road-plane "$P/raw/inputs/road_plane_navhard_warmup_logs.json" \
   --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step1000.pt --config D:/refcv6_eval_kit/ckpt/config.json \
   --ckpt-md5 7c3ad3c1fbf3d30be5c7733e7b436c65 --device cpu --precision fp32 --exact-dedup \
   --threads 8 --out "$OUT" >> "$OUT/bridge.log" 2>&1
echo "NAVHARD_BRIDGE_RESUMED_RC=$? $(date -u +%FT%TZ)" >> "$OUT/queue.log"
if [ -f "$OUT/seam_R6_A1.npz" ]; then
  bash "$P/code/score_queue6.sh" navhard_two_stage "$SC" e6s1000 R6_A1="$OUT/seam_R6_A1.npz"
fi
echo "ZZNAVHARDRESUMEDONEZZ $(date -u +%FT%TZ)" >> "$OUT/queue.log"

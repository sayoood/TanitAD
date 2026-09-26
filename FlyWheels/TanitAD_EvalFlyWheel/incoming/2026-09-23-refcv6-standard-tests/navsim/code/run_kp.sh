#!/usr/bin/env bash
# Control KP (SPEC §11): R6_A1 on warmup with the step-1,000 checkpoint on CUDA bf16 (as trained),
# the same per-scene seeds as the CPU fp32 run in raw/bridge_warmup_s1000/ — the precision floor that
# any cross-checkpoint read with differing devices is held against.
# Waits for the GPU gate (the brief's: memory.used < 4300 MiB, no other python on the card, >= 8 GB
# free RAM) for up to KP_WAIT_S; re-measures K0/KD on CUDA first (a KD miss drops the dedup lever);
# then bridges, compares the trajectories, scores (RAM-gated retrying queue) and compares the scores.
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
OUT="$P/raw/controls/KP_step1000"
E2RAW="$P/../../2026-09-19-navsim-refcv4b-bridge/raw"
CK=D:/refcv6_eval_kit/ckpt/ckpt_step1000.pt
CF=D:/refcv6_eval_kit/ckpt/config.json
KP_WAIT_S="${KP_WAIT_S:-28800}"
mkdir -p "$OUT"
L="$OUT/kp.log"
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval" OMP_NUM_THREADS=8
echo "KP_WAIT_GATE $(date -u +%FT%TZ)" >> "$L"
t=0
until "$PY" -c "
import importlib.util, sys
s = importlib.util.spec_from_file_location('g', r'$P/code/run_bridge6.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
sys.exit(0 if m.gpu_gate()['ok'] else 1)" 2>/dev/null; do
  [ $t -ge "$KP_WAIT_S" ] && { echo "KP_GATE_NEVER_OPENED after ${t}s $(date -u +%FT%TZ)" >> "$L"; exit 0; }
  sleep 60; t=$((t+60))
done
echo "KP_GATE_OPEN after ${t}s $(date -u +%FT%TZ)" >> "$L"
R6_TEST_DEVICE=cuda R6_TEST_CKPT="$CK" R6_TEST_CONFIG="$CF" "$PY" -m pytest -q -p no:cacheprovider \
   "$P/tests/test_model_seam6.py" -k "K0 or KD" > "$OUT/cuda_controls.log" 2>&1
KD="$P/raw/controls/KD_exact_dedup_cuda.json"
DEDUP=""
if [ -f "$KD" ] && "$PY" -c "import json,sys; k=json.load(open(r'$KD')); sys.exit(0 if k['sel_identical']==k['n'] and k['max_abs_traj_diff_m']<1e-3 else 1)"; then
  DEDUP="--exact-dedup"
fi
echo "KP_CUDA_CONTROLS dedup=${DEDUP:-OFF} $(date -u +%FT%TZ)" >> "$L"
"$PY" "$P/code/run_bridge6.py" --split warmup_two_stage --arms R6_A1 \
   --inputs "$E2RAW/navsim_agent_inputs.json" --speed "$P/raw/inputs/speed_limits_warmup_two_stage.json" \
   --bank2 C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923/warmup_two_stage \
   --road-plane "$P/raw/inputs/road_plane_navhard_warmup_logs.json" \
   --ckpt "$CK" --config "$CF" --ckpt-md5 7c3ad3c1fbf3d30be5c7733e7b436c65 \
   --device cuda --precision auto $DEDUP --gpu-wait-s 3600 --threads 8 --out "$OUT/bridge" \
   > "$OUT/bridge.log" 2>&1
echo "KP_BRIDGE_DONE $(date -u +%FT%TZ)" >> "$L"
"$PY" "$P/code/kp_compare.py" --a-rows "$P/raw/bridge_warmup_s1000/rows_R6_A1.jsonl" \
   --b-rows "$OUT/bridge/rows_R6_A1.jsonl" --inputs "$E2RAW/navsim_agent_inputs.json" \
   --out "$OUT/KP_precision_floor_trajectories.json" > "$OUT/compare1.log" 2>&1
if [ -f "$OUT/bridge/seam_R6_A1.npz" ]; then
  bash "$P/code/score_queue6.sh" warmup_two_stage "$OUT/scores" e6kp R6_A1="$OUT/bridge/seam_R6_A1.npz"
  "$PY" "$P/code/kp_compare.py" --a-rows "$P/raw/bridge_warmup_s1000/rows_R6_A1.jsonl" \
     --b-rows "$OUT/bridge/rows_R6_A1.jsonl" --inputs "$E2RAW/navsim_agent_inputs.json" \
     --a-csv "$P/raw/scores_warmup_s1000/score_R6_A1.csv" --b-csv "$OUT/scores/score_R6_A1.csv" \
     --out "$OUT/KP_precision_floor.json" > "$OUT/compare2.log" 2>&1
fi
echo "ZZKPDONEZZ $(date -u +%FT%TZ)" >> "$L"

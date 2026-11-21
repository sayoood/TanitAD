#!/usr/bin/env bash
# VALIDATION ONLY (SPEC §6), step 1,500, CUDA NATIVE path (KD failed on CUDA at 1.196 mm > 1 mm ->
# the milestone runs CUDA without the exact dedup): warmup all arms, then bridge-only input-path
# smokes of navtest (--limit 12) and navhard (--limit 20). Run under lock_run.py (holds the GPU lock).
set -u
TOK="$1"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
P=D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim
T=C:/Users/Admin/ev7nav/FlyWheels/TanitAD_EvalFlyWheel/incoming
R6IN=$T/2026-09-23-refcv6-standard-tests/navsim/raw/inputs
B=C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923
O=$P/raw/validation/step1500_cuda
CK="--ckpt D:/refcv7_eval_kit/ckpt/ckpt_1500.pt --ckpt-md5 c35966f799b1c724515ba1281736fd48 --config D:/refcv7_eval_kit/ckpt/config.json --device cuda --threads 6 --gpu-lock-token $TOK"
LBL="VALIDATION ONLY (step 1500, SPEC sec. 6), CUDA native"
export PYTHONIOENCODING=utf-8
$PY $P/code/run_bridge7.py --split warmup_two_stage --arms R7_A1,R7_A1_s1,R7_FILTOFF,R7_BLIND,R7_VMAXOFF,R7_A1NT,R7_NAVOFF --derived --inputs $T/2026-09-19-navsim-refcv4b-bridge/raw/navsim_agent_inputs.json --speed $R6IN/speed_limits_warmup_two_stage.json --road-plane $R6IN/road_plane_navhard_warmup_logs.json --bank2 $B/warmup_two_stage $CK --label "$LBL" --out $O/bridge_warmup
$PY $P/code/run_bridge7.py --split navtest --arms R7_A1,R7_VMAXORACLE --derived --limit 12 --inputs D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz --speed $P/raw/inputs/speed_limits_navtest.json --speed-oracle $R6IN/vmax_oracle_navtest.json --road-plane $R6IN/road_plane_navhard_warmup_logs.json --bank1 D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank --bank1-kind navtest --logs-root D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test $CK --label "$LBL (navtest input-path smoke, 12 tokens)" --out $O/smoke_navtest
$PY $P/code/run_bridge7.py --split navhard_two_stage --arms R7_A1 --derived --limit 20 --inputs C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json --speed $R6IN/speed_limits_navhard_two_stage.json --road-plane $R6IN/road_plane_navhard_warmup_logs.json --bank2 $B/navhard_s2 --bank1 $B/navhard_s1 $CK --label "$LBL (navhard input-path smoke, 20+2 tokens)" --out $O/smoke_navhard
echo ZZCUDAVALDONEZZ

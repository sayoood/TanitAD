#!/bin/sh
# smoke_r8oracle_cpu.sh -- the CPU smoke exactly as run 2026-10-21 (2 episodes x 3 windows, seed 1, CPU fp32 trunk, NO GPU):
#   (a) the plain battery child (roll_seed_r8_pf.py, unwrapped) -> plain_dump_s1 ;  (b) the oracle wrapper, every arm -> dump_s1 + oracle_s1.npz
#   (c) r8oracle_analyze.py --smoke: C1 (wrapper OWN == plain child, same device), C2, C3, C3b, C4, C6, C7.
# Run (a) and (b) one after the other: run CONCURRENTLY on the same CPU they differed by 2.9e-6 m (float noise), alone by exactly 0.
S=${1:-C:/Users/Admin/refcv8_pf/r8oracle/smoke2}; W=C:/Users/Admin/refcv8_pf/r8oracle/smoke/windows_smoke.json
HERE=C:/Users/Admin/refcv8_pf/r8oracle/code
PFC=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/code
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 R8_NO_POSE_SYNC=1 CUDA_VISIBLE_DEVICES=-1
A="--ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed 1 --windows-json $W --device cpu --os-only --smoke-cpu-fp32-trunk"
mkdir -p "$S"
[ -d "$S/plain_dump_s1" ] || "$PY" -X utf8 "$PFC/roll_seed_r8_pf.py" $A --dump-dir "$S/plain_dump_s1" --out-json "$S/plain_roll_s1.json" > "$S/plain_smoke.log" 2>&1
"$PY" -X utf8 "$HERE/r8oracle_roll.py" --oracle-out "$S/oracle_s1.npz" --save-every 2 -- $A --dump-dir "$S/dump_s1" --out-json "$S/roll_s1.json" > "$S/oracle_smoke.log" 2>&1
"$PY" -X utf8 "$HERE/r8oracle_analyze.py" --root "$S" --seeds 1 --smoke --plain-dump "$S/plain_dump_s1" > "$S/analyze_smoke.log" 2>&1
tail -1 "$S/analyze_smoke.log"

#!/bin/sh
# C1 reference: the UNMODIFIED battery child (roll_seed_r8_pf.py) on CPU, smoke windows, seed $1
S=$1; O=/c/Users/Admin/refcv8_pf/perc2plan/smoke/REF2_s$S; mkdir -p $O
HOLDW=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=-1 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
echo "ZZREFSTART $(date +%FT%T%z)" > $O/run.log
/c/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 $HOLDW/code/roll_seed_r8_pf.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed $S --dump-dir $O/dump --windows-json C:/Users/Admin/refcv8_pf/perc2plan/smoke/windows_smoke.json --device cpu --out-json $O/roll_s$S.json --smoke-cpu-fp32-trunk --os-only >> $O/run.log 2>&1
echo "ZZREFEND rc=$? $(date +%FT%T%z)" >> $O/run.log

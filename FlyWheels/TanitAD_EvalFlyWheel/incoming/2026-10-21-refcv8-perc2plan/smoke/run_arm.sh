#!/bin/sh
# perc2plan CPU smoke: one arm through the wrapper, smoke windows, seed $2.  $1 = arm.
ARM=$1; S=$2; SM=/c/Users/Admin/refcv8_pf/perc2plan/smoke; O=$SM/${ARM}_s$S; mkdir -p $O
HOLDW=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=-1 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
BANK=""
case $ARM in BASE) BANK="--donor-bank-out $SM/BASE_s$S/donor_bank_s$S.pt";; *SHUF) BANK="--donor-bank-in $SM/BASE_s$S/donor_bank_s$S.pt";; esac
echo "ZZARMSTART $ARM $(date +%FT%T%z)" > $O/run.log
/c/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 C:/Users/Admin/refcv8_pf/perc2plan/code/perc2plan_roll_r8_pf.py --arm $ARM --pf-code $HOLDW/code \
  --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed $S --dump-dir $O/dump \
  --windows-json $SM/windows_smoke.json --out-json $O/roll_s$S.json --meta-json $O/meta_s$S.json --device cpu --os-only --smoke-cpu-fp32-trunk $BANK >> $O/run.log 2>&1
echo "ZZARMEND rc=$? $(date +%FT%T%z)" >> $O/run.log

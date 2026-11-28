#!/bin/sh
# smoke_pk_cpu.sh -- PREREG R8-PICKER control K1 (CPU, dev box): the capture wrapper vs the UNWRAPPED child, same device / seed / windows,
# 2 episodes x 4 windows (smoke/windows_smoke.json), seed 0 FULL ARM and seed 1 --os-only, EMA-48,852.  No GPU, no lock.
# Markers ZZPKSMK<NAME>ZZ in smoke/smoke.log.
P=C:/Users/Admin/refcv8_pf/picker; S=$P/smoke
HOLDW=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json; W=$S/windows_smoke.json
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=-1 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
mark() { echo "ZZPKSMK$1ZZ $(date +%FT%T%z) $2" >> "$S/smoke.log"; }
mark START "pid $$"
for spec in "0:" "1:--os-only"; do
  sd=${spec%%:*}; oso=${spec#*:}
  for form in wrapped unwrapped; do
    D=$S/$form/s$sd; mkdir -p $D
    if [ $form = wrapped ]; then
      "$PY" -X utf8 $P/code/fan_roll_r8_pk.py --pf-code $HOLDW/code --ckpt $CK --config $CFG --seed $sd --dump-dir $D/dump \
          --windows-json $W --out-json $D/roll_s$sd.json --fan-meta-json $D/fan_roll_s$sd.json --device cpu --smoke-cpu-fp32-trunk $oso > $D/roll.log 2>&1
    else
      "$PY" -X utf8 $HOLDW/code/roll_seed_r8_pf.py --ckpt $CK --config $CFG --seed $sd --dump-dir $D/dump \
          --windows-json $W --device cpu --out-json $D/roll_s$sd.json --smoke-cpu-fp32-trunk $oso > $D/roll.log 2>&1
    fi
    mark "ROLL_${form}_s$sd" "rc=$?"
  done
done
mark END ""

#!/bin/bash
# CPU-ONLY code-path smoke (no GPU touched): 1 window, no LAW control. Validates remap/build/dataset/forward/pack on Thor.
source /home/nvidia/r8ns/perc8/code/thor_env.sh
export OMP_NUM_THREADS=4 R8_NO_POSE_SYNC=1 CUDA_VISIBLE_DEVICES=""
cd $P8/code
nice -n 10 $PY -X utf8 perc_pass_r8.py --ckpt /home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt --config /home/nvidia/r8ns/ckpt/config.json \
  --windows-json $P8/inputs/s2_windows.json --out $P8/out/smoke_cpu/perc --device cpu --max-windows 1 --law-control 0 \
  > $P8/logs/smoke_cpu.log 2>&1 < /dev/null
echo $? > $P8/status/smoke_cpu.exit

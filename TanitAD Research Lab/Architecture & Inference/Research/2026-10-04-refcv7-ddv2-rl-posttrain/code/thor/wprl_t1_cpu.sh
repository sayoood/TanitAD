#!/bin/bash
source /home/nvidia/refcv7_post/rl/wprl_env.sh
export CUDA_VISIBLE_DEVICES=""
BC=$R/battery_code
mkdir -p $R/battery_cpu
$PY - <<PYEOF
import json
d = json.load(open("$R/battery_win/s2_windows.json"))
ks = sorted(d)[:2]
json.dump({k: d[k][:1] for k in ks}, open("$R/battery_cpu/tiny_windows.json", "w"))
PYEOF
cd $BC
$PY roll_seed_r7.py --ckpt $R/ckpt_50400.pt --config /home/nvidia/refcv7_run/runs/refcv7-r101-s0/config.json --seed 0 --dump-dir $R/battery_cpu/dump --windows-json $R/battery_cpu/tiny_windows.json --device cpu --out-json $R/battery_cpu/roll.json --smoke-cpu-fp32-trunk --os-only > $R/battery_cpu/roll.log 2>&1 < /dev/null
echo "rc=$? artifact=$( [ -s $R/battery_cpu/roll.json ] && echo yes || echo NO )"

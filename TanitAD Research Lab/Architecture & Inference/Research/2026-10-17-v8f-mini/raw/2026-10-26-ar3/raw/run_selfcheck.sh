#!/usr/bin/env bash
# usage: run_selfcheck.sh <fullshape|live4> <SP>
export CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="C:/Users/Admin/ar3_work/stack;C:/Users/Admin/ar3_work/taniteval"
cd C:/Users/Admin/ar3_work
ARGV=$(C:/Users/Admin/venvs/tanitad/Scripts/python.exe -c "import json,sys;print(chr(10).join(json.load(open(sys.argv[1]))))" "$2/raw/argv_L_s0_dev_$1.json" | tr -d '\r')
mapfile -t A <<< "$ARGV"
start=$(date +%s)
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -u stack/scripts/v8f_mini_train.py "${A[@]}"
echo "ZZRC rc=$? seconds=$(( $(date +%s) - start ))"

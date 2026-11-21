#!/bin/bash
# usage: run_gate_clock.sh <A|B|C>   -- the gate's OWN child job (launch_gate.py check --checks G-CLOCK) on Thor CPU
set -u
H=/home/nvidia/clockref203
TREE=/home/nvidia/refcv6_run/fa43d9c5ce
L=$1
AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
if [ "$AV" -lt 25 ]; then echo "ABORT available ${AV} GB < 25"; exit 3; fi
export PYTHONPATH=$TREE/stack:$TREE/taniteval:$TREE/stack/scripts
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=-1
cd $H/work
nice -n 19 /home/nvidia/venvs/tanitad-train/bin/python $TREE/stack/scripts/launch_gate.py check --ctx $H/work/ctx_$L.json --checks G-CLOCK > $H/raw/gate_$L.log 2>&1 < /dev/null
echo "GATE_CHILD_RC=$?" >> $H/raw/gate_$L.log

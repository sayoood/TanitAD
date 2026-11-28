#!/bin/bash
# usage: run_check.sh <job> <checks>   -- the gate's OWN child job on Thor CPU; watchdog kills ONLY the child PID if MemAvailable < 25 GB
set -u
JOB=$1; CHECKS=$2
T=/home/nvidia/refcv6_run/d014ff2fd6
W=/home/nvidia/pregate_d014ff2
AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
if [ "$AV" -lt 40 ]; then echo "ABORT available ${AV} GB < 40"; exit 3; fi
export PYTHONPATH=$T/stack:$T/taniteval:$T/stack/scripts
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=-1
cd $W/work
LOG=$W/raw/gate_${JOB}.log
nice -n 19 /home/nvidia/venvs/tanitad-train/bin/python $T/stack/scripts/launch_gate.py check --ctx $W/work/ctx_${JOB}.json --checks $CHECKS > $LOG 2>&1 < /dev/null 200>&- &
CPID=$!
echo $CPID > $W/work/pid_${JOB}.txt
(
  while kill -0 $CPID 2>/dev/null; do
    A=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$A" -lt 25 ]; then echo "WATCHDOG_KILL avail=${A}GB" >> $W/raw/watchdog_${JOB}.log; kill $CPID; break; fi
    echo "$(date -u +%H:%M:%S) avail=${A}GB" > $W/raw/watchdog_${JOB}_last.txt
    sleep 5
  done
) 200>&- >/dev/null 2>&1 &
wait $CPID
echo "GATE_CHILD_RC=$?" >> $LOG

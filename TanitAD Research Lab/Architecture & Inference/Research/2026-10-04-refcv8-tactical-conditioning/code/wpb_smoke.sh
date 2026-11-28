#!/bin/bash
# WP-B CPU smoke on Thor (no GPU): I-W, 3 training steps of T2, eval of T2 on 2 windows with the forced passes.
set -u
W=/home/nvidia/refcv8_wpb
PY=/home/nvidia/venvs/tanitad-train/bin/python
export REFCV6_REPO=$W/tree REFCV6_KIT=/home/nvidia
export PYTHONPATH=$W/tree/stack:$W/tree/taniteval
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
O=$W/out_smoke_cpu
mkdir -p $O
cd $W/code
C="--cache /home/nvidia/refcv8_r1/smoke_cache --out $O --device cpu"
nice -n 19 $PY wpb_arms.py --phase iw $C --max-eval-windows 2 > $W/logs/smoke_iw.log 2>&1 < /dev/null; echo "IW_RC=$?" > $W/logs/smoke_rc.txt
nice -n 19 $PY wpb_arms.py --phase train --arm T2 --steps 3 --batch 2 $C > $W/logs/smoke_train.log 2>&1 < /dev/null; echo "TRAIN_RC=$?" >> $W/logs/smoke_rc.txt
nice -n 19 $PY wpb_arms.py --phase eval --arm T2 --eval-seeds 0 --ctrl --max-eval-windows 2 $C > $W/logs/smoke_eval.log 2>&1 < /dev/null; echo "EVAL_RC=$?" >> $W/logs/smoke_rc.txt
echo DONE >> $W/logs/smoke_rc.txt

#!/bin/bash
# SPEC_WPB (REGISTERED c952d4b4...) -- runs the CPU scoring once the WP-B chain is finished. Detached, no GPU, no lock.
# Waits on the chain's own done-marker (ZZALLDONEZZ) OR on the chain process being gone; scores whatever arms exist and
# says which. The ARTIFACT (arms/score_wpb.json) is the evidence that scoring ran, never this script's exit code.
set -u
W=/home/nvidia/refcv8_wpb
PY=/home/nvidia/venvs/tanitad-train/bin/python
CHAIN_PID=${1:?chain pid}
export PYTHONPATH=$W/tree/stack:$W/tree/taniteval
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
until grep -q "ZZALLDONEZZ" $W/logs/chain.out 2>/dev/null || ! kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 120; done
echo "[wpb-score-wait] $(date -u +%H:%M:%S) chain finished (marker: $(grep -c ZZALLDONEZZ $W/logs/chain.out 2>/dev/null))"
echo "[wpb-score-wait] arms evaluated: $(ls $W/arms/*/eval_s*.pt 2>/dev/null | wc -l) eval files"
cd $W/code
nice -n 10 $PY wpb_score.py --arms $W/arms > $W/logs/score_wpb.log 2>&1 < /dev/null
echo "[wpb-score-wait] $(date -u +%H:%M:%S) score_wpb.json $( [ -s $W/arms/score_wpb.json ] && echo WRITTEN || echo MISSING )"

#!/usr/bin/env bash
# D-SAFE-CAL — four arms, exactly as pre-registered. Sequential (one 4060).
set -u
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6 PYTHONPATH="./stack"
O="C:/Users/Admin/tanitad-data/rl-pilot"
CK="C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt"
TR="C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74"
VA="C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836"

run () {   # name reward w_anchor d_safe
  echo "=== ARM $1 : reward=$2 w_anchor=$3 d_safe=$4 ==="
  "$PY" -u stack/scripts/rl_pilot_refc21.py \
    --ckpt "$CK" --train-epdir "$TR" --train-agents "$O/pilot_train_agents.jsonl" \
    --val-epdir "$VA" --val-agents "$O/pilot_val_agents.jsonl" \
    --out "$O/$1" --steps 2000 --batch 2 --seed 0 \
    --reward "$2" --w-anchor "$3" --proximity-safe-m "$4" > "$O/$1.log" 2>&1
  echo "ARM $1 RC=$?"
}

run dsafe-A-w1-d2    proximity  1.0  2.0
run dsafe-B-w10-d2   proximity 10.0  2.0
run dsafe-C-w1-d5    proximity  1.0  5.0
run dsafe-D-floor-d2 proximity0 1.0  2.0
echo "ALL_ARMS_DONE"

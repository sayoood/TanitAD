#!/bin/bash
# Wait for the full re-eval chain (base -> ft800 -> ft300) then score all three.
set -uo pipefail
echo "[wait] blocking on CHAIN_DONE ..."
for i in $(seq 1 220); do [ -f /workspace/gate1_reeval_CHAIN_DONE ] && break; sleep 30; done
if [ ! -f /workspace/gate1_reeval_CHAIN_DONE ]; then
  echo "[wait] TIMEOUT — chain not done"; fi
echo "=== base DONE:  $(cat /workspace/gate1_reeval_base/DONE 2>/dev/null)"
echo "=== ft800 DONE: $(cat /workspace/gate1_reeval_ft800/DONE 2>/dev/null)"
echo "=== ft300 DONE: $(cat /workspace/gate1_reeval_ft300/DONE 2>/dev/null)"
cd /workspace/alpa-invest/alpasim
export PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts
export CUDA_VISIBLE_DEVICES=""
echo "############ SCORE base vs ft800 vs ft300 ############"
.venv/bin/python /workspace/gate1_score.py \
  base:/workspace/gate1_reeval_base ft800:/workspace/gate1_reeval_ft800 ft300:/workspace/gate1_reeval_ft300
echo "############ PAIRED base vs ft300 ############"
.venv/bin/python /workspace/gate1_score.py \
  base:/workspace/gate1_reeval_base ft300:/workspace/gate1_reeval_ft300 2>/dev/null | grep -A6 "PAIRED"

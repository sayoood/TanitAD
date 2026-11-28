#!/bin/bash
# Batch 3 (CPU-only, read-only on /workspace): POOL-ALONE throughput (G-LIVE-LAB's quantity, bar 6.735 sets/s) of the chunked pool at W 4/6/8, after batch 2 ends.
set -u
while [ ! -f /tmp/p6diag2/DONE ]; do sleep 5; done
OUT=/tmp/p6diag3; mkdir -p "$OUT"; rm -f "$OUT"/DONE
source /workspace/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"; MC=/workspace/data/refe_navtrain/metric_cache
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd /tmp/p6diag_tree/refe
TOKS="$("$PY" p6_prep.py sample-tokens --cache "$MC" --n 256 | sed -n 's/^ZZTOKENS //p')"
echo "start $(date -u +%FT%TZ)" > "$OUT/log.txt"
for arm in ${ARMS:-4:1 4:2 6:2 8:2}; do
  W="${arm%%:*}"; C="${arm#*:}"
  env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK timeout 900 "$PY" p6_cpu_probe.py pool --cache "$MC" --tokens "$TOKS" --workers "$W" --chunks "$C" --python "$PY" \
    --n-sets 256 --mode thru --out "$OUT/pool_thru_w${W}_c${C}.json" > "$OUT/pool_w${W}_c${C}.log" 2>&1
  echo "arm $arm rc $? $(grep -o '"sets_per_s": [0-9.]*' "$OUT/pool_thru_w${W}_c${C}.json") $(grep -o '"cpu_per_set_job_s": [0-9.]*' "$OUT/pool_thru_w${W}_c${C}.json") $(date -u +%FT%TZ)" >> "$OUT/log.txt"
done
touch "$OUT/DONE"

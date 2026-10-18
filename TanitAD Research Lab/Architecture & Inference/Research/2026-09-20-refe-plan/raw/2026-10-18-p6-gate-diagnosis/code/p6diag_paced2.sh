#!/bin/bash
# Batch 2 (CPU-only, read-only on /workspace): the chunked pool at W 4/6/8 beside a 1-core trainer stand-in whose OWN CPU share is measured
# (/proc/<pid>/stat utime+stime over the arm) -- does a wider pool starve the trainer's main thread under the 8.5-CPU quota?
set -u
while [ ! -f /tmp/p6diag/DONE ]; do sleep 5; done
OUT=/tmp/p6diag2; mkdir -p "$OUT"; rm -f "$OUT"/DONE
source /workspace/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"; MC=/workspace/data/refe_navtrain/metric_cache
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd /tmp/p6diag_tree/refe
TOKS="$("$PY" p6_prep.py sample-tokens --cache "$MC" --n 384 | sed -n 's/^ZZTOKENS //p')"
"$PY" -c "while True: pass" & SPIN=$!
TCK=$(getconf CLK_TCK)
cpu_of() { awk '{print $14 + $15}' "/proc/$1/stat"; }
echo "start $(date -u +%FT%TZ) spinner $SPIN tck $TCK" > "$OUT/log.txt"
a0=$(cpu_of $SPIN); t0=$(date +%s.%N); sleep 30; a1=$(cpu_of $SPIN); t1=$(date +%s.%N)
echo "arm idle spinner_share $(awk -v a="$a0" -v b="$a1" -v k="$TCK" -v s="$t0" -v e="$t1" 'BEGIN{printf "%.4f", (b-a)/k/(e-s)}')" >> "$OUT/log.txt"
for arm in ${ARMS:-4:1 4:2 6:2 8:2}; do
  W="${arm%%:*}"; C="${arm#*:}"
  a0=$(cpu_of $SPIN); t0=$(date +%s.%N)
  env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK timeout 900 "$PY" p6_cpu_probe.py pool --cache "$MC" --tokens "$TOKS" --workers "$W" --chunks "$C" --python "$PY" \
    --n-sets 384 --mode paced --n-micro 96 --out "$OUT/pool_paced_w${W}_c${C}.json" > "$OUT/pool_w${W}_c${C}.log" 2>&1
  rc=$?; a1=$(cpu_of $SPIN); t1=$(date +%s.%N)
  echo "arm $arm rc $rc spinner_share $(awk -v a="$a0" -v b="$a1" -v k="$TCK" -v s="$t0" -v e="$t1" 'BEGIN{printf "%.4f", (b-a)/k/(e-s)}') $(grep -o '"stall_frac": [0-9.]*' "$OUT/pool_paced_w${W}_c${C}.json") $(date -u +%FT%TZ)" >> "$OUT/log.txt"
done
kill "$SPIN"; touch "$OUT/DONE"

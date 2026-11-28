#!/bin/bash
# CPU-only diagnostic (no GPU, no training, no chain, no gate): the online labeller pool on THIS pod's cgroup, at the trainer's clock (p6_cpu_probe.py pool --mode paced),
# beside a 1-core trainer stand-in (the trainer main thread spin-waits on CUDA syncs).  Read-only on /workspace; writes only under /tmp/p6diag.
set -u
OUT=/tmp/p6diag; mkdir -p "$OUT"; rm -f "$OUT"/DONE "$OUT"/pool_*.json
rm -rf /tmp/p6diag_tree; cp -r /workspace/refe-plan-p6 /tmp/p6diag_tree
source /workspace/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"; MC=/workspace/data/refe_navtrain/metric_cache
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd /tmp/p6diag_tree/refe
echo "start $(date -u +%FT%TZ)" > "$OUT/log.txt"
TOKS="$("$PY" p6_prep.py sample-tokens --cache "$MC" --n "${NTOK:-384}" | sed -n 's/^ZZTOKENS //p')"
echo "tokens $(echo "$TOKS" | tr ',' '\n' | grep -c .) $(date -u +%FT%TZ)" >> "$OUT/log.txt"
"$PY" -c "while True: pass" & SPIN=$!
echo "spinner pid $SPIN" >> "$OUT/log.txt"
for arm in ${ARMS:-4:1 8:1 4:2 8:2 6:1 4:1r}; do
  W="${arm%%:*}"; C="${arm#*:}"; C="${C%r}"
  env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK timeout 900 "$PY" p6_cpu_probe.py pool --cache "$MC" --tokens "$TOKS" --workers "$W" --chunks "$C" --python "$PY" \
    --n-sets "${NSETS:-384}" --mode paced --n-micro "${NMICRO:-96}" --out "$OUT/pool_paced_w${W}_c${C}_${arm//:/_}.json" > "$OUT/pool_${arm//:/_}.log" 2>&1
  echo "arm $arm rc $? $(date -u +%FT%TZ) $(grep -o '"stall_frac": [0-9.]*' "$OUT/pool_paced_w${W}_c${C}_${arm//:/_}.json" 2>/dev/null)" >> "$OUT/log.txt"
done
kill "$SPIN"
echo "end $(date -u +%FT%TZ)" >> "$OUT/log.txt"; touch "$OUT/DONE"

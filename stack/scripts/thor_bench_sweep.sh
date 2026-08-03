#!/usr/bin/env bash
# REF-C batch sweep on Jetson Thor — one PROCESS PER BATCH so each run reports an
# uncontaminated torch.cuda.max_memory_allocated (the only GPU-memory probe that
# works on Thor; see thor_bench_probe.py for the three that do not).
#
# Usage: thor_bench_sweep.sh <out_root> <steps> <batch> [batch...]
set -u
OUT_ROOT=${1:?out_root}; STEPS=${2:?steps}; shift 2
export PATH=$HOME/venvs/tanitad-train/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=$HOME/TanitAD/stack:$HOME/TanitAD/stack/scripts
export OMP_NUM_THREADS=6
cd "$HOME/TanitAD/stack" || exit 1
mkdir -p "$OUT_ROOT"

for B in "$@"; do
  echo "=== BATCH $B ==="
  THOR_BENCH_OUT="$OUT_ROOT/refc_b${B}.jsonl" \
  timeout 1800 python -u scripts/thor_bench_run.py scripts/refc_train.py \
      --data-root "$HOME/thorbench/data" \
      --out "$OUT_ROOT/refc-b${B}" \
      --config base --mode classifier \
      --steps "$STEPS" --batch "$B" --log-every 1 \
      --episodes 40 --save-every 100000 --workers 4 \
      > "$OUT_ROOT/refc_b${B}.log" 2>&1
  rc=$?
  echo "batch $B exit=$rc"
  if grep -qiE 'out of memory|CUDA error|Killed' "$OUT_ROOT/refc_b${B}.log"; then
    echo "batch $B: OOM/ERROR -> stopping sweep"
    grep -iE 'out of memory|CUDA error' "$OUT_ROOT/refc_b${B}.log" | head -2
    break
  fi
  # free the allocator between arms; the next process starts clean anyway
  sleep 5
done
echo "SWEEP_DONE"

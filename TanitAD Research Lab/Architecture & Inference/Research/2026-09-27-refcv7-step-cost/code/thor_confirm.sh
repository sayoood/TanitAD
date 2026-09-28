#!/bin/bash
# refcv7 step-cost: the <=20-minute THOR confirmation. Run ONLY in a restart window: the trainer
# stopped, its supervisor stopped, the GPU EMPTY. It never touches the run dir, the checkpoint or
# the supervisor; every output goes to $OUT. The verdict is the JSON it writes, never its exit code.
#
#   usage: thor_confirm.sh <launch-tree> <harness-dir> <out-dir> [extra]
#     <launch-tree>  the SAME tree the run launches from (stack/ taniteval/ ...), read-only
#     <harness-dir>  this package's code/ (step_profiler.py rc7_argv.py ab_summarise.py),
#                    md5-verified on ship
#     [extra]        "compile" -> also run the --ab compile_map job (Inductor);
#                    "ckpt"    -> also run --ab ckpt_off (FIRST check the probe's
#                                 cuda_max_mem_gb + ~21 GB against the 128 GB unified memory)
#
# Budget at b16 (compiled trunk, ~10 s/step): job 1 ~8 min, job 2 ~7 min, job 3 ~3 min = ~18 min.
# Order = priority, so a window cut short still yields the lever number first.
set -u
TREE="$1"; H="$2"; OUT="$3"; shift 3
PY=/home/nvidia/venvs/tanitad-train/bin/python
mkdir -p "$OUT"
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -c .)
if [ "$busy" != "0" ]; then
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > "$OUT/REFUSED_GPU_BUSY"; exit 4
fi
export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE/taniteval:$H" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
"$PY" -c "import tanitad; print(tanitad.__file__)" > "$OUT/import_probe.txt" 2>&1
grep -q "^$TREE/stack/tanitad" "$OUT/import_probe.txt" || { echo "wrong tree" > "$OUT/REFUSED_TREE"; exit 5; }
: > "$OUT/identity_pathmap.txt"                     # Thor paths are native: no mapping
# the launch argv as launched (compile kept), b16, 6 workers (Linux fork: the spawn pickling
# defect of the dev box does not apply), no in-run eval, the run dir is the harness's own
COMMON=(--tree "$TREE" --pathmap "$OUT/identity_pathmap.txt" --batch 16 --workers 6
        --keep-compile --no-eval-at-end --log-every 50)
# 1. THE LEVER, in-process A/B (per-class 10 cm signal + box census on logged steps only).
#    Also yields the conflict-step extra (steps 10, 20) and data_wait per step (loader question).
timeout 780 "$PY" "$H/step_profiler.py" "${COMMON[@]}" --rung R7 --mode plain --steps 30 --warm 8 \
  --ab percls_census --ab-block 2 --out "$OUT/ab_percls_census_b16" > "$OUT/ab_percls_census_b16.log" 2>&1
"$PY" "$H/ab_summarise.py" "$OUT/ab_percls_census_b16" "$OUT/ab_percls_census_b16/ab_summary.json" \
  >> "$OUT/ab_percls_census_b16.log" 2>&1
# 2. THE ATTRIBUTION on Thor: synchronised timers (map branch fwd / recompute, s8 tap, slot
#    losses, per-class signal, backward, optimiser, conflict probe on steps 10 and 20).
timeout 660 "$PY" "$H/step_profiler.py" "${COMMON[@]}" --rung R7 --mode timers --steps 22 --warm 8 \
  --out "$OUT/R7_timers_b16" > "$OUT/R7_timers_b16.log" 2>&1
# 3. THE LOADER, CPU only: per-sample cost of one worker's job, warm OS cache, LRUs cleared.
CUDA_VISIBLE_DEVICES=-1 timeout 420 "$PY" "$H/step_profiler.py" --tree "$TREE" --require-cpu \
  --pathmap "$OUT/identity_pathmap.txt" --rung R7 --mode loader --batch 16 --workers 0 \
  --loader-n 32 --loader-clips 8 --no-eval-at-end --out "$OUT/R7_loader_warm8" \
  > "$OUT/R7_loader_warm8.log" 2>&1
for x in "$@"; do
  case "$x" in
    compile)
      timeout 900 "$PY" "$H/step_profiler.py" "${COMMON[@]}" --rung R7 --mode plain --steps 30 \
        --warm 8 --ab compile_map --ab-block 2 --compile-backend inductor \
        --out "$OUT/ab_compile_map_b16" > "$OUT/ab_compile_map_b16.log" 2>&1
      "$PY" "$H/ab_summarise.py" "$OUT/ab_compile_map_b16" "$OUT/ab_compile_map_b16/ab_summary.json" \
        >> "$OUT/ab_compile_map_b16.log" 2>&1 ;;
    ckpt)
      timeout 780 "$PY" "$H/step_profiler.py" "${COMMON[@]}" --rung R7 --mode plain --steps 30 \
        --warm 8 --ab ckpt_off --ab-block 2 --out "$OUT/ab_ckpt_off_b16" > "$OUT/ab_ckpt_off_b16.log" 2>&1
      "$PY" "$H/ab_summarise.py" "$OUT/ab_ckpt_off_b16" "$OUT/ab_ckpt_off_b16/ab_summary.json" \
        >> "$OUT/ab_ckpt_off_b16.log" 2>&1 ;;
  esac
done
ls "$OUT"/*/profile_*.json "$OUT"/*/ab_summary.json "$OUT"/R7_loader_warm8/loader_bench.json \
  2>/dev/null > "$OUT/ARTIFACTS.txt"
echo "ZZ$(wc -l < "$OUT/ARTIFACTS.txt")ZZ"

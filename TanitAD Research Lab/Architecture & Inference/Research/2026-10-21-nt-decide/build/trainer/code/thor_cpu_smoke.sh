#!/usr/bin/env bash
# H-NT-DECIDE-1 E4/E5: the 2-step refc forward/backward smoke on Thor's CPU (the dev box is reserved for C2,
# MM 2026-10-21 21:45). GPU never touched: CUDA_VISIBLE_DEVICES is empty and --device cpu. 4 cores (taskset),
# 4 BLAS/OMP threads, nice 19. Writes RUN_PID / EXIT into the smoke dir; the run dir is run_smoke/.
set -u
R=/home/nvidia/r8ns/nt/labels/a1_trainer_smoke
cd "$R" || exit 2
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$R/nt_tree/stack"
PY=/home/nvidia/venvs/tanitad-train/bin/python
nice -n 19 taskset -c 10-13 "$PY" -X utf8 "$R/code/nt_run_argv.py" --tree "$R/nt_tree/stack" \
    --argv-json "$R/out/argv/nt-smoke.argv.json" -- --device cpu > "$R/smoke.out" 2> "$R/smoke.err" &
echo $! > "$R/RUN_PID"
wait $!
echo $? > "$R/EXIT"

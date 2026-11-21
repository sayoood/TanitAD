#!/usr/bin/bash
# W-A7 chain body. CPU only, 1 thread, TMP on own scratch. Usage: run_w_a7.sh <run_dir> [extra args for w_a7_read.py]
RUN="$1"; shift
export TMP=F:/tanitad_scratch/w_a7_tmp TEMP=F:/tanitad_scratch/w_a7_tmp TMPDIR=F:/tanitad_scratch/w_a7_tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
mkdir -p "$RUN" /f/tanitad_scratch/w_a7_tmp
echo "$(date +%T) chain heartbeat: starting w_a7_read.py pid-of-bash=$$" >> "$RUN/w_a7.log"
/c/Users/Admin/navsim-crun/venv/Scripts/python.exe -u /e/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_launchkit/w_a7/w_a7_read.py --run "$RUN" "$@" >> "$RUN/w_a7.log" 2>&1
rc=$?
echo "$(date +%T) chain end rc=$rc" >> "$RUN/w_a7.log"

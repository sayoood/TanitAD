#!/bin/bash
export PATH="/usr/bin:/bin:$PATH"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 P5R_BLAS_THREADS=1 REFE_SIM_HZ=10 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export NUPLAN_MAPS_ROOT="E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
export P5C_PKG="F:/tanitad_scratch/n4nt/run_a1/w/pk" P5R_DIR="E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5r" P5R_LAMBDA_JSON="E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5r/raw/lambda_star.json" P5C_BANKDIR="F:/tanitad_scratch/n5l1/rj5/bank" P5C_DBROOT="E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
export PYTHONPATH="F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
exec C:/Users/Admin/navsim-crun/venv/Scripts/python.exe -B "F:/tanitad_scratch/rj5nt/w/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-11-11-p5c-readiness/a1/p5c_dual_labels.py" label --shard $1 --nshard 2 --out F:/tanitad_scratch/n5l1/rj5/labels

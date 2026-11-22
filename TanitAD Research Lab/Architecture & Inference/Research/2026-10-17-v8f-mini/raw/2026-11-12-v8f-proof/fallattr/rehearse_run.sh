#!/bin/bash
# rehearse_run.sh -- pod-side runner of rehearse_reader.py (card ADD110, before the rollout queue starts). CPU only. Banked, already open rollouts only.
set -u
V=/workspace/v8proof; P=$V/podread; RD=$P/root/integ10k; CODE=$P/code; FA=$V/fallattr; RH=$FA/rehearse
export HEADWAY_Q3DIR=$CODE/q3 HIERREP_STAGEB=$CODE/stageB TANITEVAL_ROOT=$CODE/tev HEADWAY_JOIN=$P/data/join_eval141.npz HEADWAY_BANK=$P/data/frames.npz SR_ESTREAD_DIR=$RD
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
[ -e "$RH/REHEARSE_CONTROLS.json" ] && { echo "ZZ REHEARSE STOP output exists"; exit 3; }
date -u +%Y-%m-%dT%H:%M:%SZ > "$RH/REHEARSE_START.txt"
( cd "$RD" && /workspace/venv/bin/python "$RH/rehearse_reader.py" "$RH/REHEARSE_CONTROLS.json" > "$RH/rehearse.log" 2>&1 ); rc=$?
echo "ZZ REHEARSE DONE rc $rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RH/REHEARSE_DONE.txt"

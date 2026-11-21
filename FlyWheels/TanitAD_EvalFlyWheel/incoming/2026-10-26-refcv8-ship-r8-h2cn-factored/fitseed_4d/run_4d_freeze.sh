#!/usr/bin/env bash
# AMENDMENT 4d PHASE 1: fit + freeze the H2cn stage at fit seeds 0 (K-REPRO0), 1, 2, ONE AFTER THE OTHER.  NO eval-side output.  CPU only.
# Judged on ARTIFACTS (the .sha256 + FREEZE_INFO json), never on an exit code read through a pipe.
cd "C:/Users/Admin/refcv8_pf/ship_r8_sub_fs" || exit 9
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
mkdir -p logs
for s in 0 1 2; do
  echo "ZZSTART_FS${s}_$(date +%FT%T%z)ZZ" >> logs/run_4d_freeze.marks
  "$PY" -X utf8 h2cn_freeze.py --fit-seed "$s" > "logs/freeze_fs${s}.log" 2> "logs/freeze_fs${s}.err"
  rc=$?
  echo "ZZEND_FS${s}_rc${rc}_$(date +%FT%T%z)ZZ" >> logs/run_4d_freeze.marks
  if [ ! -s "H2CN_FS${s}_STAGE_FROZEN.sha256" ] || [ ! -s "H2CN_FS${s}_FREEZE_INFO.json" ]; then
    echo "ZZSTOP_FS${s}_NO_ARTIFACTZZ" >> logs/run_4d_freeze.marks
    exit 4
  fi
done
echo "ZZALLDONEZZ" >> logs/run_4d_freeze.marks

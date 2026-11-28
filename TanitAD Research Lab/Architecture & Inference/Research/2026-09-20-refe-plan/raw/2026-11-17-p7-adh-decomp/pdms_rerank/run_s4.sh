#!/bin/bash
# MM 2026-11-18: run the re-rank PDMS scoring (the gate sweep is finished) with a free-commit guard.
while true; do
  free=$(powershell -NoProfile -Command "[int][math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" | tr -d '\r\n ')
  [ "${free:-0}" -ge 8 ] && break
  sleep 30
done
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 TMP=F:/tanitad_scratch/rerankscore/tmp TEMP=F:/tanitad_scratch/rerankscore/tmp
mkdir -p /f/tanitad_scratch/rerankscore/tmp
C:/Users/Admin/venvs/tanitad/Scripts/python.exe E:/Projects/TanitAD-artifacts/scorer_gym_20261107/adh_decomp/scripts/s4_pdms_rerank.py F:/tanitad_scratch/rerankscore 2>&1 | tail -20

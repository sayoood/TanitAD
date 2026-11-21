#!/bin/bash
# usage: run1.sh <tree> <test> <tag> <timeout_s>
D="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
d="/f/$1/$D"; W=$(cygpath -w "$d"); cd "$d"
s=$SECONDS
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="$W" SCORER1_TMP='F:\scorer1_triage\w\t' timeout $4 C:/Users/Admin/venvs/tanitad/Scripts/python.exe $2 > /f/scorer1_triage/w/runs/$3.log 2>&1
echo "$3 rc=$? secs=$((SECONDS-s))" >> /f/scorer1_triage/w/runs/summary.txt

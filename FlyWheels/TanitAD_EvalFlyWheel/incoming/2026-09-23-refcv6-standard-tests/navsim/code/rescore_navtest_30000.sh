#!/usr/bin/env bash
# Re-score navtest @ step 30,000 (R6_A1, all 12,146 tokens). Its first scorer FAILED the count guard on
# 2026-09-26 (E1 RAM guard x6; the partial outputs are quarantined in
# raw/milestones/step30000/INVALID_partial_navtest_score_20260926/). The bridge seam is complete (CUDA
# as-trained), so the runner re-runs with ZERO new rows and only the scorer + post-processing do work.
# Waits for the step-30,000 waiter to finish first (its navhard scorer is the other RAM user).
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
L="$P/raw/milestones/rescore_navtest_30000.log"
until grep -q "ZZMILESTONE30000DONEZZ" "$P/raw/milestones/waiter_30000.log" 2>/dev/null; do sleep 120; done
echo "RESCORE_START $(date -u +%FT%TZ)" >> "$L"
R6_EXPENSIVE_GPU_WAIT_S=900 PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval" \
  "$PY" "$P/code/run_navsim_refcv6.py" --ckpt D:/refcv6_eval_kit/ckpt/ckpt_30000.pt \
  --md5 0c5c67b3ecf28495d54348e3436371bf --splits navtest --device auto --gpu-wait-s 900 \
  --out "$(cygpath -m "$P/raw/milestones/step30000")" >> "$L" 2>&1
c="$P/raw/milestones/step30000/scores_navtest/r6s30000_R6_A1/r6s30000_R6_A1.counts.json"
if "$PY" -c "import json,sys; sys.exit(0 if json.load(open(r'$(cygpath -m "$c")')).get('status')=='PASS' else 1)" 2>/dev/null; then
  echo "RESCORE_PASS $(date -u +%FT%TZ)" >> "$L"
else
  echo "RESCORE_NO_PASS $(date -u +%FT%TZ)" >> "$L"
fi
echo "ZZRESCORENAVTEST30000DONEZZ $(date -u +%FT%TZ)" >> "$L"

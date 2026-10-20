#!/bin/bash
# FINISH step: chain mutation matrix (run2, expectation fixed) then the gate-level matrix on the FINAL tree (run3).  One python at a time; stops if C: free < 60 MB.
export PATH="/usr/bin:/bin:$PATH"
H=C:/Users/Admin/par18/hnfp3
export TEMP=$H/tmp_fin TMP=$H/tmp_fin CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
REFE="$H/work/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
free_mb() { df -m /c | awk 'NR==2{print $4}'; }
cd "$REFE"
for step in "test_hnscreen_ctx_chain.py:test_hnscreen_ctx_chain_matrix_run2" "test_hn_ctx_anchor.py:test_hn_ctx_anchor_matrix_run3"; do
  f=${step%%:*}; l=$H/logs/${step##*:}.log
  if [ -f "$l.DONE" ]; then continue; fi
  fm=$(free_mb)
  if [ "$fm" -lt 60 ]; then echo "STOPPED_LOW_DISK free ${fm} MB before $f" > "$l"; echo "EXIT_CODE 99" >> "$l"; exit 99; fi
  echo "free C: ${fm} MB at start" > "$l"
  $PY $f >> "$l" 2>&1 < /dev/null
  echo "EXIT_CODE $?" >> "$l"
  echo "free C: $(free_mb) MB at end" >> "$l"
  echo done > "$l.DONE"
done
echo ALLDONE > $H/logs/finish_matrices.ALLDONE

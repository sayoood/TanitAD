#!/bin/bash
# waits for post_chain.sh to end (ZZPOST-DONE / ZZPOST-STOP in logs/post_chain.log), then writes READ_DAG_HACC.md + MANIFEST_SHA256.txt
PK=/e/Projects/TanitAD-artifacts/daghacc
for i in $(seq 1 480); do
  if grep -q -E "ZZPOST-DONE|ZZPOST-STOP" $PK/logs/post_chain.log 2>/dev/null; then break; fi
  sleep 15
done
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I $PK/post/mk_readme.py > $PK/logs/post_readme.log 2>&1
echo "$(date '+%m-%d %H:%M:%S') readme rc=$?" >> $PK/logs/post_chain.log
echo ZZPOST2-DONE >> $PK/logs/post_chain.log

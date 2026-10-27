#!/usr/bin/env bash
# The queue-reading test files + the DAgger / init-ckpt tests, one pytest run per file, on C:/Users/Admin/ofx = tip 793a78ea's stack/taniteval
# (blob-verified) with oaccflatq's queue (6d26a613) and 6 label-set tests overlaid.  PYTHONPATH pins the tree (the venv's tanitad is the dead G:).
# Disk (rerun 2): basetemp and TMP on F:/TanitAD-scratch (PI ok 2026-10-27, scratch only); the C: watchdog stays.
cd /c/Users/Admin/ofx || exit 2
OUT=E:/Projects/TanitAD-artifacts/dagland/pytest_sweep_oaf_rerun2.log
: > $OUT
export TMP=F:/TanitAD-scratch/tmp TEMP=F:/TanitAD-scratch/tmp CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="C:/Users/Admin/ofx/stack;C:/Users/Admin/ofx/taniteval;C:/Users/Admin/ofx/stack/scripts"
FILES="stack/tests/test_v8f_dagger_train.py stack/tests/test_v8f_ego_perturb.py"
echo "FILES $(echo $FILES | wc -w) start $(date +%H:%M)" >> $OUT
for f in $FILES; do
  a=$(df -m /c | tail -1 | awk '{print $4}'); b=$(du -sm /f/TanitAD-scratch/ofx_pt 2>/dev/null | cut -f1); free=$(( a + ${b:-0} ))
  [ "$free" -lt 400 ] && { echo "SWEEP-STOP-DISK before $f (C: $free MB)" >> $OUT; exit 3; }
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp=F:/TanitAD-scratch/ofx_pt "$f" > /c/Users/Admin/qland/work/sweep_cur.txt 2>&1 &
  pid=$!; t0=$(date +%s); killed=""
  while kill -0 $pid 2>/dev/null; do
    sleep 5; echo "$(date +%H:%M:%S) C: $(df -m /c | tail -1 | awk '{print $4}') pt2 $(du -sm /f/TanitAD-scratch/ofx_pt 2>/dev/null | cut -f1)" >> /c/Users/Admin/qland/work/sweep_disk.txt
    c=$(df -m /c | tail -1 | awk '{print $4}')
    if [ "$c" -lt 80 ]; then kill $pid; killed="DISK-KILL at C: $c MB"; fi
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { kill $pid; killed="TIMEOUT 1800 s"; }
  done
  wait $pid; rc=$?
  echo "$f rc=$rc ${killed} | $(tail -1 /c/Users/Admin/qland/work/sweep_cur.txt)" >> $OUT
  grep -E "^(FAILED|ERROR) " /c/Users/Admin/qland/work/sweep_cur.txt | sed 's/^/    /' | cut -c1-300 >> $OUT
done
echo "SWEEP-DONE $(date +%H:%M)" >> $OUT

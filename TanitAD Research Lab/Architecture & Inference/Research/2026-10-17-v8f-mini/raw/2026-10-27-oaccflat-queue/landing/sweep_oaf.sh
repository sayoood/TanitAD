#!/usr/bin/env bash
# The queue-reading test files + the DAgger / init-ckpt tests, one pytest run per file, on C:/Users/Admin/ofx = tip 793a78ea's stack/taniteval
# (blob-verified) with oaccflatq's queue (6d26a613) and 6 label-set tests overlaid.  PYTHONPATH pins the tree (the venv's tanitad is the dead G:).
# Disk: a file starts only if C: free + the cleared basetemp >= 400 MB; while it runs, C: < 80 MB kills THAT pytest by its own PID (marker DISK-KILL).
cd /c/Users/Admin/ofx || exit 2
OUT=E:/Projects/TanitAD-artifacts/dagland/pytest_sweep_oaf.log
: > $OUT
export CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="C:/Users/Admin/ofx/stack;C:/Users/Admin/ofx/taniteval;C:/Users/Admin/ofx/stack/scripts"
FILES="$(grep -l v8f_mini_box2_queue.cmd stack/tests/*.py) stack/tests/test_v8f_init_ckpt_o_interface.py stack/tests/test_v8f_dagger_train.py stack/tests/test_v8f_ego_perturb.py stack/tests/test_v8f_dagger.py"
echo "FILES $(echo $FILES | wc -w) start $(date +%H:%M)" >> $OUT
for f in $FILES; do
  a=$(df -m /c | tail -1 | awk '{print $4}'); b=$(du -sm /c/Users/Admin/ofx_pt2 2>/dev/null | cut -f1); free=$(( a + ${b:-0} ))
  [ "$free" -lt 400 ] && { echo "SWEEP-STOP-DISK before $f (C: $free MB)" >> $OUT; exit 3; }
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp=C:/Users/Admin/ofx_pt2 "$f" > /c/Users/Admin/qland/work/sweep_cur.txt 2>&1 &
  pid=$!; t0=$(date +%s); killed=""
  while kill -0 $pid 2>/dev/null; do
    sleep 5
    c=$(df -m /c | tail -1 | awk '{print $4}')
    if [ "$c" -lt 80 ]; then kill $pid; killed="DISK-KILL at C: $c MB"; fi
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { kill $pid; killed="TIMEOUT 1800 s"; }
  done
  wait $pid; rc=$?
  echo "$f rc=$rc ${killed} | $(tail -1 /c/Users/Admin/qland/work/sweep_cur.txt)" >> $OUT
  grep -E "^(FAILED|ERROR) " /c/Users/Admin/qland/work/sweep_cur.txt | sed 's/^/    /' | cut -c1-300 >> $OUT
done
echo "SWEEP-DONE $(date +%H:%M)" >> $OUT

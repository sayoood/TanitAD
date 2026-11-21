#!/usr/bin/env bash
# One pytest run per test file, same --basetemp (pytest clears it at the start of each run), so C:'s peak is one file's tmp, not the sweep's.
# Stops (marker SWEEP-STOP-DISK) before a file if C: has < 600 MB free. Output: one summary line per file in $OUT.
cd /c/Users/Admin/ofx || exit 2
OUT=E:/Projects/TanitAD-artifacts/dagland/pytest_sweep_by_file.log
echo "RESUME $(date +%H:%M)" >> $OUT
FILES="$(grep -l v8f_mini_box2_queue.cmd stack/tests/*.py) stack/tests/test_v8f_init_ckpt_o_interface.py stack/tests/test_v8f_dagger_train.py stack/tests/test_v8f_ego_perturb.py stack/tests/test_v8f_dagger.py"
SKIP="stack/tests/test_v8f_cz.py stack/tests/test_v8f_dscale.py stack/tests/test_v8f_flat.py stack/tests/test_v8f_grid.py"
for f in $FILES; do
  case " $SKIP " in *" $f "*) continue;; esac
  a=$(df -m /c | tail -1 | awk '{print $4}'); b=$(du -sm /c/Users/Admin/ofx_pt2 2>/dev/null | cut -f1); free=$(( a + ${b:-0} ))
  [ "$free" -lt 700 ] && { echo "SWEEP-STOP-DISK before $f (C: $free MB)" >> $OUT; exit 3; }
  r=$(CUDA_VISIBLE_DEVICES=-1 timeout 1500 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp=C:/Users/Admin/ofx_pt2 "$f" 2>&1)
  rc=$?
  echo "$f rc=$rc | $(echo "$r" | tail -1)" >> $OUT
  echo "$r" | grep -E "^(FAILED|ERROR) " | sed 's/^/    /' >> $OUT
done
echo "SWEEP-DONE" >> $OUT

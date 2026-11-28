#!/bin/bash
cd F:/TanitAD-scratch/rollhist/build
until [ -f b1/logs/chain.done ]; do sleep 15; done
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
for f in test_v8f_t1_readout test_rollhist_pair_check test_v8f_rollhist; do
  bash ./runpy.sh tree final_$f -m pytest -q -p no:cacheprovider --tb=short tests/$f.py > b1/logs/final_$f.log 2>&1
done
$PY b1/finalize.py > b1/logs/finalize.log 2>&1
rm -rf b1/save
$PY b1/build_report.py > b1/logs/build_report.log 2>&1
# cleanup of own temp data (literal paths), logged
for d in $(ls tmp); do
  s=$(du -sm "F:/TanitAD-scratch/rollhist/build/tmp/$d" | cut -f1)
  rm -rf "F:/TanitAD-scratch/rollhist/build/tmp/$d"
  echo "$(date -u +%FT%TZ) B1 build agent: deleted F:/TanitAD-scratch/rollhist/build/tmp/$d ${s} MB (pytest TMP, nothing needs it)" >> F:/TanitAD-scratch/rollhist/DELETIONS.log
done
echo POST_DONE > b1/logs/post.done

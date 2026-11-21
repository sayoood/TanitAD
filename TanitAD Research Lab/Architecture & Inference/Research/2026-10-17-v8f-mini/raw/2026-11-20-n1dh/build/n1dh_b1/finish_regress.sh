#!/bin/bash
# finish_regress.sh -- after regress_new is ALLDONE: run the failing files on the bcf3ba3f base (3 core files swapped for the mirror's blobs, restored from pd/b and sha-verified), junit diff, REPORT_B1.json, MANIFEST_SHA256.txt, own-temp cleanup.
B=F:/TanitAD-scratch/n1dc/build/n1dh_b1
M=C:/Users/Admin/tanitad-push
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd $B
until grep -q ALLDONE regress_new/_summary.txt 2>/dev/null; do sleep 10; done
FAILS=$(grep -v "rc=0" regress_new/_summary.txt | grep "rc=" | awk '{print $1".py"}' | tr '\n' ' ')
echo "failing files on the patched tree: $FAILS" > logs/finish.log
CORE="stack/tanitad/models/v8f_mini.py stack/scripts/v8f_mini_train.py stack/scripts/strat1_check.py"
if [ -n "$FAILS" ]; then
  for f in $CORE; do git -C $M show bcf3ba3f:$f > $B/tree/$f; done
  mkdir -p regress_base tmp/rb
  bash run_regress.sh $B/tree $B/regress_base $B/tmp/rb $FAILS >> logs/finish.log 2>&1
  for f in $CORE; do cp $B/pd/b/$f $B/tree/$f; done
fi
for f in $CORE; do a=$(sha256sum $B/tree/$f | cut -d' ' -f1); b=$(grep " $f " TREE_FILES_SHA256.txt | cut -d' ' -f1); [ "$a" = "$b" ] && echo "restored $f ok" >> logs/finish.log || echo "RESTORE MISMATCH $f" >> logs/finish.log; done
$PY regress_cmp.py >> logs/finish.log 2>&1
$PY mkpatch_n1dh.py >> logs/finish.log 2>&1
$PY make_report_n1dh.py >> logs/finish.log 2>&1
( sha256sum n1dh.patch TREE_FILES_SHA256.txt HANDOFF_B1.md REPORT_B1.json REGRESSION.json mutation_table_n1dh.json run_mutants_n1dh.py mkpatch_n1dh.py regress_cmp.py finish_regress.sh run_regress.sh runpt.sh envrun.sh klim_synth.py make_report_n1dh.py records/klim/KLIM_synthetic.json patch_scripts/*.py logs/*.log ) > MANIFEST_SHA256.txt 2>&1
for d in tmp/rn tmp/rb tmp/klim; do [ -d "$B/$d" ] && rm -rf "$B/$d" && echo "$(date -u +%FT%TZ) $B/$d (own pytest/klim temp) removed" >> F:/TanitAD-scratch/n1dc/DELETIONS.log; done
echo FINISHED >> logs/finish.log

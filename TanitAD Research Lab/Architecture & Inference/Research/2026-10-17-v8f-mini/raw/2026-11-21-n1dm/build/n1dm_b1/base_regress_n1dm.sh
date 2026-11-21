#!/bin/bash
# base_regress_n1dm.sh -- MM, 2026-11-21. After the N1DM finisher ends (logs/finish.log FINISHED), run the 1f217bf0 BASE on every test file that
# failed on the patched tree (regress_new rc != 0), except files new in the patch, then regress_cmp.py and make_manifest.sh.
# The base tree and its pytest temp live on C: (F: is near its 6 GB floor). Done when logs/base.log ends with BASEDONE.
B=F:/TanitAD-scratch/n1dc/build/n1dm_b1; LOG=$B/logs/base.log; BT=C:/Users/Admin/n1dmbase
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
say start
until grep -q FINISHED $B/logs/finish.log; do sleep 30; done; say "finisher FINISHED"
if [ ! -d $BT/tree/stack ]; then
  mkdir -p $BT/tree && git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive 1f217bf0 stack taniteval | tar -x -C $BT/tree; say "exported 1f217bf0 rc=$?"
fi
fails=$(tr -d '\r' < $B/regress_new/_summary.txt | awk '$2 != "rc=0" && $1 != "ALLDONE" {print $1".py"}')
run=""; for f in $fails; do if [ -f $BT/tree/stack/tests/$f ]; then run="$run $f"; else say "new in the patch, no base: $f"; fi; done
say "base files:$run"
bash $B/run_regress.sh $BT/tree $B/regress_base $BT/rg $run; say "base run done"
cd $B && $PY -I regress_cmp.py >> $LOG 2>&1; say "regress_cmp rc=$?"
bash $B/make_manifest.sh >> $LOG 2>&1; say "manifest rc=$?"
say BASEDONE

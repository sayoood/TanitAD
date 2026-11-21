#!/bin/bash
# lg_after_base.sh -- MM 2026-11-21: after base_regress_n1dm.sh ends (logs/base.log BASEDONE), run test_launch_gate.py whole on the N1DM tree
# (the new argv value `modes` against the gate's profile fingerprints); if it fails, the same file on the 1f217bf0 base tree. Log: logs/lg.log
B=F:/TanitAD-scratch/n1dc/build/n1dm_b1; LOG=$B/logs/lg.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
say start
until grep -q BASEDONE $B/logs/base.log; do sleep 30; done; say "base BASEDONE"
bash $B/run_regress.sh $B/tree $B/regress_lg_new $B/tmp/lg test_launch_gate.py; say "new: $(grep rc= $B/regress_lg_new/_summary.txt)"
if ! grep -q "rc=0" $B/regress_lg_new/_summary.txt; then
  bash $B/run_regress.sh C:/Users/Admin/n1dmbase/tree $B/regress_lg_base C:/Users/Admin/n1dmbase/lg test_launch_gate.py; say "base: $(grep rc= $B/regress_lg_base/_summary.txt)"
fi
say LGDONE

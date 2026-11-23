#!/bin/bash
# make_mutants_marker_ki.sh <scratch dir> -- mutants of rh_ki_marker_watch.sh, each must turn test_ki_marker_watch.sh RED (rc != 0). The control (unmutated copy) must be GREEN.
HERE=$(cd "$(dirname "$0")" && pwd)
W=$1; rm -rf "$W"; mkdir -p "$W"; allred=1
run() {   # name, sed expression
  local n=$1 e=$2; mkdir -p "$W/$n"
  cp $HERE/test_ki_marker_watch.sh "$W/$n/"
  if [ -n "$e" ]; then sed -e "$e" $HERE/rh_ki_marker_watch.sh > "$W/$n/rh_ki_marker_watch.sh"; else cp $HERE/rh_ki_marker_watch.sh "$W/$n/"; fi
  if [ -n "$e" ] && cmp -s $HERE/rh_ki_marker_watch.sh "$W/$n/rh_ki_marker_watch.sh"; then echo "$n: the sed expression changed nothing (mutant test bug)"; allred=0; return; fi
  timeout 120 bash "$W/$n/test_ki_marker_watch.sh" "$W/$n/w" > "$W/$n/out.txt" 2>&1; local rc=$?
  if [ -z "$e" ]; then echo "M0 control rc=$rc ($([ $rc = 0 ] && echo 'GREEN as required' || echo 'NOT GREEN'))"; [ $rc = 0 ] || allred=0
  else echo "$n test_rc=$rc ($([ $rc != 0 ] && echo 'RED as required' || echo 'STILL GREEN: the test missed it')): $(grep FAIL "$W/$n/out.txt" | head -2 | tr '\n' ';')"; [ $rc != 0 ] || allred=0; fi
}
run ctl ""
run mut_clobber 's/( set -C; printf/( printf/'
run mut_clear_any 's/\[ "\$(cat "\$M" 2>\/dev\/null)" = "\$(cat "\$MINE")" \]/true/'
run mut_nobaseline 's/^B=\$(wc -l < \$CL)/B=0/'
run mut_chainend_rc0 's/say "the ROLLHIST-KI chain ended before the lock line"; exit 1/say "the ROLLHIST-KI chain ended before the lock line"; exit 0/'
run mut_nochainproc_rc0 's/clear_mine; say "no rh_ki_chain.sh process"; exit 2/clear_mine; say "no rh_ki_chain.sh process"; exit 0/'
run mut_noclear_at_lock 's/clear_mine; say "the ROLLHIST-KI chain holds the GPU lock"; exit 0/say "the ROLLHIST-KI chain holds the GPU lock"; exit 0/'
echo "ZZ KI-MARKER-MUTANTS $([ $allred = 1 ] && echo PASS || echo FAIL)"
[ $allred = 1 ]

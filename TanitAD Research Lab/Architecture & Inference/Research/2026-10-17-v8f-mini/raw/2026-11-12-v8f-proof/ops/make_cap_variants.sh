#!/usr/bin/env bash
# make_cap_variants.sh : coordination decision D6a. Writes <name>_cap.sh beside each dev-box read script of this fork. A variant differs from its source only in:
#   the Python entry (ops/pycap.sh: hard commit limit per job, one capped job at a time, own gate), the free-memory thresholds of the script's own gate
#   (5.5 GB -> 2.5 GB), and the names of nested read scripts (-> their _cap variants). No reader, no pair list, no path of a read changes.
# The sources are not edited (some are running). An existing variant is not overwritten.
H=/f/TanitAD-scratch/v8proof; cd $H || exit 1
OLDPY='PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe'
NEWPY='PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh'
rc=0
for f in imagplan/read_p5b_k8.sh imagplan/read_k8_four.sh qhead_n1/qhead_read_qf.sh qhead_n1/qhead_read_qfls.sh actlevel/early_read.sh actlevel/early_wait.sh \
         actlevel/wait_actlevel.sh actlevel/wait_alg.sh qhead_n1/ops/wait_qhead_dose_ls.sh qhead_n1/ops/wait_qhead_qfls.sh; do
  o=${f%.sh}_cap.sh
  [ -e "$o" ] && { echo "exists, kept: $o"; continue; }
  sed -e "s#^$OLDPY\$#$NEWPY#" -e 's/5767168/2621440/g' -e 's/-ge 5632 \]/-ge 2560 ]/g' -e 's/5\.5 GB/2.5 GB/g' \
      -e 's#actlevel/early_read\.sh#actlevel/early_read_cap.sh#g' -e 's#qhead_n1/qhead_read_qfls\.sh#qhead_n1/qhead_read_qfls_cap.sh#g' "$f" > "$o"
  nsrc=$(grep -c -- "^$OLDPY\$" "$f"); nnew=$(grep -c -- "^$NEWPY\$" "$o"); left=$(grep -c -E -- 'venvs/tanitad/Scripts/python|5767168|-ge 5632|5\.5 GB' "$o")
  opt=$(grep -c -E -- '"\$PY" -[abd-zA-Z]' "$o")
  bash -n "$o"; syn=$?
  dl=$(diff "$f" "$o" | grep -c '^>')
  ok=OK; { [ "$nsrc" = "$nnew" ] && [ "$left" = 0 ] && [ "$opt" = 0 ] && [ "$syn" = 0 ]; } || { ok=FAIL; rc=1; }
  echo "$ok $o : PY lines $nsrc -> $nnew | old text left $left | unhandled options $opt | syntax rc $syn | changed lines $dl | sha $(sha256sum "$o" | cut -c1-16)"
done
[ $rc = 0 ] && echo "ZZ CAP-VARIANTS DONE" || echo "ZZ CAP-VARIANTS FAIL"

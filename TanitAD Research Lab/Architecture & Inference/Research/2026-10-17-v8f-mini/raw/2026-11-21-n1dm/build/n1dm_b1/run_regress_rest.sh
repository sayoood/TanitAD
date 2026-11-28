#!/bin/bash
# run_regress_rest.sh -- the remaining whole-file regress runs (one pytest at a time, commit-gated) for the files not yet in regress_new/_summary.txt, then regress_cmp.py. Detached; done when regress_new/_summary.txt ends with ALLDONE.
B=F:/TanitAD-scratch/n1dc/build/n1dm_b1
done_list=$(awk '{print $1".py"}' $B/regress_new/_summary.txt | grep -v ALLDONE.py)
rest=$(grep -vxF -f <(echo "$done_list") $B/regress_order.txt | tr -d '\r')
bash $B/run_regress.sh "$B/tree" "$B/regress_new" "$B/tmp/rg" $rest

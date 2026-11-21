#!/usr/bin/env bash
# push_fit.sh -- dev box: push the kit of card ADD99 (CROPREAD, fit and read side) and the wrapped tools to the pod, file by file, with a sha256 compare on both sides.
# Never overwrites a pod file (noclobber): a second run with unchanged files prints OK for each. The lead runs it. It does not print the ssh helper.
set -u
V=/f/TanitAD-scratch/v8proof
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/cropread/kit/fit
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
timeout 60 bash $P "mkdir -p $GP && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
for f in crop_fit.py crop_dump.py read_cropread.py pod_cropread_fit_queue.sh podread_cropread.sh; do push $V/cropread/fit/$f $GP/$f; done
for f in kinclose2.py kinclose3.py kinclose4.py kinclose5.py kinclose6.py; do push $V/kinclose/$f $GP/$f; done      # kinclose6 imports 2 to 5; the closeread tools stay in /workspace/v8proof/closeread/tools
push $V/errpart/make_part_joins.py $GP/make_part_joins.py                                                            # the routine of the three parts of card ADD98 (numpy only)
echo "ZZ PUSH-CROPREAD-FIT bad=$bad"

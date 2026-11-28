#!/usr/bin/env bash
# push_gaprule_b.sh -- dev box: push files of the kit of card ADD95 to the pod (new root gaprule10kb), one by one, with a sha256 compare on both sides. Never overwrites a pod file (noclobber).
# Usage: bash push_gaprule_b.sh <stage>     stage 1: new root and the files for the joins, the wrapper check and the queue     stage 2: the reader, its pod script and the test output of the lead
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/gaprule10kb
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
case "${1:-}" in
  1)
    timeout 60 bash $P "mkdir $GP && mkdir $GP/kit && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
    for f in make_planted_joins_b.py check_wrapper_j.py pod_gaprule10k_queue.sh pod_check_wrapper_jb.sh; do push gaprule/$f $GP/kit/$f; done
    ;;
  2)
    for f in read_gaprule10kb.py podread_gaprule10kb.sh lead_pytest_b.out; do push gaprule/$f $GP/kit/$f; done
    ;;
  *) echo "usage: push_gaprule_b.sh 1|2"; exit 2;;
esac
echo "ZZ PUSH-GAPRULE-B stage ${1:-} bad=$bad"

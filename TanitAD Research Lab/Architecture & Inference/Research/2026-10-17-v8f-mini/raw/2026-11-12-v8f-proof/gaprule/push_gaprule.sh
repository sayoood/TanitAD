#!/usr/bin/env bash
# push_gaprule.sh -- dev box: push files of the kit of card ADD94 to the pod, one by one, with a sha256 compare on both sides. Never overwrites a pod file (noclobber).
# Usage: bash push_gaprule.sh <stage>     stage 1: new root, kit files and the wrapper     stage 2: the read script and the reader modules
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/gaprule10k; B=/workspace/v8proof/integ10k/build
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
    for f in make_planted_joins.py check_wrapper_j.py pod_gaprule10k_queue.sh pod_check_wrapper_j.sh; do push gaprule/$f $GP/kit/$f; done
    push gaprule/v8f_t1_eval_reader_rule_rzgfhqetj.py $B/v8f_t1_eval_reader_rule_rzgfhqetj.py
    ;;
  2)
    for f in read_gaprule10k.py podread_gaprule10k.sh lead_pytest.out; do push gaprule/$f $GP/kit/$f; done
    ;;
  *) echo "usage: push_gaprule.sh 1|2"; exit 2;;
esac
echo "ZZ PUSH-GAPRULE stage ${1:-} bad=$bad"

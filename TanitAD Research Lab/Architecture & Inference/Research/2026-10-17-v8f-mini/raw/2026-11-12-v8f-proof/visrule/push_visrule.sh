#!/usr/bin/env bash
# push_visrule.sh -- dev box: push the kit of card ADD102 to the pod (new root visrule10k), file by file, with a sha256 compare on both sides. Never overwrites a pod file (noclobber).
# The wrapper variant goes to the build folder of the pod (integ10k/build) as a new file; every other file goes to $GP/kit.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/visrule10k
B=/workspace/v8proof/integ10k/build
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
timeout 60 bash $P "mkdir $GP && mkdir $GP/kit && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
push visrule/v8f_t1_eval_reader_rule_rzgfhqetjv.py $B/v8f_t1_eval_reader_rule_rzgfhqetjv.py
for f in make_vis_gapfile.py check_wrapper_v.py pod_check_wrapper_v.sh pod_visrule10k_queue.sh read_visrule10k.py podread_visrule10k.sh lead_pytest.out; do push visrule/$f $GP/kit/$f; done
echo "ZZ PUSH-VISRULE bad=$bad"

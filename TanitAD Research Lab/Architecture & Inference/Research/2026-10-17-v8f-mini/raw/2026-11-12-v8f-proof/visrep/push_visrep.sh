#!/usr/bin/env bash
# push_visrep.sh -- dev box: push the kit of card ADD103 to the pod (new root visrep10k), file by file, with a sha256 compare on both sides. Never overwrites a pod file (noclobber).
# The wrapper variant of card ADD102 is not pushed again (it is on the pod in integ10k/build and its hash is checked by the queue). Every file goes to $GP/kit. Prints no helper script.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/visrep10k
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
[ -s visrep/lead_pytest.out ] || { echo "REFUSED: visrep/lead_pytest.out is missing (run the pytest of the kit first)"; exit 2; }
timeout 60 bash $P "mkdir $GP && mkdir $GP/kit && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
for f in make_rep_gapfile.py pod_visrep10k_queue.sh read_visrep10k.py podread_visrep10k.sh lead_pytest.out; do push visrep/$f $GP/kit/$f; done
echo "ZZ PUSH-VISREP bad=$bad"

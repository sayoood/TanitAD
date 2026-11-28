#!/usr/bin/env bash
# push_fallattr_gap.sh -- dev box: push the gap-file and rollout-queue kit of card ADD110 to the pod (folder fallattr/kit_gap), file by file, with a sha256 compare on both sides. Never overwrites a pod file (noclobber).
# The wrapper variant of card ADD102 is not pushed again (it is on the pod in integ10k/build and its hash is checked by the queue). Prints no pod address and no helper script. The lead runs this script, not the kit author.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
FA=/workspace/v8proof/fallattr
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
[ -s fallattr/gap/lead_pytest.out ] || { echo "REFUSED: fallattr/gap/lead_pytest.out is missing (run the pytest of the kit first)"; exit 2; }
timeout 60 bash $P "mkdir -p $FA && mkdir $FA/kit_gap && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
for f in make_attr_gapfiles.py make_fallattr_queue.py pod_fallattr10k_queue.sh lead_pytest.out; do push fallattr/gap/$f $FA/kit_gap/$f; done
echo "ZZ PUSH-FALLATTR-GAP bad=$bad"

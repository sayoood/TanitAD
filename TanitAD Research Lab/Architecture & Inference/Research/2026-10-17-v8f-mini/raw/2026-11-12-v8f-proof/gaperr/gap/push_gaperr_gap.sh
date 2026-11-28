#!/usr/bin/env bash
# push_gaperr_gap.sh -- dev box: push the gap-file and rollout-queue kit of card ADD111 to the pod (folder gaperr/kit_gap), file by file, with a sha256 compare on both sides. Never overwrites a pod file (noclobber).
# The wrapper variant of card ADD102 is not pushed again (it is on the pod in integ10k/build and its hash is checked by the queue). Prints no pod address and no helper script. The lead runs this script, not the kit author.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GE=/workspace/v8proof/gaperr
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
[ -s gaperr/gap/lead_pytest.out ] || { echo "REFUSED: gaperr/gap/lead_pytest.out is missing (run the pytest of the kit first)"; exit 2; }
timeout 60 bash $P "mkdir -p $GE && mkdir $GE/kit_gap && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
for f in make_gaperr_gapfiles.py make_gaperr_queue.py pod_gaperr10k_queue.sh lead_pytest.out; do push gaperr/gap/$f $GE/kit_gap/$f; done
echo "ZZ PUSH-GAPERR-GAP bad=$bad"

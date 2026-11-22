#!/usr/bin/env bash
# push_cropscale_fit.sh -- dev box: push the fit, gap-file and rollout-queue kit of card ADD106 to the pod (folder cropscale/kit_fit), file by file, with a sha256 compare on both sides. Never overwrites a pod file (noclobber).
# The wrapper variant of card ADD102 is not pushed again (it is on the pod in integ10k/build and its hash is checked by the queue); the fit tools of card ADD99 stay in their pod folder. Prints no helper script.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/cropscale
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
[ -s cropscale/fit/lead_pytest.out ] || { echo "REFUSED: cropscale/fit/lead_pytest.out is missing (run the pytest of the kit first)"; exit 2; }
timeout 60 bash $P "mkdir -p $GP && mkdir $GP/kit_fit && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
for f in make_scale_gapfiles.py pod_cropscale_fit_queue.sh pod_cropscale10k_queue.sh lead_pytest.out; do push cropscale/fit/$f $GP/kit_fit/$f; done
echo "ZZ PUSH-CROPSCALE-FIT bad=$bad"

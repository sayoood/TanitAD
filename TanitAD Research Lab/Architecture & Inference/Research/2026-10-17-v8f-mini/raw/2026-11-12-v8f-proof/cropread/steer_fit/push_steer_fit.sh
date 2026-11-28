#!/usr/bin/env bash
# push_steer_fit.sh -- dev box: push the kit of card ADD100 (STEERREAD, fit and read side) to the pod, file by file, with a sha256 compare on both sides.
# Pushes into /workspace/v8proof/cropread/kit/steer_fit/ only; the kit of card ADD99 in kit/fit/ (and the wrapped tools) is used as it is there.
# Never overwrites a pod file (noclobber): a second run with unchanged files prints OK for each. The lead runs it. It does not print the ssh helper.
set -u
V=/f/TanitAD-scratch/v8proof
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/cropread/kit/steer_fit
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
timeout 60 bash $P "mkdir -p $GP && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
for f in steer_common.py loc_fit.py loc_steer.py crop_fit_s.py crop_dump_s.py read_steerread.py pod_steer_fit_queue.sh podread_steerread.sh; do push $V/cropread/steer_fit/$f $GP/$f; done
echo "ZZ PUSH-STEERREAD-FIT bad=$bad"

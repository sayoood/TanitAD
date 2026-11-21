#!/usr/bin/env bash
# push_croprule.sh -- dev box: push the kit of card ADD101 to the pod (new root croprule10k), file by file, with a sha256 compare on both sides. Never overwrites a pod file (noclobber).
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/croprule10k
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
timeout 60 bash $P "mkdir $GP && mkdir $GP/kit && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
for f in make_crop_joins.py pod_croprule10k_queue.sh read_croprule10k.py podread_croprule10k.sh lead_pytest.out; do push croprule/$f $GP/kit/$f; done
echo "ZZ PUSH-CROPRULE bad=$bad"

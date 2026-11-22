#!/usr/bin/env bash
# push_lookat.sh -- dev box: push the kit of card ADD107 to the pod (folder /workspace/v8proof/lookat/kit), file by file, with a sha256 compare on both sides. Never overwrites a pod file (noclobber). Prints no helper script.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/lookat
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
[ -s lookat/lead_pytest.out ] || { echo "REFUSED: lookat/lead_pytest.out is missing (run the pytest of the kit first)"; exit 2; }
timeout 60 bash $P "mkdir -p $GP && mkdir $GP/kit && echo MK-OK" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r'
for f in locator_fit.py read_lookat.py pod_lookat_queue.sh podread_lookat.sh lead_pytest.out; do push lookat/$f $GP/kit/$f; done
echo "ZZ PUSH-LOOKAT bad=$bad"

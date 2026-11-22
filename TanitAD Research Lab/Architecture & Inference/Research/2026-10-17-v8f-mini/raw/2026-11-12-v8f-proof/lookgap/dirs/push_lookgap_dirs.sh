#!/usr/bin/env bash
# push_lookgap_dirs.sh -- dev box: push the kit of stage A of card ADD108 to the pod (folder /workspace/v8proof/lookgap/kit_dirs), file by file, with a sha256 compare on both sides. Never overwrites a pod file (noclobber). Prints no helper script.
# locator_fit.py is not pushed: the queue uses the one in /workspace/v8proof/lookat/kit unchanged.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/lookgap
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
[ -s lookgap/dirs/lead_pytest.out ] || { echo "REFUSED: lookgap/dirs/lead_pytest.out is missing (run the pytest of the kit first)"; exit 2; }
timeout 60 bash $P "mkdir -p $GP && mkdir $GP/kit_dirs && echo MK-OK" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r'
for f in loc_dirs.py pod_lookgap_dirs_queue.sh lead_pytest.out; do push lookgap/dirs/$f $GP/kit_dirs/$f; done
echo "ZZ PUSH-LOOKGAP-DIRS bad=$bad"

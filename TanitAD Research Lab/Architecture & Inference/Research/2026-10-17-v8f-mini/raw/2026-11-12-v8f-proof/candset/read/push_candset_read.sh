#!/usr/bin/env bash
# push_candset_read.sh -- dev box: push the reader kit of card ADD113 to the pod (folder candset/kit_read), file by file, with a sha256 compare on both sides. Never overwrites a pod file (noclobber).
# The reader imports the model reader of card ADD106 from cropscale/kit_read on the pod (pushed by its own script). The lead's test outputs are not pushed by this script: the lead writes them to
# candset/lead_tests on the pod (control d4). Prints no helper script.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/candset
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
timeout 60 bash $P "mkdir -p $GP && mkdir $GP/kit_read && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
for f in read_candset10k.py podread_candset10k.sh; do push candset/read/$f $GP/kit_read/$f; done
echo "ZZ PUSH-CANDSET-READ bad=$bad"

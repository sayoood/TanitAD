#!/usr/bin/env bash
# push_lookgap_read.sh -- dev box: push the read kit of card ADD108 to the pod (folder /workspace/v8proof/lookgap/kit_read), file by file, with a sha256 compare on both sides.
# Never overwrites a pod file (noclobber). Prints no helper script.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/lookgap
SRC=lookgap/read
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
[ -s $SRC/lead_pytest.out ] || { echo "REFUSED: $SRC/lead_pytest.out is missing (the lead runs the pytest of the kit first)"; exit 2; }
timeout 60 bash $P "mkdir -p $GP && mkdir $GP/kit_read && echo MK-OK" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r'
for f in read_lookgap.py podread_lookgap.sh lead_pytest.out; do push $SRC/$f $GP/kit_read/$f; done
echo "ZZ PUSH-LOOKGAP-READ bad=$bad"

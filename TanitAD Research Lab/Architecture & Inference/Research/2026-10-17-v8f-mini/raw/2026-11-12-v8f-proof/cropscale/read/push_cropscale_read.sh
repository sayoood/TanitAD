#!/usr/bin/env bash
# push_cropscale_read.sh -- dev box: push the reader kit of card ADD106 to the pod (folder cropscale/kit_read), file by file, with a sha256 compare on both sides. Never overwrites a pod file (noclobber).
# The reader imports make_scale_gapfiles.py from kit_fit and ctl_g0g1.py from kit_enc (pushed by their own scripts); the engine modules are on the pod in the folders of the earlier readers. Prints no helper script.
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
[ -s cropscale/read/lead_pytest.out ] || { echo "REFUSED: cropscale/read/lead_pytest.out is missing (run the pytest of the kit first)"; exit 2; }
timeout 60 bash $P "mkdir -p $GP && mkdir $GP/kit_read && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
for f in read_cropscale10k.py podread_cropscale10k.sh lead_pytest.out; do push cropscale/read/$f $GP/kit_read/$f; done
echo "ZZ PUSH-CROPSCALE-READ bad=$bad"

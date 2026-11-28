#!/usr/bin/env bash
# push_gaperr_read.sh -- dev box: push the reader kit of card ADD111 to the pod (folder gaperr/kit_read), file by file, with a sha256 compare on both sides. Never overwrites a pod file (noclobber).
# The reader imports the model reader of card ADD106 from cropscale/kit_read on the pod (pushed by its own script). Prints no helper script.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/gaperr
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
[ -s gaperr/read/lead_pytest.out ] || { echo "REFUSED: gaperr/read/lead_pytest.out is missing (run the pytest of the kit first)"; exit 2; }
timeout 60 bash $P "mkdir -p $GP && mkdir $GP/kit_read && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
for f in read_gaperr10k.py podread_gaperr10k.sh lead_pytest.out; do push gaperr/read/$f $GP/kit_read/$f; done
echo "ZZ PUSH-GAPERR-READ bad=$bad"

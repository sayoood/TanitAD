#!/usr/bin/env bash
# push_cropscale_enc.sh -- dev box: push the ENCODE kit of card ADD106 to the pod (/workspace/v8proof/cropscale/kit_enc), file by file, with a sha256 compare on both sides.
# Never overwrites a pod file (noclobber: a second push of a changed file reports MISMATCH and the lead decides). The pilot's kit on the pod is not touched.
# Refuses without cropscale/enc/lead_pytest.out (the lead's pytest run of the kit). Prints no helper script, no pod address, no token.
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
[ -s cropscale/enc/lead_pytest.out ] || { echo "REFUSED: cropscale/enc/lead_pytest.out is missing (run the pytest of the kit first)"; exit 2; }
timeout 60 bash $P "mkdir $GP/kit_enc && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
for f in make_large_list.py gen_window_encode_d.py ctl_g0g1.py pod_cropscale_enc_queue.sh lead_pytest.out; do push cropscale/enc/$f $GP/kit_enc/$f; done
echo "ZZ PUSH-CROPSCALE-ENC bad=$bad"
exit $bad

#!/usr/bin/env bash
# push_lookgap_fit.sh -- dev box: push the FIT kit of card ADD108 to the pod (/workspace/v8proof/lookgap/kit_fit), file by file, with a sha256 compare on both sides.
# Never overwrites a pod file (noclobber: a second push of a changed file reports MISMATCH and the lead decides). The pilot's kit on the pod is not touched.
# Refuses without lookgap/fit/lead_pytest.out (the lead's pytest run of the kit). Prints no helper script, no pod address, no token.
set -u
cd /f/TanitAD-scratch/v8proof || exit 2
P=/c/Users/Admin/v8pod/pssh.sh
GP=/workspace/v8proof/lookgap
bad=0
push() { # local file, pod path
  tr -d '\r' < "$1" | timeout 120 bash $P "set -C; cat > $2" 2>/dev/null
  l=$(tr -d '\r' < "$1" | sha256sum | cut -c1-16)
  r=$(timeout 60 bash $P "sha256sum $2 | cut -c1-16" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  if [ "$l" = "$r" ]; then echo "OK $l $2"; else echo "MISMATCH $l $r $2"; bad=1; fi
}
[ -s lookgap/fit/lead_pytest.out ] || { echo "REFUSED: lookgap/fit/lead_pytest.out is missing (run the pytest of the kit first)"; exit 2; }
timeout 60 bash $P "mkdir -p $GP/kit_fit && echo MK-OK" 2>/dev/null | sed '/hf_/d' | tr -d '\r'
for f in pod_lookgap_fit_queue.sh cross_dump.py lead_pytest.out; do push lookgap/fit/$f $GP/kit_fit/$f; done
echo "ZZ PUSH-LOOKGAP-FIT bad=$bad"
exit $bad

#!/bin/bash
# mm_wait_add125.sh -- MM 2026-11-23. Polls the ADD125 chain log on the LoRA pod every 120 s (cap 4 h). On "ZZ ADD125 DONE ZZ" it pulls
# /root/dscale_lead/out_eqp_nockpt.tar.gz (read only, pcat) to F:/TanitAD-scratch/v8proof/dscale_lead/add125/back/ and checks its sha256 against the pod's.
# On STOP / FAIL it prints the chain's tail. Exit 0 = DONE and pulled; 1 = STOP/FAIL; 2 = pull mismatch; 3 = timeout.
B=F:/TanitAD-scratch/v8proof/dscale_lead/add125/back
q() { timeout 90 bash -c 'source C:/Users/Admin/lorapod/podenv.sh; pssh "$0"' "$1" < /dev/null 2>/dev/null; }
for i in $(seq 1 150); do
  L=$(q "tail -n 12 /root/dscale_lead/mm_add125_chain_r2.log")
  if printf '%s\n' "$L" | grep -q "ZZ ADD125 DONE ZZ"; then
    mkdir -p "$B"
    PS=$(q "cut -d' ' -f1 /root/dscale_lead/out_eqp_nockpt.tar.gz.sha256" | tr -cd 'a-f0-9')
    timeout 900 bash /c/Users/Admin/lorapod/pcat.sh /root/dscale_lead/out_eqp_nockpt.tar.gz "$B/out_eqp_nockpt.tar.gz" < /dev/null > /dev/null 2>&1
    LS=$(sha256sum "$B/out_eqp_nockpt.tar.gz" | cut -d' ' -f1)
    if [ "$PS" = "$LS" ] && [ ${#LS} = 64 ]; then echo "$LS  out_eqp_nockpt.tar.gz" > "$B/out_eqp_nockpt.tar.gz.sha256"; echo "ADD125 DONE, pulled $LS"; printf '%s\n' "$L" | tail -n 6; exit 0; fi
    echo "ADD125 DONE but PULL MISMATCH pod [$PS] local [$LS]"; exit 2
  fi
  if printf '%s\n' "$L" | grep -qE "ZZ ADD125 (STOP|FAIL)"; then echo "ADD125 STOP/FAIL"; printf '%s\n' "$L"; exit 1; fi
  sleep 120
done
echo "ADD125 WAIT TIMEOUT"; exit 3

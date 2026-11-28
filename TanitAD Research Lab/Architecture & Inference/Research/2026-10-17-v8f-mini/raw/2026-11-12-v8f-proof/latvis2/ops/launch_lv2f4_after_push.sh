#!/usr/bin/env bash
# launch_lv2f4_after_push.sh <push output file> : dev box. Note PREREG_V8F_PROOF_ADD38_N4. Waits (script polling, bounded 40 min) for the upload of the four
# banked rollouts and of pod_lv2_forecast4.sh to end, checks the four hash lines and the script hash, starts the pod CPU job (no GPU job, no training;
# the pod script refuses below 24 GB of free container memory) and then runs the waiter wait_lv2f4.sh.
O=${1:?push output file}
P=/c/Users/Admin/v8pod/pssh.sh
H=/f/TanitAD-scratch/v8proof/latvis2
k=0
until grep -q "^script local" "$O" 2>/dev/null; do k=$((k+1)); [ $k -gt 120 ] && { echo "ZZ LV2F4 NOT STARTED: the upload did not end in 40 min"; exit 1; }; sleep 20; done
n=$(grep -c "^pushed LATENTS_CTX_.*_R3g.npz ok" "$O" | tr -d '\r')
l=$(grep "^script local" "$O" | awk '{print $3}'); r=$(grep "^script local" "$O" | awk '{print $5}')
echo "[$(date -u +%H:%M:%SZ)] upload ended: $n of 4 rollouts ok; script local $l remote $r"
[ "$n" = 4 ] && [ -n "$l" ] && [ "$l" = "$r" ] || { echo "ZZ LV2F4 NOT STARTED: upload incomplete or hash mismatch"; exit 1; }
timeout 60 bash "$P" 'cd /workspace/v8proof/runs/lv2_z || exit 1; [ -e lv2f4_all.log ] && { echo "POD-NOT-RUN: lv2f4_all.log exists"; exit 0; }; nohup setsid bash pod_lv2_forecast4.sh > lv2f4_all.log 2>&1 < /dev/null & sleep 5; date -u +%H:%M:%SZ; head -n 3 lv2f4_all.log | cut -c1-200' 2>/dev/null | sed '/hf_/d'
bash $H/ops/wait_lv2f4.sh

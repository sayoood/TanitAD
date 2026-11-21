#!/usr/bin/env bash
# wake_v8floraar2.sh <driver pid>: exits when the LoRA AR-2 chain writes ZZV8FMINI-DONE / -STOP, its driver dies, or its guard stops. Script polling, no model turns.
L=/c/Users/Admin/box2_ship/logs; O=$L/chain_v8f_lora_ar2.out; P=${1:?driver pid}
for i in $(seq 1 600); do
  grep -aq '^ZZV8FMINI-STOP\|^ZZV8FMINI-DONE' $O 2>/dev/null && { tail -15 $O; exit 0; }
  grep -aq ZZGUARD-STOP $L/v8floraar2_guard.out 2>/dev/null && { echo "GUARD: $(grep -a ZZGUARD-STOP $L/v8floraar2_guard.out | tail -1)"; tail -8 $O; exit 4; }
  tasklist //FI "PID eq $P" //NH 2>/dev/null | grep -qi bash || { sleep 20; grep -aq '^ZZV8FMINI-STOP\|^ZZV8FMINI-DONE' $O && { tail -15 $O; exit 0; }; echo "DRIVER GONE (pid $P) with no marker"; tail -8 $O; exit 3; }
  sleep 60
done
echo "WAKER TIMEOUT"; tail -5 $O

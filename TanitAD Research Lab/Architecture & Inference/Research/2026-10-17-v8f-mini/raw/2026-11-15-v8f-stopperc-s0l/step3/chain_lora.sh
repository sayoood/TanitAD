#!/bin/bash
# Pod-side (MM 2026-11-15): after the folded run ends, the trained-path cache: lora-zero control (3 clips, must be bitwise equal to cache_pod/eval), then lora (141 clips).
# Log /root/logs/s0l_lora_chain.log; ends with ZZ LORACHAIN DONE ZZ or ZZ LORACHAIN STOP ZZ <why>.
. /root/ops/lorapod2_env.sh; env_lora; cd /root
L=/root/logs/s0l_lora_chain.log
say() { echo "[chain $(date -u +%FT%TZ)] $*" >> $L; }
for i in $(seq 1 60); do grep -q "ZZ LMERGE MERGED END" /root/logs/s0l_lmerge.log && break; sleep 10; done
grep -q "ZZ LMERGE MERGED END" /root/logs/s0l_lmerge.log || { say "ZZ LORACHAIN STOP ZZ the folded run did not end"; exit 2; }
say "folded run ended: $(grep 'ZZ LMERGE MERGED END' /root/logs/s0l_lmerge.log)"
C=$(head -3 /root/ops/clips141.txt | tr -d '\r' | tr '\n' ' ')
$PY /root/ops/s0l/cache_lmerged.py --mode lora-zero --out /root/data/s0l_lorazero --only $C >> $L 2>&1
rc=$?
grep -q "c3 bitwise equal to cache_pod/eval on all 3 clips: True" $L || { say "ZZ LORACHAIN STOP ZZ lora-zero control not bitwise equal (rc $rc)"; exit 3; }
say "lora-zero control PASS (rc $rc)"
$PY /root/ops/s0l/cache_lmerged.py --mode lora --out /root/data/s0l_cache_L_s0_lora >> $L 2>&1
rc=$?
say "lora run rc $rc"
[ $rc -eq 0 ] && say "ZZ LORACHAIN DONE ZZ" || say "ZZ LORACHAIN STOP ZZ lora run rc $rc"

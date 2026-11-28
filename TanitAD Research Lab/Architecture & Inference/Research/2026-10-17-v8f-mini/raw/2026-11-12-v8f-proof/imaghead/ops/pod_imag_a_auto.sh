# pod_imag_a_auto.sh : runs ON THE POD (nohup). Card PREREG_V8F_PROOF_ADD36. Waits for a free GPU slot, dry-gates RL_s0 (selfcheck only, trains nothing) and,
# only if the gate passes, starts chain imag_a (RL_s0, RL_s1; from scratch, 19,164 steps each) behind chain motion_e (one arm with a rollout loss at a time).
set -u
W=/workspace/v8proof; Z=$W/runs/imag_z; LOG=$Z/imag_auto.log; mkdir -p $Z; cd $W/chain2 || exit 1
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> $LOG; }
[ -e $W/chains/imag_a ] && { say 'chain imag_a exists, nothing done'; exit 0; }
[ -f arms_imag_a.txt ] || { say 'arms_imag_a.txt missing'; echo "no arms file ($(ts))" > $Z/IMAG_NOT_STARTED.txt; exit 1; }
say 'auto start: waits for a free GPU slot'
k=0; while :; do n1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); sleep 45; n2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "$n1" -lt 3 ] && [ "$n2" -lt 3 ] && break; k=$((k+1)); [ $k -gt 640 ] && { say 'no free GPU slot in 8 h'; echo "no slot ($(ts))" > $Z/IMAG_NOT_STARTED.txt; exit 1; }; done
a=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); m=$(cat /sys/fs/cgroup/memory.max); [ $((m - a)) -gt 20000000000 ] || { say 'less than 20 GB of the memory limit free'; echo "memory ($(ts))" > $Z/IMAG_NOT_STARTED.txt; exit 1; }
say "slot free (GPU jobs $n2), memory free $(( (m - a) / 1000000000 )) GB: dry gate RL_s0"
bash dry_gate.sh arms_imag_a.txt RL_s0 PREREG_V8F_PROOF_ADD36 > $W/chains/imag_dry_RL_s0.out 2>&1; rc=$?
say "dry gate RL_s0 rc $rc: $(tail -n 1 $W/chains/imag_dry_RL_s0.out | cut -c1-200)"
[ $rc = 0 ] && grep -q "ZZ DRY RL_s0 PASS" $W/chains/imag_dry_RL_s0.out || { echo "dry gate RL_s0 did not pass ($(ts))" > $Z/IMAG_NOT_STARTED.txt; exit 1; }
/workspace/venv/bin/python - >> $LOG 2>&1 <<'PYEOF'
import json
s = json.load(open('/workspace/v8proof/runs/motion_dry/RL_s0/out/selfcheck.json')); lv = s.get('levers', {}); live = s.get('live') or {}
print('SELFCHECK RL_s0 dvb', s.get('declared_vs_built'), 'live', live.get('pass'), 'steps_total', s.get('steps_total'))
print('   levers', {k: lv.get(k) for k in ('wm_l1_residual', 'wm_l1_roll_k', 'w_wm_roll', 'wm_l1_diag', 'lead_sup', 'w_lead_sup', 'lead_hist', 'lead_sup_detach', 'wm_l1_pred', 'wm_state')})
print('   live terms', {k: v for k, v in live.items() if isinstance(v, (int, float)) and ('roll' in k or 'lead' in k or k in ('total', 'wm'))})
PYEOF
C2_CARD=PREREG_V8F_PROOF_ADD36 C2_WAIT_FOR=$W/chains/motion_e/motion_e_DONE.txt C2_IDLE_MIB=30000 nohup setsid bash chain_arms.sh imag_a arms_imag_a.txt > $W/chains/imag_a_launch.out 2>&1 < /dev/null &
sleep 40; say "chain imag_a started behind motion_e: chain processes $(pgrep -f '[c]hain_arms.sh imag_a' | wc -l); $(tail -n 1 $W/chains/imag_a/chain.log 2>/dev/null | cut -c1-140)"
echo "started $(ts)" > $Z/IMAG_STARTED.txt

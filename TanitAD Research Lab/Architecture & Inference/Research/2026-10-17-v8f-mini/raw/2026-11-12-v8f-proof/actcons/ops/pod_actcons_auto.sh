# pod_actcons_auto.sh : runs ON THE POD (nohup). Card PREREG_V8F_PROOF_ADD37 with note N1. Waits for a free GPU slot and 28 GB of the memory limit free, dry-gates
# LA_s0 (selfcheck only, trains nothing) and, only if the gate passes and the probe of the term passes, starts chain actcons_a (LA_s0, LA_s1, then the dose arms
# LA01_s0, LA01_s1; one arm at a time on one slot). Never a fourth GPU job.
set -u
W=/workspace/v8proof; Z=$W/runs/actcons_z; LOG=$Z/actcons_auto.log; mkdir -p $Z; cd $W/chain2 || exit 1
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> $LOG; }
stopf() { say "$1"; echo "$1 ($(ts))" > $Z/ACTCONS_NOT_STARTED.txt; exit 1; }
[ -e $W/chains/actcons_a ] && { say 'chain actcons_a exists, nothing done'; exit 0; }
[ -f arms_actcons_a.txt ] || stopf 'arms file missing'
slot() { local k=0 n1 n2 a m; while :; do n1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); sleep 45; n2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    a=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); m=$(cat /sys/fs/cgroup/memory.max)
    [ "$n1" -lt 3 ] && [ "$n2" -lt 3 ] && [ $((m - a)) -gt 28000000000 ] && { say "slot free: GPU jobs $n2, memory free $(( (m - a) / 1000000000 )) GB"; return 0; }
    k=$((k+1)); [ $k -gt 640 ] && return 1; done; }
say 'auto start: waits for a free GPU slot and 28 GB of memory'
slot || stopf 'no free GPU slot with 28 GB of memory in 8 h'
bash dry_gate.sh arms_actcons_a.txt LA_s0 PREREG_V8F_PROOF_ADD37 > $W/chains/actcons_dry_LA_s0.out 2>&1; rc=$?
say "dry gate LA_s0 rc $rc: $(tail -n 1 $W/chains/actcons_dry_LA_s0.out | cut -c1-200)"
[ $rc = 0 ] && grep -q "ZZ DRY LA_s0 PASS" $W/chains/actcons_dry_LA_s0.out || stopf 'dry gate LA_s0 did not pass'
/workspace/venv/bin/python - >> $LOG 2>&1 <<'PYEOF'
import json, sys
s = json.load(open('/workspace/v8proof/runs/motion_dry/LA_s0/out/selfcheck.json')); lv = s.get('levers', {}); p = s.get('act_cons_probe') or {}
print('SELFCHECK LA_s0 dvb', s.get('declared_vs_built'), 'live', (s.get('live') or {}).get('pass'), 'steps_total', s.get('steps_total'), 'w_act_cons', lv.get('w_act_cons'), 'act_cons_max', lv.get('act_cons_max'), 'lead_sup', lv.get('lead_sup'))
print('   act_cons probe', json.dumps(p)[:900])
open('/workspace/v8proof/runs/actcons_z/probe_pass.txt', 'w').write('PASS' if p.get('pass') is True else 'FAIL')
PYEOF
[ "$(cat $Z/probe_pass.txt 2>/dev/null)" = PASS ] || stopf 'the probe of the term did not pass in the selfcheck of LA_s0'
say 'dry gate and probe passed: waits for the end of chain leadcond_a or leadcond_b (a slot that stays free; a gap between two arms of a chain is not one)'
k=0; while :; do for c in leadcond_a leadcond_b; do [ -f $W/chains/$c/${c}_DONE.txt ] || [ -f $W/chains/$c/${c}_STOP.txt ] && break 2; done; k=$((k+1)); [ $k -gt 480 ] && stopf 'neither lead-input chain ended in 8 h'; sleep 60; done
say "a lead-input chain ended: $(ls $W/chains/leadcond_a/*_DONE.txt $W/chains/leadcond_b/*_DONE.txt $W/chains/leadcond_a/*_STOP.txt $W/chains/leadcond_b/*_STOP.txt 2>/dev/null | sed 's|.*/||' | tr '
' ' ')"
slot || stopf 'no free GPU slot after the end of a lead-input chain'
C2_CARD=PREREG_V8F_PROOF_ADD37 C2_WAIT_FOR=$W/runs/leadcond_z/Q3B_DONE.txt C2_IDLE_MIB=30000 nohup setsid bash chain_arms_ft6.sh actcons_a arms_actcons_a.txt > $W/chains/actcons_a_launch.out 2>&1 < /dev/null &
sleep 40; say "chain started: actcons_a processes $(pgrep -f '[c]hain_arms_ft6.sh actcons_a' | wc -l); $(tail -n 1 $W/chains/actcons_a/chain.log 2>/dev/null | cut -c1-140)"
echo "started $(ts)" > $Z/ACTCONS_STARTED.txt

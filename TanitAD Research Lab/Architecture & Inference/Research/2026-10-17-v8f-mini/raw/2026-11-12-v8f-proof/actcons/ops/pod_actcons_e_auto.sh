# pod_actcons_e_auto.sh : runs ON THE POD (nohup). Card PREREG_V8F_PROOF_ADD37, note N4 (the low weight on RL bases of new seeds). Form of pod_actcons_c_auto.sh.
# Writes the arms files (imag_b: RL_s2, RL_s3 = the line of RL_s0 with the new seed; actcons_e<k>: RLFT_s<k>, RLAm_s<k> = the lines of arms_actcons_c0.txt with the new
# base checkpoint and seed). Then: waits until the launcher of chain actcons_c1 has started its chain (or has written its not-started marker), waits for a free GPU slot
# with 28 GB of the memory limit free, dry-gates RL_s2 (selfcheck only, trains nothing) and starts chain imag_b (one arm at a time). Per new base k: waits for ARM DONE
# RL_s<k>, waits until the dumps of the last arm of chain actcons_c1 exist (the read of record of note N3 needs 24 GB of free memory for its dumps; bounded wait), waits
# for a slot, dry-gates RLFT_s<k> and RLAm_s<k>, checks the selfchecks (weight, start checkpoint), starts chain actcons_e<k>. Never a fourth GPU job. Fail closed.
set -u
W=/workspace/v8proof; Z=$W/runs/actcons_z; LOG=$Z/actcons_e_auto.log; mkdir -p $Z; cd $W/chain2 || exit 1
CARD=PREREG_V8F_PROOF_ADD37
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> $LOG; }
stopf() { say "$1"; echo "$1 ($(ts))" > $Z/ACTCONS_E_NOT_STARTED.txt; exit 1; }
[ -e $W/chains/imag_b ] && { say 'chain imag_b exists, nothing done'; exit 0; }
[ -f arms_imag_a.txt ] && [ -f arms_actcons_c0.txt ] || stopf 'arms_imag_a.txt or arms_actcons_c0.txt missing'
A=arms_imag_b.txt
if [ ! -e $A ]; then
  { echo "# RL bases of new seeds ($CARD, note N4): RL_s2 and RL_s3 = the line of RL_s0 of arms_imag_a.txt with seed 2 and seed 3."
    grep '^PIN|' arms_imag_a.txt
    for k in 2 3; do grep '^ARM|RL_s0|' arms_imag_a.txt | awk -F'|' -v OFS='|' -v k=$k '{$2="RL_s" k; $5=k; print}'; done; } > $A
  say "arms file $A written: PIN $(grep -c '^PIN|' $A), ARM $(grep -c '^ARM|' $A), sha256 $(sha256sum $A | cut -c1-64)"
fi
[ "$(grep -c '^ARM|' $A)" = 2 ] && [ "$(grep -c '^PIN|' $A)" = "$(grep -c '^PIN|' arms_imag_a.txt)" ] || stopf "arms file $A is not two arms with the pins of arms_imag_a.txt"
[ "$(grep '^ARM|RL_s2|' $A | cut -d'|' -f3-4,6-)" = "$(grep '^ARM|RL_s0|' arms_imag_a.txt | cut -d'|' -f3-4,6-)" ] || stopf 'the line of RL_s2 differs from RL_s0 in more than the name and the seed'
[ "$(grep '^ARM|RL_s3|' $A | cut -d'|' -f5)" = 3 ] || stopf 'RL_s3 does not carry seed 3'
for k in 2 3; do
  A=arms_actcons_e$k.txt
  [ -e $A ] && continue
  { echo "# The low weight on a new base ($CARD, note N4), base $k: control RLFT, arm RLAm (weight 0.0001); the lines of arms_actcons_c0.txt with the checkpoint of RL_s$k of chain imag_b."
    grep '^PIN|' arms_actcons_c0.txt
    for a in RLFT RLAm; do grep "^ARM|${a}_s0|" arms_actcons_c0.txt | awk -F'|' -v OFS='|' -v k=$k -v a=$a '{$2=a "_s" k; $5=k; gsub("/runs/imag_a/RL_s0/", "/runs/imag_b/RL_s" k "/", $8); print}'; done; } > $A
  say "arms file $A written: PIN $(grep -c '^PIN|' $A), ARM $(grep -c '^ARM|' $A), sha256 $(sha256sum $A | cut -c1-64)"
  [ "$(grep -c '^ARM|' $A)" = 2 ] && [ "$(grep -c "runs/imag_b/RL_s$k/ckpt_last.pt" $A)" = 2 ] && [ "$(grep -c 'runs/imag_a/' $A)" = 0 ] && [ "$(grep "^ARM|RLAm_s$k|" $A | grep -c -- '--w-act-cons 0.0001')" = 1 ] && [ "$(grep "^ARM|RLFT_s$k|" $A | grep -c -- '--w-act-cons')" = 0 ] || stopf "arms file $A is not the two lines this script expects"
done
slot() { local k=0 n1 n2 a m; while :; do n1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); sleep 45; n2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    a=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); m=$(cat /sys/fs/cgroup/memory.max)
    [ "$n1" -lt 3 ] && [ "$n2" -lt 3 ] && [ $((m - a)) -gt 28000000000 ] && { say "slot free: GPU jobs $n2, memory free $(( (m - a) / 1000000000 )) GB"; return 0; }
    k=$((k+1)); [ $k -gt 640 ] && return 1; done; }
selfcheck() { # selfcheck <tag> <arm>... : prints the selfcheck of each arm; PASS file only if every arm built what its name says
  local tag=$1; shift
  ARMS_="$*" TAG_=$tag /workspace/venv/bin/python - >> $LOG 2>&1 <<'PYEOF'
import json, os
ok = True
for a in os.environ["ARMS_"].split():
    s = json.load(open('/workspace/v8proof/runs/motion_dry/%s/out/selfcheck.json' % a)); lv = s.get('levers', {}); p = s.get('act_cons_probe') or {}
    k = a.rsplit('_s', 1)[1]
    init = json.dumps(lv.get('init_ckpt'))
    print('SELFCHECK', a, 'dvb', s.get('declared_vs_built'), 'live', (s.get('live') or {}).get('pass'), 'steps_total', s.get('steps_total'), 'w_act_cons', lv.get('w_act_cons'), 'roll_k', lv.get('wm_l1_roll_k'),
          'residual', lv.get('wm_l1_residual'), 'init', init[:200])
    if a.startswith('RLAm_'):
        print('   act_cons probe', json.dumps(p)[:700])
        if abs(float(lv.get('w_act_cons') or 0.0) - 0.0001) > 1e-12: ok = False
    if a.startswith('RLFT_') and float(lv.get('w_act_cons') or 0.0) != 0.0: ok = False
    if a.startswith(('RLAm_', 'RLFT_')) and ('imag_b/RL_s%s/' % k) not in init: ok = False
    if a.startswith('RL_') and (lv.get('wm_l1_roll_k') not in (10, '10') or str(lv.get('wm_l1_residual')) != 'on'): ok = False
open('/workspace/v8proof/runs/actcons_z/selfcheck_%s_pass.txt' % os.environ["TAG_"], 'w').write('PASS' if ok else 'FAIL')
PYEOF
  [ "$(cat $Z/selfcheck_${tag}_pass.txt 2>/dev/null)" = PASS ]
}
dry() { local a rc; for a in "$@"; do
    bash dry_gate.sh "$ARMSF" $a $CARD > $W/chains/actcons_e_dry_$a.out 2>&1; rc=$?
    say "dry gate $a rc $rc: $(tail -n 1 $W/chains/actcons_e_dry_$a.out | cut -c1-220)"
    [ $rc = 0 ] && grep -q "ZZ DRY $a PASS" $W/chains/actcons_e_dry_$a.out || stopf "dry gate $a did not pass"
  done; }
say 'auto start (card ADD37, note N4): waits for the launcher of chain actcons_c1'
j=0; while [ ! -f $Z/ACTCONS_C1_STARTED.txt ] && [ ! -f $Z/ACTCONS_C_NOT_STARTED.txt ]; do j=$((j+1)); [ $j -gt 600 ] && stopf 'chain actcons_c1 neither started nor refused in 10 h'; sleep 60; done
sleep 180
slot || stopf 'imag_b: no free GPU slot with 28 GB of memory in 8 h'
ARMSF=arms_imag_b.txt; dry RL_s2
selfcheck imag_b RL_s2 || stopf 'imag_b: the selfcheck of RL_s2 is not the one of an RL arm'
slot || stopf 'imag_b: no free GPU slot after the dry gate'
C2_CARD=$CARD C2_WAIT_FOR=$W/chains/imag_a/chain.log C2_IDLE_MIB=30000 nohup setsid bash chain_arms.sh imag_b arms_imag_b.txt > $W/chains/imag_b_launch.out 2>&1 < /dev/null &
sleep 40; say "chain started: imag_b processes $(pgrep -f '[c]hain_arms.sh imag_b' | wc -l); $(tail -n 1 $W/chains/imag_b/chain.log 2>/dev/null | cut -c1-140)"
echo "started $(ts)" > $Z/IMAG_B_STARTED.txt
for k in 2 3; do
  say "base $k: waits for ARM DONE RL_s$k in chain imag_b"
  j=0; while ! grep -aq "ARM DONE RL_s$k" $W/chains/imag_b/chain.log 2>/dev/null; do
    [ -f $W/chains/imag_b/imag_b_STOP.txt ] && stopf "chain imag_b stopped before RL_s$k was done"
    j=$((j+1)); [ $j -gt 900 ] && stopf "RL_s$k not done in 15 h"; sleep 60
  done
  say "base $k: RL_s$k is done (checkpoint sha256 $(sha256sum $W/runs/imag_b/RL_s$k/ckpt_last.pt | cut -c1-16))"
  # the read of record of note N3 (dev-box waiter) dumps the arms of chain actcons_c1 on the pod CPU and needs 24 GB of free memory: wait for its last dump (at most 4 h
  # after chain actcons_c1 has ended; if that chain did not start or stopped, do not wait)
  j=0; t=0; while [ ! -f $W/runs/imag_z/dump/DUMP_RLFTb_s1.npz ] && [ -f $Z/ACTCONS_C1_STARTED.txt ] && [ ! -f $W/chains/actcons_c1/actcons_c1_STOP.txt ]; do
    t=$((t+1)); [ $t = 1 ] && say "base $k: waits for the dumps of the read of record of chain actcons_c1"
    [ -f $W/chains/actcons_c1/actcons_c1_DONE.txt ] && j=$((j+1))
    [ $j -gt 240 ] && { say "base $k: no dump of RLFTb_s1 four hours after chain actcons_c1 ended, going on"; break; }
    [ $t -gt 900 ] && { say "base $k: chain actcons_c1 neither done nor stopped after 15 h of waiting, going on"; break; }; sleep 60
  done
  slot || stopf "base $k: no free GPU slot with 28 GB of memory in 8 h"
  ARMSF=arms_actcons_e$k.txt; dry RLFT_s$k RLAm_s$k
  selfcheck e$k RLFT_s$k RLAm_s$k || stopf "base $k: a selfcheck does not show the weight or the start checkpoint its arm names"
  slot || stopf "base $k: no free GPU slot after the dry gates"
  C2_CARD=$CARD C2_WAIT_FOR=$W/chains/imag_b/chain.log C2_IDLE_MIB=30000 nohup setsid bash chain_arms_ft6.sh actcons_e$k arms_actcons_e$k.txt > $W/chains/actcons_e${k}_launch.out 2>&1 < /dev/null &
  sleep 40; say "chain started: actcons_e$k processes $(pgrep -f "[c]hain_arms_ft6.sh actcons_e$k" | wc -l); $(tail -n 1 $W/chains/actcons_e$k/chain.log 2>/dev/null | cut -c1-140)"
  echo "started $(ts)" > $Z/ACTCONS_E${k}_STARTED.txt
done
say 'auto end: chain imag_b and both fine-tune chains started'

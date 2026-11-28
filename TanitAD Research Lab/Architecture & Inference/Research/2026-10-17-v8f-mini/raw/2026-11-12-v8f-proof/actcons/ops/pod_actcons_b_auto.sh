# pod_actcons_b_auto.sh : runs ON THE POD (nohup). Card PREREG_V8F_PROOF_ADD37 with note N3 (the term on the base with the rollout loss, RL).
# Writes the two arms files (base 0, base 1), then per base: waits for the base arm RL_s<k> of chain imag_a to be DONE (and, for base 1, for the arms of base 0),
# waits for a free GPU slot with 28 GB of the memory limit free, dry-gates RLFT_s<k> and RLA_s<k> (selfcheck only, trains nothing), checks the probe of the term,
# and only then starts chain actcons_b<k> (RLFT, RLA, RLAm, RLFTb; one arm at a time on one slot). Never a fourth GPU job. Fail closed.
set -u
W=/workspace/v8proof; Z=$W/runs/actcons_z; LOG=$Z/actcons_b_auto.log; mkdir -p $Z; cd $W/chain2 || exit 1
CARD=PREREG_V8F_PROOF_ADD37
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> $LOG; }
stopf() { say "$1"; echo "$1 ($(ts))" > $Z/ACTCONS_B_NOT_STARTED.txt; exit 1; }
[ -e $W/chains/actcons_b0 ] && { say 'chain actcons_b0 exists, nothing done'; exit 0; }
RLX='levers.wm_state=grid levers.wm_z1_width=640 levers.wm_cell_dim=16 levers.wm_l1_pred=flat levers.wm_l1_pred_width=256 levers.wm_l1_act=acc_yaw levers.wm_l1_futtok_k=0 levers.w_wm_futtok=0 levers.lead_sup=on levers.w_lead_sup=1 levers.lead_hist=5 levers.lead_sup_detach=off levers.wm_l1_residual=on levers.wm_l1_roll_k=10 levers.w_wm_roll=1 levers.wm_l1_diag=on'
RLA_ARGV='--wm-state grid --wm-cell-dim 16 --lead-sup on --w-lead-sup 1.0 --wm-l1-residual on --wm-l1-roll-k 10 --w-wm-roll 1.0 --wm-l1-diag on'
[ "$(grep '^ARM|RL_s0|' arms_imag_a.txt | cut -d'|' -f8)" = "$RLA_ARGV" ] || stopf 'the argv of RL in arms_imag_a.txt is not the one this script carries'
[ "$(grep '^ARM|RL_s0|' arms_imag_a.txt | cut -d'|' -f9)" = "$RLX info.n_clips=4512" ] || stopf 'the expectations of RL in arms_imag_a.txt are not the ones this script carries'
for k in 0 1; do
  A=arms_actcons_b$k.txt
  [ -e $A ] && continue
  I="--init-ckpt $W/runs/imag_a/RL_s$k/ckpt_last.pt --steps 6000 --lr 1e-4 --warmup 100"
  {
    echo "# ACTCONS on base RL ($CARD, note N3), base $k: control RLFT, named arm RLA (weight 1.0), dose arm RLAm (0.0001), rerun of the control RLFTb."
    grep '^PIN|' arms_actcons_a.txt
    echo "ARM|RLFT_s$k|$W/tree_actcons|HIERFT6|$k|540:600|$W/data/cache_pod|$RLA_ARGV $I|$RLX levers.w_act_cons=0 info.n_clips=4512"
    echo "ARM|RLA_s$k|$W/tree_actcons|HIERFT6|$k|540:600|$W/data/cache_pod|$RLA_ARGV $I --w-act-cons 1.0|$RLX levers.w_act_cons=1 levers.act_cons_max=1.5 info.n_clips=4512"
    echo "ARM|RLAm_s$k|$W/tree_actcons|HIERFT6|$k|540:600|$W/data/cache_pod|$RLA_ARGV $I --w-act-cons 0.0001|$RLX levers.w_act_cons=0.0001 levers.act_cons_max=1.5 info.n_clips=4512"
    echo "ARM|RLFTb_s$k|$W/tree_actcons|HIERFT6|$k|540:600|$W/data/cache_pod|$RLA_ARGV $I|$RLX levers.w_act_cons=0 info.n_clips=4512"
  } > $A
  say "arms file $A written: PIN $(grep -c '^PIN|' $A), ARM $(grep -c '^ARM|' $A), sha256 $(sha256sum $A | cut -c1-64)"
done
slot() { local k=0 n1 n2 a m; while :; do n1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); sleep 45; n2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    a=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); m=$(cat /sys/fs/cgroup/memory.max)
    [ "$n1" -lt 3 ] && [ "$n2" -lt 3 ] && [ $((m - a)) -gt 28000000000 ] && { say "slot free: GPU jobs $n2, memory free $(( (m - a) / 1000000000 )) GB"; return 0; }
    k=$((k+1)); [ $k -gt 640 ] && return 1; done; }
stage() { # stage <k>
  local k=$1 j=0 a rc
  say "base $k: waits for ARM DONE RL_s$k in chain imag_a"
  while ! grep -aq "ARM DONE RL_s$k" $W/chains/imag_a/chain.log 2>/dev/null; do
    [ -f $W/chains/imag_a/imag_a_STOP.txt ] && stopf "chain imag_a stopped before RL_s$k was done"
    j=$((j+1)); [ $j -gt 900 ] && stopf "RL_s$k not done in 15 h"; sleep 60
  done
  if [ $k = 1 ]; then j=0; while [ ! -f $W/chains/actcons_b0/actcons_b0_DONE.txt ]; do
      [ -f $W/chains/actcons_b0/actcons_b0_STOP.txt ] && stopf 'chain actcons_b0 stopped; base 1 is not started'
      j=$((j+1)); [ $j -gt 900 ] && stopf 'chain actcons_b0 not done in 15 h'; sleep 60; done; fi
  say "base $k: RL_s$k is done (checkpoint sha256 $(sha256sum $W/runs/imag_a/RL_s$k/ckpt_last.pt | cut -c1-16)); waits for a free GPU slot and 28 GB of memory"
  slot || stopf "base $k: no free GPU slot with 28 GB of memory in 8 h"
  for a in RLFT_s$k RLA_s$k; do
    bash dry_gate.sh arms_actcons_b$k.txt $a $CARD > $W/chains/actcons_b_dry_$a.out 2>&1; rc=$?
    say "dry gate $a rc $rc: $(tail -n 1 $W/chains/actcons_b_dry_$a.out | cut -c1-220)"
    [ $rc = 0 ] && grep -q "ZZ DRY $a PASS" $W/chains/actcons_b_dry_$a.out || stopf "dry gate $a did not pass"
  done
  K=$k /workspace/venv/bin/python - >> $LOG 2>&1 <<'PYEOF'
import json, os
k = os.environ["K"]
ok = True
for a in ("RLFT_s" + k, "RLA_s" + k):
    s = json.load(open('/workspace/v8proof/runs/motion_dry/%s/out/selfcheck.json' % a)); lv = s.get('levers', {}); p = s.get('act_cons_probe') or {}
    print('SELFCHECK', a, 'dvb', s.get('declared_vs_built'), 'live', (s.get('live') or {}).get('pass'), 'steps_total', s.get('steps_total'), 'w_act_cons', lv.get('w_act_cons'), 'roll_k', lv.get('wm_l1_roll_k'), 'residual', lv.get('wm_l1_residual'), 'init', json.dumps(lv.get('init_ckpt'))[:200])
    print('   act_cons probe', json.dumps(p)[:700])
    if a.startswith("RLA_") and p.get('pass') is not True: ok = False
open('/workspace/v8proof/runs/actcons_z/probe_b%s_pass.txt' % k, 'w').write('PASS' if ok else 'FAIL')
PYEOF
  [ "$(cat $Z/probe_b${k}_pass.txt 2>/dev/null)" = PASS ] || stopf "base $k: the probe of the term did not pass in the selfcheck of RLA_s$k"
  slot || stopf "base $k: no free GPU slot after the dry gates"
  C2_CARD=$CARD C2_WAIT_FOR=$W/chains/imag_a/chain.log C2_IDLE_MIB=30000 nohup setsid bash chain_arms_ft6.sh actcons_b$k arms_actcons_b$k.txt > $W/chains/actcons_b${k}_launch.out 2>&1 < /dev/null &
  sleep 40; say "chain started: actcons_b$k processes $(pgrep -f "[c]hain_arms_ft6.sh actcons_b$k" | wc -l); $(tail -n 1 $W/chains/actcons_b$k/chain.log 2>/dev/null | cut -c1-140)"
  echo "started $(ts)" > $Z/ACTCONS_B${k}_STARTED.txt
}
say 'auto start (card ADD37, note N3)'
stage 0
stage 1
say 'auto end: both chains started'

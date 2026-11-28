# pod_actcons_g_auto.sh : runs ON THE POD (nohup). Card PREREG_V8F_PROOF_ADD37, notes N4 and N5. It takes the place of pod_actcons_e_auto.sh, which never ran.
# Fills the three GPU slots:
#   chain imag_b2 (RL_s2) and chain imag_b3 (RL_s3): the line of RL_s0 of arms_imag_a.txt with the new seed, both at once (note N5 section 2);
#   chain actcons_f: the dose arms RLA3 (weight 0.001) and RLA2 (0.01) at bases 0 and 1 (note N5 section 3);
#   per new base k, after ARM DONE RL_s<k>: chain actcons_e<k> = RLFT_s<k>, RLAm_s<k> (note N4), then RLA3_s<k>, RLA2_s<k> (note N5).
# Before a chain starts: a free GPU slot with 28 GB of the memory limit free, the dry gates named in the notes (selfcheck only, they train nothing) and a check of the
# selfchecks (weight, start checkpoint). Never a fourth GPU job. Fail closed: a refusal writes ACTCONS_G_NOT_STARTED.txt and ends the script; chains already started go on.
set -u
W=/workspace/v8proof; Z=$W/runs/actcons_z; LOG=$Z/actcons_g_auto.log; mkdir -p $Z; cd $W/chain2 || exit 1
CARD=PREREG_V8F_PROOF_ADD37
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> $LOG; }
stopf() { say "$1"; echo "$1 ($(ts))" > $Z/ACTCONS_G_NOT_STARTED.txt; exit 1; }
for c in imag_b imag_b2 imag_b3 actcons_f actcons_e2 actcons_e3; do [ -e $W/chains/$c ] && { say "chain $c exists, nothing done"; exit 0; }; done
[ -f arms_imag_a.txt ] && [ -f arms_actcons_c0.txt ] && [ -f arms_actcons_c1.txt ] || stopf 'an arms file of the earlier chains is missing'
dose() { # dose <source arms file> <source arm> <new arm> <weight> : the line of RLAm with only the name and the weight changed
  grep "^ARM|$2|" "$1" | awk -F'|' -v OFS='|' -v n="$3" -v w="$4" '{$2=n; sub("--w-act-cons 0.0001", "--w-act-cons " w, $8); sub("levers.w_act_cons=0.0001", "levers.w_act_cons=" w, $9); print}'; }
rebase() { # rebase <k> : stdin = lines of base 0 -> the same lines on the checkpoint of RL_s<k> of chain imag_b<k>, seed k
  awk -F'|' -v OFS='|' -v k="$1" '{sub("_s0$", "_s" k, $2); $5=k; gsub("/runs/imag_a/RL_s0/", "/runs/imag_b" k "/RL_s" k "/", $8); print}'; }
for k in 2 3; do
  A=arms_imag_b$k.txt
  [ -e $A ] || { { echo "# RL base of a new seed ($CARD, notes N4 and N5): RL_s$k = the line of RL_s0 of arms_imag_a.txt with seed $k."
      grep '^PIN|' arms_imag_a.txt
      grep '^ARM|RL_s0|' arms_imag_a.txt | awk -F'|' -v OFS='|' -v k=$k '{$2="RL_s" k; $5=k; print}'; } > $A
    say "arms file $A written: PIN $(grep -c '^PIN|' $A), ARM $(grep -c '^ARM|' $A), sha256 $(sha256sum $A | cut -c1-64)"; }
  [ "$(grep -c '^ARM|' $A)" = 1 ] && [ "$(grep "^ARM|RL_s$k|" $A | cut -d'|' -f3-4,6-)" = "$(grep '^ARM|RL_s0|' arms_imag_a.txt | cut -d'|' -f3-4,6-)" ] && [ "$(grep "^ARM|RL_s$k|" $A | cut -d'|' -f5)" = $k ] || stopf "arms file $A is not the line of RL_s0 with seed $k"
  A=arms_actcons_e$k.txt
  [ -e $A ] || { { echo "# The low weight and the dose arms on a new base ($CARD, notes N4 and N5), base $k: RLFT, RLAm (0.0001), RLA3 (0.001), RLA2 (0.01) on the checkpoint of RL_s$k of chain imag_b$k."
      grep '^PIN|' arms_actcons_c0.txt
      grep '^ARM|RLFT_s0|' arms_actcons_c0.txt | rebase $k
      grep '^ARM|RLAm_s0|' arms_actcons_c0.txt | rebase $k
      dose arms_actcons_c0.txt RLAm_s0 RLA3_s0 0.001 | rebase $k
      dose arms_actcons_c0.txt RLAm_s0 RLA2_s0 0.01 | rebase $k; } > $A
    say "arms file $A written: PIN $(grep -c '^PIN|' $A), ARM $(grep -c '^ARM|' $A), sha256 $(sha256sum $A | cut -c1-64)"; }
  [ "$(grep -c '^ARM|' $A)" = 4 ] && [ "$(grep -c "runs/imag_b$k/RL_s$k/ckpt_last.pt" $A)" = 4 ] && [ "$(grep -c 'runs/imag_a/' $A)" = 0 ] && [ "$(cut -d'|' -f2,5 $A | grep -c "_s$k|$k\$")" = 4 ] \
    && [ "$(grep "^ARM|RLAm_s$k|" $A | grep -c -- '--w-act-cons 0.0001 *|.*levers.w_act_cons=0.0001 ')" = 1 ] && [ "$(grep "^ARM|RLA3_s$k|" $A | grep -c -- '--w-act-cons 0.001 *|.*levers.w_act_cons=0.001 ')" = 1 ] \
    && [ "$(grep "^ARM|RLA2_s$k|" $A | grep -c -- '--w-act-cons 0.01 *|.*levers.w_act_cons=0.01 ')" = 1 ] && [ "$(grep "^ARM|RLFT_s$k|" $A | grep -c -- 'w.act.cons')" = 0 ] || stopf "arms file $A is not the four lines this script expects"
done
A=arms_actcons_f.txt
[ -e $A ] || { { echo "# Dose arms on the existing bases ($CARD, note N5): RLA3 (weight 0.001) and RLA2 (0.01) = the line of RLAm with only the weight changed, bases 0 and 1."
    grep '^PIN|' arms_actcons_c0.txt
    dose arms_actcons_c0.txt RLAm_s0 RLA3_s0 0.001; dose arms_actcons_c0.txt RLAm_s0 RLA2_s0 0.01
    dose arms_actcons_c1.txt RLAm_s1 RLA3_s1 0.001; dose arms_actcons_c1.txt RLAm_s1 RLA2_s1 0.01; } > $A
  say "arms file $A written: PIN $(grep -c '^PIN|' $A), ARM $(grep -c '^ARM|' $A), sha256 $(sha256sum $A | cut -c1-64)"; }
[ "$(grep -c '^ARM|' $A)" = 4 ] && [ "$(grep -c -- '--w-act-cons 0.001 *|.*levers.w_act_cons=0.001 ' $A)" = 2 ] && [ "$(grep -c -- '--w-act-cons 0.01 *|.*levers.w_act_cons=0.01 ' $A)" = 2 ] && [ "$(grep -c 'runs/imag_a/RL_s0/ckpt_last.pt' $A)" = 2 ] && [ "$(grep -c 'runs/imag_a/RL_s1/ckpt_last.pt' $A)" = 2 ] || stopf "arms file $A is not the four dose lines this script expects"
slot() { local k=0 n1 n2 a m; while :; do n1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); sleep 45; n2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    a=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); m=$(cat /sys/fs/cgroup/memory.max)
    [ "$n1" -lt 3 ] && [ "$n2" -lt 3 ] && [ $((m - a)) -gt 28000000000 ] && { say "slot free: GPU jobs $n2, memory free $(( (m - a) / 1000000000 )) GB"; return 0; }
    k=$((k+1)); [ $k -gt 640 ] && return 1; done; }
selfcheck() { # selfcheck <tag> <arm>... : prints the selfcheck of each arm; PASS only if every arm built what its name says
  local tag=$1; shift
  ARMS_="$*" TAG_=$tag /workspace/venv/bin/python - >> $LOG 2>&1 <<'PYEOF'
import json, os
WT = {'RLAm': 0.0001, 'RLA3': 0.001, 'RLA2': 0.01}
ok = True
for a in os.environ["ARMS_"].split():
    s = json.load(open('/workspace/v8proof/runs/motion_dry/%s/out/selfcheck.json' % a)); lv = s.get('levers', {}); p = s.get('act_cons_probe') or {}
    name, k = a.rsplit('_s', 1)
    init = json.dumps(lv.get('init_ckpt'))
    w = float(lv.get('w_act_cons') or 0.0)
    print('SELFCHECK', a, 'dvb', s.get('declared_vs_built'), 'live', (s.get('live') or {}).get('pass'), 'steps_total', s.get('steps_total'), 'w_act_cons', lv.get('w_act_cons'), 'roll_k', lv.get('wm_l1_roll_k'),
          'residual', lv.get('wm_l1_residual'), 'init', init[:200])
    if name in WT:
        print('   act_cons probe', json.dumps(p)[:700])
        if abs(w - WT[name]) > 1e-12: ok = False
    if name == 'RLFT' and w != 0.0: ok = False
    if name != 'RL':
        base = ('imag_a/RL_s%s/' % k) if k in ('0', '1') else ('imag_b%s/RL_s%s/' % (k, k))
        if base not in init: ok = False
    if name == 'RL' and (str(lv.get('wm_l1_roll_k')) not in ('10', '10.0') or str(lv.get('wm_l1_residual')).lower() not in ('on', 'true', '1')): ok = False
open('/workspace/v8proof/runs/actcons_z/selfcheck_%s_pass.txt' % os.environ["TAG_"], 'w').write('PASS' if ok else 'FAIL')
PYEOF
  [ "$(cat $Z/selfcheck_${tag}_pass.txt 2>/dev/null)" = PASS ]
}
dry() { local a rc; for a in "$@"; do
    bash dry_gate.sh "$ARMSF" $a $CARD > $W/chains/actcons_g_dry_$a.out 2>&1; rc=$?
    say "dry gate $a rc $rc: $(tail -n 1 $W/chains/actcons_g_dry_$a.out | cut -c1-220)"
    [ $rc = 0 ] && grep -q "ZZ DRY $a PASS" $W/chains/actcons_g_dry_$a.out || stopf "dry gate $a did not pass"
  done; }
start() { # start <chain> <arms file> <chain script> <wait-for file> <idle MiB>
  slot || stopf "$1: no free GPU slot with 28 GB of memory in 8 h"
  C2_CARD=$CARD C2_WAIT_FOR=$4 C2_IDLE_MIB=$5 nohup setsid bash $3 $1 $2 > $W/chains/${1}_launch.out 2>&1 < /dev/null &
  sleep 40; say "chain started: $1 processes $(pgrep -f "[c]hain_arms.*sh $1 " | wc -l); $(tail -n 1 $W/chains/$1/chain.log 2>/dev/null | cut -c1-140)"
  echo "started $(ts)" > $Z/STARTED_$1.txt
}
say 'auto start (card ADD37, notes N4 and N5)'
slot || stopf 'no free GPU slot with 28 GB of memory in 8 h'
ARMSF=arms_imag_b2.txt; dry RL_s2
selfcheck imag_b RL_s2 || stopf 'the selfcheck of RL_s2 is not the one of an RL arm'
start imag_b2 arms_imag_b2.txt chain_arms.sh $W/chains/imag_a/chain.log 30000
start imag_b3 arms_imag_b3.txt chain_arms.sh $W/chains/imag_a/chain.log 30000
ARMSF=arms_actcons_f.txt; dry RLA3_s0 RLA2_s0
selfcheck f RLA3_s0 RLA2_s0 || stopf 'a selfcheck of the dose arms does not show the weight or the start checkpoint its arm names'
start actcons_f arms_actcons_f.txt chain_arms_ft6.sh $W/chains/imag_a/chain.log 30000
for k in 2 3; do
  say "base $k: waits for ARM DONE RL_s$k in chain imag_b$k"
  j=0; while ! grep -aq "ARM DONE RL_s$k" $W/chains/imag_b$k/chain.log 2>/dev/null; do
    [ -f $W/chains/imag_b$k/imag_b${k}_STOP.txt ] && { say "chain imag_b$k stopped before RL_s$k was done; base $k is left out"; continue 2; }
    j=$((j+1)); [ $j -gt 900 ] && { say "RL_s$k not done in 15 h; base $k is left out"; continue 2; }; sleep 60
  done
  say "base $k: RL_s$k is done (checkpoint sha256 $(sha256sum $W/runs/imag_b$k/RL_s$k/ckpt_last.pt | cut -c1-16))"
  slot || stopf "base $k: no free GPU slot with 28 GB of memory in 8 h"
  ARMSF=arms_actcons_e$k.txt; dry RLFT_s$k RLAm_s$k RLA3_s$k
  selfcheck e$k RLFT_s$k RLAm_s$k RLA3_s$k || stopf "base $k: a selfcheck does not show the weight or the start checkpoint its arm names"
  start actcons_e$k arms_actcons_e$k.txt chain_arms_ft6.sh $W/chains/imag_b$k/chain.log 30000
done
say 'auto end'

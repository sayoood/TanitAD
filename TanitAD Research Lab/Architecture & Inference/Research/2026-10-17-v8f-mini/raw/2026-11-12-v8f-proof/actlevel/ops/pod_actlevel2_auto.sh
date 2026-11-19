# pod_actlevel2_auto.sh : runs ON THE POD (nohup). Card PREREG_V8F_PROOF_ADD37, note N7 (two interior weights of the level term).
# Made by actlevel/ops/make_actlevel2.py from pod_actlevel_auto.sh (note N6). Two chains, one per base k in 0 1:
# actlevel2_<k> = RLVc_s<k> (W 0.001, V 0.002), RLVd_s<k> (W 0.001, V 0.004), each the line of RLAm_s<k> of arms_actcons_c<k>.txt with the tree, the name
# and the named flags changed. Tree: /workspace/v8proof/tree_actlevel (pins unchanged).
# Before a chain starts: a free slot (fewer than three GPU jobs on two probes 45 s apart) with 28 GB of the memory limit free, the pins of the tree, the dry
# gates of its arms (selfcheck only, they train nothing) and a check of the selfchecks (both weights, start checkpoint). Chain actlevel2_1 starts only after
# chain actlevel2_0 has started and a slot is free again. Fail closed: a refusal writes ACTLEVEL2_NOT_STARTED.txt and ends the script. No process test
# decides anything. It deletes nothing.
set -u
W=/workspace/v8proof; Z=$W/runs/actlevel2_z; LOG=$Z/actlevel2_auto.log; mkdir -p $Z; cd $W/chain2 || exit 1
CARD=PREREG_V8F_PROOF_ADD37
TREE=/workspace/v8proof/tree_actlevel
PINS=$W/chain2/pins_actlevel.txt
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> $LOG; }
stopf() { say "$1"; echo "$1 ($(ts))" > $Z/ACTLEVEL2_NOT_STARTED.txt; exit 1; }
for c in actlevel2_0 actlevel2_1; do [ -e $W/chains/$c ] && { say "chain $c exists, nothing done"; exit 0; }; done
[ -f arms_actcons_c0.txt ] && [ -f arms_actcons_c1.txt ] || stopf 'an arms file of the earlier chains is missing'
[ -d $TREE ] && [ -f $PINS ] || stopf 'the tree of the level term or its pins file is missing'
np=0
while IFS='|' read -r tag tree rel sha; do
  [ "$tag" = PIN ] || continue
  [ "$tree" = $TREE ] || stopf "a pin names another tree: $tree"
  [ "$(sha256sum "$tree/$rel" 2>/dev/null | cut -c1-64)" = "$sha" ] || stopf "pin does not hold: $rel"
  np=$((np+1))
done < $PINS
[ $np -ge 14 ] || stopf "only $np pins in the pins file"
say "pins hold: $np files of $TREE"
mk() { # mk <source arms file> <source arm> <new arm> <consistency weight> <level weight or 0>
  grep "^ARM|$2|" "$1" | awk -F'|' -v OFS='|' -v n="$3" -v w="$4" -v v="$5" -v t="$TREE" '{ $2=n; $3=t
    c1=sub("--w-act-cons 0.0001", "--w-act-cons " w, $8); c2=sub("levers.w_act_cons=0.0001", "levers.w_act_cons=" w, $9); c3=1
    if (v != "0") { $8=$8 " --w-act-level " v; c3=sub("levers.act_cons_max=1.5", "levers.act_cons_max=1.5 levers.w_act_level=" v, $9) }
    if (c1==1 && c2==1 && c3==1) print }'; }
for k in 0 1; do
  A=arms_actlevel2_$k.txt
  [ -e $A ] || { { echo "# Two interior weights of the level term ($CARD, note N7), base $k: RLVc (W 0.001, V 0.002), RLVd (W 0.001, V 0.004); each the line of RLAm_s$k with the tree, the name and the named flags changed."
      grep '^PIN|' $PINS
      mk arms_actcons_c$k.txt RLAm_s$k RLVc_s$k 0.001 0.002
      mk arms_actcons_c$k.txt RLAm_s$k RLVd_s$k 0.001 0.004; } > $A
    say "arms file $A written: PIN $(grep -c '^PIN|' $A), ARM $(grep -c '^ARM|' $A), sha256 $(sha256sum $A | cut -c1-64)"; }
  [ "$(grep -c '^ARM|' $A)" = 2 ] && [ "$(grep -c "^ARM|[A-Za-z0-9]*_s$k|$TREE|" $A)" = 2 ] && [ "$(grep -c "runs/imag_a/RL_s$k/ckpt_last.pt" $A)" = 2 ] \
    && [ "$(grep "^ARM|RLVc_s$k|" $A | grep -c -- '--w-act-cons 0.001 --w-act-level 0.002|.*levers.w_act_cons=0.001 levers.act_cons_max=1.5 levers.w_act_level=0.002 ')" = 1 ] \
    && [ "$(grep "^ARM|RLVd_s$k|" $A | grep -c -- '--w-act-cons 0.001 --w-act-level 0.004|.*levers.w_act_cons=0.001 levers.act_cons_max=1.5 levers.w_act_level=0.004 ')" = 1 ] || stopf "arms file $A is not the two lines this script expects"
done
# order of the pod: first the rollouts of the LS dose arms (note ADD42_N3) must have ended (their DONE or STOP marker; at most 4 h, then it goes on);
# after that the slot wait decides (fewer than three GPU jobs and 28 GB of memory free).
j=0; while [ -d $W/qhead_dose_ls/run_roll ] && [ ! -e $W/qhead_dose_ls/run_roll/ROLLOUTS_DONE.txt ] && [ ! -e $W/qhead_dose_ls/run_roll/ROLLOUTS_STOP.txt ]; do j=$((j+1)); [ $j -gt 240 ] && { say 'the rollouts of the LS dose arms have not ended in 4 h; going on'; break; }; sleep 60; done
say "rollouts of the LS dose arms: $(ls $W/qhead_dose_ls/run_roll/ 2>/dev/null | grep -c ROLLOUTS_) end marker(s)"
slot() { local k=0 n1 n2 a m; while :; do n1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); sleep 45; n2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    a=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); m=$(cat /sys/fs/cgroup/memory.max)
    [ "$n1" -lt 3 ] && [ "$n2" -lt 3 ] && [ $((m - a)) -gt 28000000000 ] && { say "slot free: GPU jobs $n2, memory free $(( (m - a) / 1000000000 )) GB"; return 0; }
    k=$((k+1)); [ $k -gt 640 ] && return 1; done; }
selfcheck() { # selfcheck <tag> <arm>... : prints the selfcheck of each arm; PASS only if every arm built what its name says
  local tag=$1; shift
  ARMS_="$*" TAG_=$tag /workspace/venv/bin/python - >> $LOG 2>&1 <<'PYEOF'
import json, os
WT = {'RLVc': (0.001, 0.002), 'RLVd': (0.001, 0.004)}
ok = True
for a in os.environ["ARMS_"].split():
    s = json.load(open('/workspace/v8proof/runs/motion_dry/%s/out/selfcheck.json' % a)); lv = s.get('levers', {}); p = s.get('act_cons_probe') or {}
    name, k = a.rsplit('_s', 1)
    init = json.dumps(lv.get('init_ckpt'))
    w = float(lv.get('w_act_cons') or 0.0); v = float(lv.get('w_act_level') or 0.0)
    print('SELFCHECK', a, 'dvb', s.get('declared_vs_built'), 'live', (s.get('live') or {}).get('pass'), 'steps_total', s.get('steps_total'), 'w_act_cons', lv.get('w_act_cons'),
          'w_act_level', lv.get('w_act_level'), 'roll_k', lv.get('wm_l1_roll_k'), 'init', init[:200])
    print('   act_cons probe', json.dumps(p)[:900])
    if name not in WT: ok = False; continue
    if abs(w - WT[name][0]) > 1e-12 or abs(v - WT[name][1]) > 1e-12: ok = False
    if ('imag_a/RL_s%s/' % k) not in init: ok = False
    if p.get('pass') is not True: ok = False
open('/workspace/v8proof/runs/actlevel2_z/selfcheck_%s_pass.txt' % os.environ["TAG_"], 'w').write('PASS' if ok else 'FAIL')
PYEOF
  [ "$(cat $Z/selfcheck_${tag}_pass.txt 2>/dev/null)" = PASS ]
}
dry() { local a rc; for a in "$@"; do
    bash dry_gate.sh "$ARMSF" $a $CARD > $W/chains/actlevel2_dry_$a.out 2>&1; rc=$?
    say "dry gate $a rc $rc: $(tail -n 1 $W/chains/actlevel2_dry_$a.out | cut -c1-220)"
    [ $rc = 0 ] && grep -q "ZZ DRY $a PASS" $W/chains/actlevel2_dry_$a.out || stopf "dry gate $a did not pass"
  done; }
start() { # start <chain> <arms file>
  slot || stopf "$1: no free GPU slot with 28 GB of memory in 8 h"
  C2_CARD=$CARD C2_WAIT_FOR=$W/chains/imag_a/chain.log C2_IDLE_MIB=30000 nohup setsid bash chain_arms_ft6.sh $1 $2 > $W/chains/${1}_launch.out 2>&1 < /dev/null &
  sleep 40; say "chain started: $1; $(tail -n 1 $W/chains/$1/chain.log 2>/dev/null | cut -c1-140)"
  echo "started $(ts)" > $Z/STARTED_$1.txt
}
say 'auto start (card ADD37, note N7)'
slot || stopf 'no free GPU slot with 28 GB of memory in 8 h'
ARMSF=arms_actlevel2_0.txt; dry RLVc_s0 RLVd_s0
selfcheck k0 RLVc_s0 RLVd_s0 || stopf 'a selfcheck at base 0 does not show the weights or the start checkpoint its arm names'
start actlevel2_0 arms_actlevel2_0.txt
slot || stopf 'actlevel2_1: no free GPU slot with 28 GB of memory in 8 h'
ARMSF=arms_actlevel2_1.txt; dry RLVc_s1 RLVd_s1
selfcheck k1 RLVc_s1 RLVd_s1 || stopf 'a selfcheck at base 1 does not show the weights or the start checkpoint its arm names'
start actlevel2_1 arms_actlevel2_1.txt
say 'auto end'

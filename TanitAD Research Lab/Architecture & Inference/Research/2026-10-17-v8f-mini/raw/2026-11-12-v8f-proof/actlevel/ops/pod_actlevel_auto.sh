# pod_actlevel_auto.sh : runs ON THE POD (nohup). Card PREREG_V8F_PROOF_ADD37, note N6 (level term inside the consistency loss).
# Two chains, one per base k in 0 1: actlevel_<k> = RLVa_s<k> (W 0.001, V 0.001), RLVb_s<k> (W 0.001, V 0.01), RLAh_s<k> (W 0.0003, no level term),
# each the line of RLAm_s<k> of arms_actcons_c<k>.txt with the tree, the name and the named flags changed. Tree: /workspace/v8proof/tree_actlevel.
# Order of the pod (note N6 section 3): this script first waits until the queue of card ADD42 has ended (its DONE or STOP marker; at most 5 h, then it goes on),
# until both chains actcons_e2 and actcons_e3 have ended, and until the pilot of card ADD41 has ended. After that the only other GPU user is the tool sequence of
# the dev box (one job at a time, it refuses at three GPU jobs), so two chains keep the pod at three GPU jobs at most.
# Before a chain starts: a free slot with 28 GB of the memory limit free, the pins of the tree, the dry gates (selfcheck only, they train nothing) and a check of
# the selfchecks (both weights, start checkpoint). Fail closed: a refusal writes ACTLEVEL_NOT_STARTED.txt and ends the script. No process test decides anything.
set -u
W=/workspace/v8proof; Z=$W/runs/actlevel_z; LOG=$Z/actlevel_auto.log; mkdir -p $Z; cd $W/chain2 || exit 1
CARD=PREREG_V8F_PROOF_ADD37
TREE=/workspace/v8proof/tree_actlevel
PINS=$W/chain2/pins_actlevel.txt
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> $LOG; }
stopf() { say "$1"; echo "$1 ($(ts))" > $Z/ACTLEVEL_NOT_STARTED.txt; exit 1; }
for c in actlevel_0 actlevel_1; do [ -e $W/chains/$c ] && { say "chain $c exists, nothing done"; exit 0; }; done
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
  A=arms_actlevel_$k.txt
  [ -e $A ] || { { echo "# Level term inside the consistency loss ($CARD, note N6), base $k: RLVa (W 0.001, V 0.001), RLVb (W 0.001, V 0.01), RLAh (W 0.0003, no level term); each the line of RLAm_s$k with the tree, the name and the named flags changed."
      grep '^PIN|' $PINS
      mk arms_actcons_c$k.txt RLAm_s$k RLVa_s$k 0.001 0.001
      mk arms_actcons_c$k.txt RLAm_s$k RLVb_s$k 0.001 0.01
      mk arms_actcons_c$k.txt RLAm_s$k RLAh_s$k 0.0003 0; } > $A
    say "arms file $A written: PIN $(grep -c '^PIN|' $A), ARM $(grep -c '^ARM|' $A), sha256 $(sha256sum $A | cut -c1-64)"; }
  [ "$(grep -c '^ARM|' $A)" = 3 ] && [ "$(grep -c "^ARM|[A-Za-z0-9]*_s$k|$TREE|" $A)" = 3 ] && [ "$(grep -c "runs/imag_a/RL_s$k/ckpt_last.pt" $A)" = 3 ] \
    && [ "$(grep "^ARM|RLVa_s$k|" $A | grep -c -- '--w-act-cons 0.001 --w-act-level 0.001|.*levers.w_act_cons=0.001 levers.act_cons_max=1.5 levers.w_act_level=0.001 ')" = 1 ] \
    && [ "$(grep "^ARM|RLVb_s$k|" $A | grep -c -- '--w-act-cons 0.001 --w-act-level 0.01|.*levers.w_act_cons=0.001 levers.act_cons_max=1.5 levers.w_act_level=0.01 ')" = 1 ] \
    && [ "$(grep "^ARM|RLAh_s$k|" $A | grep -c -- '--w-act-cons 0.0003|.*levers.w_act_cons=0.0003 levers.act_cons_max=1.5 info')" = 1 ] \
    && [ "$(grep "^ARM|RLAh_s$k|" $A | grep -c -- 'act.level')" = 0 ] || stopf "arms file $A is not the three lines this script expects"
done
# order of the pod (second version, 2026-11-19 03:15Z; the first version tested the two end markers with one ls, which is only true when BOTH exist, so its wait
# could never end; it was stopped by the lead while it waited and had started nothing): the chains of notes N4 and N5 and the pilot of card ADD41 must have ended
# before the first dry gate; chain actlevel_0 starts then, beside the queue of card ADD42 (one job at a time) and the tool sequence of the dev box (one job at a
# time): three GPU jobs at most. Chain actlevel_1 starts only after the queue of card ADD42 has ended (its DONE or STOP marker).
ended() { [ -e "$1" ] || [ -e "$2" ]; }
for c in actcons_e2 actcons_e3; do
  j=0; while [ -d $W/chains/$c ] && ! ended $W/chains/$c/${c}_DONE.txt $W/chains/$c/${c}_STOP.txt; do j=$((j+1)); [ $j -gt 600 ] && stopf "chain $c has not ended in 10 h"; sleep 60; done
  say "chain $c: $(ls $W/chains/$c/${c}_DONE.txt $W/chains/$c/${c}_STOP.txt 2>/dev/null | tr '\n' ' ')"
done
j=0; while [ -f $W/runs/finegrid/FINEGRID_START_TAKEN.txt ] && ! ended $W/runs/finegrid/FINEGRID_DONE.txt $W/runs/finegrid/FINEGRID_STOP.txt; do j=$((j+1)); [ $j -gt 600 ] && stopf 'the pilot of card ADD41 has not ended in 10 h'; sleep 60; done
say "pilot of card ADD41: $(ls $W/runs/finegrid/FINEGRID_DONE.txt $W/runs/finegrid/FINEGRID_STOP.txt 2>/dev/null | tr '\n' ' ')"
waitq() { local j=0; while ! ended $W/qhead_run/QHEAD_DONE.txt $W/qhead_run/QHEAD_STOP.txt; do j=$((j+1)); [ $j -gt 360 ] && { say 'the queue of card ADD42 has not ended in 6 h; going on'; break; }; sleep 60; done
  say "queue of card ADD42: $(ls $W/qhead_run/QHEAD_DONE.txt $W/qhead_run/QHEAD_STOP.txt 2>/dev/null | tr '\n' ' ')"; }
slot() { local k=0 n1 n2 a m; while :; do n1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); sleep 45; n2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    a=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); m=$(cat /sys/fs/cgroup/memory.max)
    [ "$n1" -lt 3 ] && [ "$n2" -lt 3 ] && [ $((m - a)) -gt 28000000000 ] && { say "slot free: GPU jobs $n2, memory free $(( (m - a) / 1000000000 )) GB"; return 0; }
    k=$((k+1)); [ $k -gt 640 ] && return 1; done; }
selfcheck() { # selfcheck <tag> <arm>... : prints the selfcheck of each arm; PASS only if every arm built what its name says
  local tag=$1; shift
  ARMS_="$*" TAG_=$tag /workspace/venv/bin/python - >> $LOG 2>&1 <<'PYEOF'
import json, os
WT = {'RLVa': (0.001, 0.001), 'RLVb': (0.001, 0.01), 'RLAh': (0.0003, 0.0)}
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
open('/workspace/v8proof/runs/actlevel_z/selfcheck_%s_pass.txt' % os.environ["TAG_"], 'w').write('PASS' if ok else 'FAIL')
PYEOF
  [ "$(cat $Z/selfcheck_${tag}_pass.txt 2>/dev/null)" = PASS ]
}
dry() { local a rc; for a in "$@"; do
    bash dry_gate.sh "$ARMSF" $a $CARD > $W/chains/actlevel_dry_$a.out 2>&1; rc=$?
    say "dry gate $a rc $rc: $(tail -n 1 $W/chains/actlevel_dry_$a.out | cut -c1-220)"
    [ $rc = 0 ] && grep -q "ZZ DRY $a PASS" $W/chains/actlevel_dry_$a.out || stopf "dry gate $a did not pass"
  done; }
start() { # start <chain> <arms file>
  slot || stopf "$1: no free GPU slot with 28 GB of memory in 8 h"
  C2_CARD=$CARD C2_WAIT_FOR=$W/chains/imag_a/chain.log C2_IDLE_MIB=30000 nohup setsid bash chain_arms_ft6.sh $1 $2 > $W/chains/${1}_launch.out 2>&1 < /dev/null &
  sleep 40; say "chain started: $1; $(tail -n 1 $W/chains/$1/chain.log 2>/dev/null | cut -c1-140)"
  echo "started $(ts)" > $Z/STARTED_$1.txt
}
say 'auto start (card ADD37, note N6)'
slot || stopf 'no free GPU slot with 28 GB of memory in 8 h'
ARMSF=arms_actlevel_0.txt; dry RLVa_s0 RLVb_s0 RLAh_s0
selfcheck k0 RLVa_s0 RLVb_s0 RLAh_s0 || stopf 'a selfcheck at base 0 does not show the weights or the start checkpoint its arm names'
start actlevel_0 arms_actlevel_0.txt
waitq
slot || stopf 'actlevel_1: no free GPU slot with 28 GB of memory in 8 h'
ARMSF=arms_actlevel_1.txt; dry RLVa_s1
selfcheck k1 RLVa_s1 || stopf 'the selfcheck of RLVa_s1 does not show the weights or the start checkpoint its arm names'
start actlevel_1 arms_actlevel_1.txt
say 'auto end'

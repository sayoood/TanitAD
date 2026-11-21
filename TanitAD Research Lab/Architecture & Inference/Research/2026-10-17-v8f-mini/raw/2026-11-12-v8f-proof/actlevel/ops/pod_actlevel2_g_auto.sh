# pod_actlevel2_g_auto.sh <ARM>:<W>:<V> : runs ON THE POD (nohup). Card PREREG_V8F_PROOF_ADD37, note N7 section 4 (confirmation of a candidate, committed before the data):
# the arm named by the read at bases 0 and 1 is trained at the bases of the new seeds 2 and 3. <ARM> is RLVc or RLVd; W is the consistency weight,
# V the level weight (0 = no level term). Two chains, one arm each: alg_<ARM>_2 = <ARM>_s2 and alg_<ARM>_3 = <ARM>_s3, each the line of RLAm_s<k> of
# arms_actcons_e<k>.txt with the tree, the name and the named flags changed. Tree: /workspace/v8proof/tree_actlevel.
# MODE=arms : writes and checks the two arms files and ends (no GPU, nothing started). Default MODE=run: after the arms files a free slot (fewer than three
# GPU jobs on two probes 45 s apart and 28 GB of the memory limit free), the pins of the tree, the dry gates (selfcheck only, they train nothing), a check of the
# selfchecks (both weights, start checkpoint of the own base) and then the chains, each behind its own slot wait. Never a fourth GPU job.
# Fail closed: a refusal writes ALG_<ARM>_NOT_STARTED.txt and ends the script. No process test decides anything. Deletes nothing.
set -u
SPEC=${1:?usage: pod_actlevel2_g_auto.sh ARM:W:V}
ARM=${SPEC%%:*}; R_=${SPEC#*:}; WC=${R_%%:*}; WV=${R_##*:}
case "$ARM:$WC:$WV" in RLVc:0.001:0.002|RLVd:0.001:0.004) ;; *) echo "not an arm of note N7: $SPEC"; exit 2 ;; esac
MODE=${MODE:-run}
W=/workspace/v8proof; Z=$W/runs/actlevel_z; LOG=$Z/alg_${ARM}_auto.log; mkdir -p $Z; cd $W/chain2 || exit 1
CARD=PREREG_V8F_PROOF_ADD37
TREE=/workspace/v8proof/tree_actlevel
PINS=$W/chain2/pins_actlevel.txt
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" >> $LOG; }
stopf() { say "$1"; echo "$1 ($(ts))" > $Z/ALG_${ARM}_NOT_STARTED.txt; exit 1; }
if [ "$MODE" = run ]; then for k in 2 3; do [ -e $W/chains/alg_${ARM}_$k ] && { say "chain alg_${ARM}_$k exists, nothing done"; exit 0; }; done; fi
[ -f arms_actcons_e2.txt ] && [ -f arms_actcons_e3.txt ] || stopf 'an arms file of the chains of the new seeds is missing'
[ -d $TREE ] && [ -f $PINS ] || stopf 'the tree of the level term or its pins file is missing'
np=0
while IFS='|' read -r tag tree rel sha; do
  [ "$tag" = PIN ] || continue
  [ "$tree" = $TREE ] || stopf "a pin names another tree: $tree"
  [ "$(sha256sum "$tree/$rel" 2>/dev/null | cut -c1-64)" = "$sha" ] || stopf "pin does not hold: $rel"
  np=$((np+1))
done < $PINS
[ $np -ge 14 ] || stopf "only $np pins in the pins file"
say "pins hold: $np files of $TREE (spec $SPEC, mode $MODE)"
mk() { # mk <source arms file> <source arm> <new arm> <consistency weight> <level weight or 0>
  grep "^ARM|$2|" "$1" | awk -F'|' -v OFS='|' -v n="$3" -v w="$4" -v v="$5" -v t="$TREE" '{ $2=n; $3=t
    c1=sub("--w-act-cons 0.0001", "--w-act-cons " w, $8); c2=sub("levers.w_act_cons=0.0001", "levers.w_act_cons=" w, $9); c3=1
    if (v != "0") { $8=$8 " --w-act-level " v; c3=sub("levers.act_cons_max=1.5", "levers.act_cons_max=1.5 levers.w_act_level=" v, $9) }
    if (c1==1 && c2==1 && c3==1) print }'; }
for k in 2 3; do
  A=arms_alg_${ARM}_$k.txt
  [ -e $A ] || { { echo "# Level term inside the consistency loss ($CARD, note N6 section 6), base $k of the new seeds: ${ARM}_s$k (W $WC, V $WV; V 0 = no level term); the line of RLAm_s$k with the tree, the name and the named flags changed."
      grep '^PIN|' $PINS
      mk arms_actcons_e$k.txt RLAm_s$k ${ARM}_s$k $WC $WV; } > $A
    say "arms file $A written: PIN $(grep -c '^PIN|' $A), ARM $(grep -c '^ARM|' $A), sha256 $(sha256sum $A | cut -c1-64)"; }
  ok=1
  [ "$(grep -c '^ARM|' $A)" = 1 ] && [ "$(grep -c "^ARM|${ARM}_s$k|$TREE|" $A)" = 1 ] && [ "$(grep -c "runs/imag_b$k/RL_s$k/ckpt_last.pt" $A)" = 1 ] || ok=0
  if [ "$WV" = 0 ]; then
    [ "$(grep "^ARM|${ARM}_s$k|" $A | grep -c -- "--w-act-cons $WC|.*levers.w_act_cons=$WC levers.act_cons_max=1.5 info")" = 1 ] && [ "$(grep "^ARM|${ARM}_s$k|" $A | grep -c -- 'act.level')" = 0 ] || ok=0
  else
    [ "$(grep "^ARM|${ARM}_s$k|" $A | grep -c -- "--w-act-cons $WC --w-act-level $WV|.*levers.w_act_cons=$WC levers.act_cons_max=1.5 levers.w_act_level=$WV ")" = 1 ] || ok=0
  fi
  [ $ok = 1 ] || stopf "arms file $A is not the one line this script expects"
done
if [ "$MODE" = arms ]; then say 'mode arms: arms files written and checked, nothing started'; echo "ARMS-OK $ARM $(sha256sum arms_alg_${ARM}_2.txt arms_alg_${ARM}_3.txt | cut -c1-16 | tr '\n' ' ')"; exit 0; fi
slot() { local k=0 n1 n2 a m; while :; do n1=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); sleep 45; n2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    a=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); m=$(cat /sys/fs/cgroup/memory.max)
    [ "$n1" -lt 3 ] && [ "$n2" -lt 3 ] && [ $((m - a)) -gt 28000000000 ] && { say "slot free: GPU jobs $n2, memory free $(( (m - a) / 1000000000 )) GB"; return 0; }
    k=$((k+1)); [ $k -gt 640 ] && return 1; done; }
selfcheck() { # selfcheck <tag> <arm>... : prints the selfcheck of each arm; PASS only if every arm built what its name says
  local tag=$1; shift
  ARMS_="$*" TAG_=$tag WC_=$WC WV_=$WV /workspace/venv/bin/python - >> $LOG 2>&1 <<'PYEOF'
import json, os
wc, wv = float(os.environ["WC_"]), float(os.environ["WV_"])
ok = True
for a in os.environ["ARMS_"].split():
    s = json.load(open('/workspace/v8proof/runs/motion_dry/%s/out/selfcheck.json' % a)); lv = s.get('levers', {}); p = s.get('act_cons_probe') or {}
    name, k = a.rsplit('_s', 1)
    init = json.dumps(lv.get('init_ckpt'))
    w = float(lv.get('w_act_cons') or 0.0); v = float(lv.get('w_act_level') or 0.0)
    print('SELFCHECK', a, 'dvb', s.get('declared_vs_built'), 'live', (s.get('live') or {}).get('pass'), 'steps_total', s.get('steps_total'), 'w_act_cons', lv.get('w_act_cons'),
          'w_act_level', lv.get('w_act_level'), 'roll_k', lv.get('wm_l1_roll_k'), 'init', init[:200])
    print('   act_cons probe', json.dumps(p)[:900])
    if abs(w - wc) > 1e-12 or abs(v - wv) > 1e-12: ok = False
    if ('imag_b%s/RL_s%s/' % (k, k)) not in init: ok = False
    if p.get('pass') is not True: ok = False
    if s.get('steps_total') != 6000: ok = False
open('/workspace/v8proof/runs/actlevel_z/selfcheck_%s_pass.txt' % os.environ["TAG_"], 'w').write('PASS' if ok else 'FAIL')
PYEOF
  [ "$(cat $Z/selfcheck_${tag}_pass.txt 2>/dev/null)" = PASS ]
}
dry() { local a rc; for a in "$@"; do
    bash dry_gate.sh "$ARMSF" $a $CARD > $W/chains/alg_dry_$a.out 2>&1; rc=$?
    say "dry gate $a rc $rc: $(tail -n 1 $W/chains/alg_dry_$a.out | cut -c1-220)"
    [ $rc = 0 ] && grep -q "ZZ DRY $a PASS" $W/chains/alg_dry_$a.out || stopf "dry gate $a did not pass"
  done; }
start() { # start <chain> <arms file>
  slot || stopf "$1: no free GPU slot with 28 GB of memory in 8 h"
  C2_CARD=$CARD C2_WAIT_FOR=$W/chains/imag_a/chain.log C2_IDLE_MIB=30000 nohup setsid bash chain_arms_ft6.sh $1 $2 > $W/chains/${1}_launch.out 2>&1 < /dev/null &
  sleep 40; say "chain started: $1; $(tail -n 1 $W/chains/$1/chain.log 2>/dev/null | cut -c1-140)"
  echo "started $(ts)" > $Z/STARTED_$1.txt
}
say "auto start (card ADD37, note N6 section 6, arm $ARM at bases 2 and 3)"
for k in 2 3; do
  slot || stopf "no free GPU slot with 28 GB of memory in 8 h (base $k)"
  ARMSF=arms_alg_${ARM}_$k.txt; dry ${ARM}_s$k
  selfcheck alg_${ARM}_$k ${ARM}_s$k || stopf "the selfcheck of ${ARM}_s$k does not show the weights or the start checkpoint its arm names"
  start alg_${ARM}_$k arms_alg_${ARM}_$k.txt
done
say 'auto end'

#!/bin/bash
# pairing_n1dm.sh: render the two N1DM argvs (CPU), pair each against the BANKED N1D argv of its seed (strat1_check.py pair: the N1DM argv = N1D's + exactly `--aux-route modes --w-aux-route 0.5`, own --out), then three negative controls that must be REFUSED. Output dir: /root/strat1/n1dm/b2/argv
# Adapted from pairing_n1dc.sh (which ran). Exit 0 = all pass.
N=/root/strat1/n1dm; TREE=$N/tree; PY=/root/venv/bin/python; CHK=$TREE/stack/scripts/strat1_check.py; A=$N/b2/argv; mkdir -p $A
export CUDA_VISIBLE_DEVICES= PYTHONPATH=$TREE/stack PYTHONDONTWRITEBYTECODE=1
REF=/root/strat1/navrbd/runs
rc=0
for s in s0 s1; do
  tag=N1DM_$s
  $PY $N/render_argv_n1dm.py N1DM $s $A/$tag.argv.json $N/runs/$tag/train || { echo "RENDER FAIL $tag"; rc=1; continue; }
  $PY $CHK pair N1DM --argv-json $A/$tag.argv.json --ref-argv-json $REF/N1D_$s/argv.json > $A/pair_$tag.log 2>&1 && echo "PAIR PASS $tag" || { echo "PAIR FAIL $tag"; cat $A/pair_$tag.log; rc=1; }
done
# each arm has its own --out
o0=$($PY -c "import json;t=json.load(open('$A/N1DM_s0.argv.json'));print(t[t.index('--out')+1])"); o1=$($PY -c "import json;t=json.load(open('$A/N1DM_s1.argv.json'));print(t[t.index('--out')+1])")
[ -n "$o0" ] && [ -n "$o1" ] && [ "$o0" != "$o1" ] && echo "OUT distinct: $o0 | $o1" || { echo "OUT NOT DISTINCT (BAD): $o0 | $o1"; rc=1; }
# negative controls (each must be REFUSED, i.e. exit non-zero)
$PY $CHK pair N1DM --argv-json $A/N1DM_s0.argv.json --ref-argv-json $REF/N1D_s1/argv.json > $A/neg1.log 2>&1 && { echo "NEG1 NOT REFUSED (BAD)"; rc=1; } || echo "NEG1 REFUSED (seed mismatch): $(tail -c 200 $A/neg1.log)"
$PY $CHK pair N1DM --argv-json $REF/N1D_s0/argv.json --ref-argv-json $REF/N1D_s0/argv.json > $A/neg2.log 2>&1 && { echo "NEG2 NOT REFUSED (BAD)"; rc=1; } || echo "NEG2 REFUSED (N1D argv labelled N1DM, no route flags): $(tail -c 200 $A/neg2.log)"
$PY -c "import json;t=json.load(open('$A/N1DM_s0.argv.json'));json.dump(t+['--aux-route','modes'],open('$A/neg3.argv.json','w'))"
$PY $CHK pair N1DM --argv-json $A/neg3.argv.json --ref-argv-json $REF/N1D_s0/argv.json > $A/neg3.log 2>&1 && { echo "NEG3 NOT REFUSED (BAD)"; rc=1; } || echo "NEG3 REFUSED (--aux-route named twice): $(tail -c 200 $A/neg3.log)"
# NEG4: the N1DC flag named beside the route flags; NEG5: the route weight moved off the card's 0.5 (both must be REFUSED)
$PY -c "import json;t=json.load(open('$A/N1DM_s0.argv.json'));json.dump(t+['--o-code-skip','on'],open('$A/neg4.argv.json','w'))"
$PY $CHK pair N1DM --argv-json $A/neg4.argv.json --ref-argv-json $REF/N1D_s0/argv.json > $A/neg4.log 2>&1 && { echo "NEG4 NOT REFUSED (BAD)"; rc=1; } || echo "NEG4 REFUSED (--o-code-skip named): $(tail -c 200 $A/neg4.log)"
$PY -c "import json;t=json.load(open('$A/N1DM_s0.argv.json'));t[t.index('--w-aux-route')+1]='1.0';json.dump(t,open('$A/neg5.argv.json','w'))"
$PY $CHK pair N1DM --argv-json $A/neg5.argv.json --ref-argv-json $REF/N1D_s0/argv.json > $A/neg5.log 2>&1 && { echo "NEG5 NOT REFUSED (BAD)"; rc=1; } || echo "NEG5 REFUSED (route weight 1.0, not 0.5): $(tail -c 200 $A/neg5.log)"
# NEG6: an N1DH-style argv (--aux-route cond) labelled N1DM; NEG7: the reference argv itself names the modes flags -- both must be REFUSED
$PY -c "import json;t=json.load(open('$A/N1DM_s0.argv.json'));t[t.index('--aux-route')+1]='cond';json.dump(t,open('$A/neg6.argv.json','w'))"
$PY $CHK pair N1DM --argv-json $A/neg6.argv.json --ref-argv-json $REF/N1D_s0/argv.json > $A/neg6.log 2>&1 && { echo "NEG6 NOT REFUSED (BAD)"; rc=1; } || echo "NEG6 REFUSED (--aux-route cond, the N1DH arm): $(tail -c 200 $A/neg6.log)"
$PY $CHK pair N1DM --argv-json $A/N1DM_s0.argv.json --ref-argv-json $A/N1DM_s0.argv.json > $A/neg7.log 2>&1 && { echo "NEG7 NOT REFUSED (BAD)"; rc=1; } || echo "NEG7 REFUSED (the reference names the modes flags): $(tail -c 200 $A/neg7.log)"
exit $rc

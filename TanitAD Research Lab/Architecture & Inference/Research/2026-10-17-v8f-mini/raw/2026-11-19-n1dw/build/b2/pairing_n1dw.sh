#!/bin/bash
# pairing_n1dw.sh: render the four N1DW argvs (CPU), pair each against the BANKED N1D argv of its seed, then two negative controls that must be REFUSED. Output dir: /root/strat1/n1dw/b2/argv
N=/root/strat1/n1dw; TREE=$N/tree; PY=/root/venv/bin/python; CHK=$TREE/stack/scripts/strat1_check.py; A=$N/b2/argv; mkdir -p $A
export CUDA_VISIBLE_DEVICES= PYTHONPATH=$TREE/stack PYTHONDONTWRITEBYTECODE=1
rc=0
for arm in N1DW4:s0 N1DW4:s1 N1DW2:s0 N1DW2:s1; do
  k=${arm%%:*}; s=${arm##*:}; tag=${k}_$s
  $PY $N/b2/render_argv_n1dw.py $k $s $A/$tag.argv.json $N/runs/$tag/train || { echo "RENDER FAIL $tag"; rc=1; continue; }
  $PY $CHK pair $k --argv-json $A/$tag.argv.json --ref-argv-json /root/strat1/navrbd/runs/N1D_$s/argv.json > $A/pair_$tag.log 2>&1 && echo "PAIR PASS $tag" || { echo "PAIR FAIL $tag"; cat $A/pair_$tag.log; rc=1; }
done
# negative controls (each must be REFUSED, i.e. exit non-zero)
$PY $CHK pair N1DW4 --argv-json $A/N1DW4_s0.argv.json --ref-argv-json /root/strat1/navrbd/runs/N1D_s1/argv.json > $A/neg1.log 2>&1 && { echo "NEG1 NOT REFUSED (BAD)"; rc=1; } || echo "NEG1 REFUSED (seed mismatch): $(tail -c 200 $A/neg1.log)"
$PY $CHK pair N1DW2 --argv-json $A/N1DW4_s0.argv.json --ref-argv-json /root/strat1/navrbd/runs/N1D_s0/argv.json > $A/neg2.log 2>&1 && { echo "NEG2 NOT REFUSED (BAD)"; rc=1; } || echo "NEG2 REFUSED (w=4 argv labelled N1DW2): $(tail -c 200 $A/neg2.log)"
# a third: the unweighted N1D argv labelled N1DW4 (no flag)
$PY $CHK pair N1DW4 --argv-json /root/strat1/navrbd/runs/N1D_s0/argv.json --ref-argv-json /root/strat1/navrbd/runs/N1D_s0/argv.json > $A/neg3.log 2>&1 && { echo "NEG3 NOT REFUSED (BAD)"; rc=1; } || echo "NEG3 REFUSED (N1D argv labelled N1DW4): $(tail -c 200 $A/neg3.log)"
exit $rc

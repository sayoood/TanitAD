#!/bin/bash
# MM: seed-matched R2 repeat for the s1 arms. Waits (cap 8 h) until the 4th N1DW arm has trained, then, under the GPU queue lock,
# runs two N1D_s1 selfcheck-only repeats and compares N1DW4_s1 and N1DW2_s1 against them with the unchanged b2/r2_compare.py.
# Output: /root/strat1/n1dw/b2/r2s1_compare.out, r2s1_result.json. Log: /root/strat1/logs/n1dw_r2s1.log ; last line ZZ N1DW R2S1 DONE ZZ | ZZ N1DW R2S1 FAIL ZZ <why>
N=/root/strat1/n1dw; B=$N/b2; L=/root/strat1/logs/n1dw_r2s1.log
fail() { echo "ZZ N1DW R2S1 FAIL ZZ $1" >> $L; exit 1; }
for i in $(seq 1 480); do [ -e $N/runs/N1DW2_s1/ARM_DONE ] && break; tail -n1 /root/strat1/logs/n1dw.log | grep -q "ZZ N1DW CHAIN FAIL ZZ" && fail "chain failed before the 4th arm"; sleep 60; done
[ -e $N/runs/N1DW2_s1/ARM_DONE ] || fail "4th arm not trained within 8 h"
echo "$(date -u +%FT%TZ) 4th arm trained; waiting for the GPU lock" >> $L
cd $N
flock /root/gpu_queue.lock bash -c "/root/venv/bin/python $B/r2s1_run.py A >> $L 2>&1 && /root/venv/bin/python $B/r2s1_run.py B >> $L 2>&1" || fail "repeat run rc=$?"
/root/venv/bin/python $B/r2_compare.py $B/r2s1_result.json $B/r2s1/N1D_A/selfcheck.json $B/r2s1/N1D_B/selfcheck.json \
  N1DW4_s1=$N/runs/N1DW4_s1/gate/selfcheck.json N1DW2_s1=$N/runs/N1DW2_s1/gate/selfcheck.json > $B/r2s1_compare.out 2>&1
echo "$(date -u +%FT%TZ) compare rc=$? (1 = some term moved more than the repeat; read r2s1_compare.out)" >> $L
echo "ZZ N1DW R2S1 DONE ZZ" >> $L

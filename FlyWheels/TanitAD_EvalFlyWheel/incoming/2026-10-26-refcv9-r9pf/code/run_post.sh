#!/bin/bash
# after run_keq.sh: bars -> verdict -> manifest; judged on artifacts.  Waits for ZZKEQENDZZ (or stops on ZZKEQNORECZZ).
W=C:/Users/Admin/refcv8_pf/refcv9_r9pf
until grep -q "ZZKEQENDZZ\|ZZKEQNORECZZ" $W/logs/run_keq.log; do sleep 30; done
if grep -q "ZZKEQNORECZZ" $W/logs/run_keq.log; then echo "ZZPOSTSTOPZZ roll failed" >> $W/logs/run_post.log; exit 3; fi
source $W/code/env_r9pf.sh
[ -f $W/result/R9PF_KEQ.json ] || { echo "ZZPOSTSTOPZZ no R9PF_KEQ.json" >> $W/logs/run_post.log; exit 4; }
$PY -X utf8 $W/code/a3_bars.py > $W/logs/a3_bars.log 2>&1
[ -f $W/result/R9PF_PA_BARS.json ] || { echo "ZZPOSTSTOPZZ no bars" >> $W/logs/run_post.log; exit 5; }
$PY -X utf8 $W/code/a3_verdict.py > $W/logs/a3_verdict.log 2>&1
echo "ZZPOSTDONEZZ $(date +%FT%T%z) $(grep -o 'ZZA3VERDICT_[A-Z-]*ZZ' $W/logs/a3_verdict.log)" >> $W/logs/run_post.log

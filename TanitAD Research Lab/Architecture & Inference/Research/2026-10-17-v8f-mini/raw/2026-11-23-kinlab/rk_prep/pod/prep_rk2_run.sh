#!/bin/bash
# prep_rk2_run.sh -- REVISION 2. The two CPU prep jobs that run BEFORE the lock, in order, one at a time, nice 19: (1) kin_cache_precompute.sh (both seeds), then (2) rk_gsuite.sh (G-SUITE at 1a8b11a1), only if (1) ended DONE.
# The tree prep (rk_tree_prep.sh) is run first, by hand (seconds). Log: $HERE/prep_logs/prep_run.log; last line ZZ RK-PREP DONE ZZ or ZZ RK-PREP STOP ZZ <why>.
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/rk_env.sh
mkdir -p $HERE/prep_logs
L=$HERE/prep_logs/prep_run.log
echo "$(date -u +%FT%TZ) precompute start" >> $L
nice -n 19 bash $HERE/kin_cache_precompute.sh 0 1 >> $L 2>&1
tail -n 1 $KC/precompute.log | grep -q "ZZ RK-CACHE-PRECOMPUTE DONE ZZ" || { echo "ZZ RK-PREP STOP ZZ precompute: $(tail -n 1 $KC/precompute.log)" >> $L; exit 1; }
echo "$(date -u +%FT%TZ) precompute done; G-SUITE start" >> $L
nice -n 19 bash $HERE/rk_gsuite.sh >> $L 2>&1
tail -n 1 $R/logs/rk_gsuite.log | grep -q "ZZ RK GSUITE DONE ZZ" || { echo "ZZ RK-PREP STOP ZZ gsuite: $(tail -n 1 $R/logs/rk_gsuite.log)" >> $L; exit 1; }
echo "ZZ RK-PREP DONE ZZ" >> $L

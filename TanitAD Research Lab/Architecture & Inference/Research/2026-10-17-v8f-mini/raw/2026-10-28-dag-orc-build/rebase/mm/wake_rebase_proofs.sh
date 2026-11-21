#!/usr/bin/env bash
# MM waker: exits when all four re-base proof jobs have written their end markers (or after 120 min). Script polling, not model turns.
R=/e/Projects/TanitAD-artifacts/dagorc_rebase
done4() {
  local n=0
  [ -f $R/chain/selftest_chain_result.json ] && grep -aq 'SELFTEST:' $R/chain/logs/selftest_chain_try2.out 2>/dev/null && n=$((n+1))
  grep -aq ZZLABEL-DONE $R/chain/logs/label_tests_driver.out 2>/dev/null && n=$((n+1))
  grep -aq 'ZZLANEA-DONE\|ZZLANEA-STOP' $R/read/logs/lane_a.out 2>/dev/null && n=$((n+1))
  [ -f $R/read/mutation.json ] && n=$((n+1))
  echo $n
}
for i in $(seq 1 120); do
  n=$(done4); [ "$n" -eq 4 ] && { echo "ALL 4 DONE $(date +%H:%M)"; exit 0; }
  grep -aq 'ZZLANEA-STOP' $R/read/logs/lane_a.out 2>/dev/null && { echo "LANE A STOP"; tail -3 $R/read/logs/lane_a.out; exit 0; }
  sleep 60
done
echo "WAKER TIMEOUT: $(done4) of 4 done"

#!/bin/bash
# pod: rank-0 (running) -> aug -> score, all on the POD's log share; opaque ZZ markers only
LOG=/workspace/chain_after_rank0.log
R0=/workspace/dataprep_rank0_10hz.log
base_exit=$(grep -c "ZZRANK0_RUNNER_EXIT" $R0)
base_ok=$(grep -c "ZZOK rank0" $R0)
echo "ZZCHAIN_START base_exit=$base_exit base_ok=$base_ok $(date -u +%FT%TZ)" >> $LOG
while [ "$(grep -c "ZZRANK0_RUNNER_EXIT" $R0)" -le "$base_exit" ]; do sleep 120; done
if [ "$(grep -c "ZZOK rank0" $R0)" -le "$base_ok" ]; then
  echo "ZZCHAIN_ABORT rank0 runner exited without a new ZZOK rank0 $(date -u +%FT%TZ)" >> $LOG; exit 1
fi
echo "ZZCHAIN_RANK0_DONE $(tail -n 3 $R0 | tr '\n' ' ') $(date -u +%FT%TZ)" >> $LOG
echo $$ > /workspace/chain.pgid
cd /workspace
LOGS_FILE=/workspace/pod_logs_10hz.txt REFE_SIM_HZ=10 STAGES="aug" SHARDS=7 THREADS=1 \
  bash refe-plan/code/pod_dataprep.sh >> /workspace/dataprep_aug_10hz.log 2>&1
rc=$?
echo "ZZCHAIN_AUG_EXIT $rc $(date -u +%FT%TZ)" >> $LOG
grep -q "ZZOK aug" /workspace/dataprep_aug_10hz.log || { echo "ZZCHAIN_ABORT aug not OK" >> $LOG; exit 1; }
LOGS_FILE=/workspace/pod_logs_10hz.txt REFE_SIM_HZ=10 STAGES="score" SHARDS=7 THREADS=1 \
  bash refe-plan/code/pod_dataprep.sh >> /workspace/dataprep_score_10hz.log 2>&1
echo "ZZCHAIN_SCORE_EXIT $? $(date -u +%FT%TZ)" >> $LOG

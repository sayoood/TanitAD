#!/bin/bash
# Restart the teacher-lane training relabel for SFT-4's catch-up window (PREREG_SFT4 Amendments 2-3), then stop it.
# Why: at ~19:50 UTC the OOM killer took both lane relabel MAIN processes (they carried oom_score_adj 1000 so that no
# training run is ever the victim). Their paused workers were orphans and were killed by explicit PID. The relabeller
# resumes by (key, ckpt_step), so a restart redoes nothing.
#  1. wait for ZZPDM_TRAIN_EXIT (the PDM labels keep the CPU until then);
#  2. relabel with 5 + 4 workers (the pod's quota is 7.65 CPUs; R32), nice 19, OOM-killer first;
#  3. when chain v2 logs "lane window closed", stop the relabel by explicit PID, so SFT-4 gets the pod's memory
#     (labels written after launch are not read by the run anyway).
set -u
L=/workspace/data/refe_sft2/lane_pause.log
log() { echo "$(date -u +%FT%T) $*" >> $L; }
source /workspace/teacher_env.sh
export NUPLAN_DATA_ROOT=/workspace/data REFE_NUPLAN_DB_ROOT=/workspace/data/navtrain_dbs OMP_NUM_THREADS=1
echo 1000 > /proc/self/oom_score_adj
until grep -q "ZZPDM_TRAIN_EXIT" /workspace/data/refe_sft2/pdm_train.log 2>/dev/null; do sleep 60 200>&-; done
cd /workspace/refe-sft2/refe
Q=/workspace/data/refe_sft2/train_subset_queue
O=/workspace/data/refe_sft2/train_teacher_lane
nice -n 19 $DRIVERL_EVAL_PYTHON onpolicy_relabel_teacher_lane.py --queue $Q --files 'props_r0_*' --rank 0 --out $O --workers 5 --repair-last-heading >> /workspace/data/refe_sft2/tl_train_r0.log 2>&1 200>&- &
P0=$!
nice -n 19 $DRIVERL_EVAL_PYTHON onpolicy_relabel_teacher_lane.py --queue $Q --files 'props_r1_*' --rank 1 --out $O --workers 4 --repair-last-heading >> /workspace/data/refe_sft2/tl_train_r1.log 2>&1 200>&- &
P1=$!
log "lane relabel RESTARTED (pids $P0 $P1; rows $(cat $O/*.jsonl | wc -l))"
until grep -q "lane window closed" /workspace/data/refe_sft4/chain.log 2>/dev/null || ! kill -0 $P0 $P1 2>/dev/null; do sleep 30 200>&-; done
W="$(pgrep -P $P0) $(pgrep -P $P1)"
kill $P0 $P1 2>/dev/null; sleep 3; kill $W 2>/dev/null; sleep 2; kill -9 $P0 $P1 $W 2>/dev/null
log "lane relabel STOPPED for SFT-4 (rows $(cat $O/*.jsonl | wc -l))"

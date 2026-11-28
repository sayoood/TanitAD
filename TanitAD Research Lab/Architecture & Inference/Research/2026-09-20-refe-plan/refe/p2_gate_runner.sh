#!/bin/bash
# P2 launch gate on the pod (eval/PREREG_P2.md). Waits for the PDM training relabel to finish (its ~20 GB return to the
# pool) and for >= 18 GB of cgroup headroom beside SFT-4, then runs refe/p2_gate.py as the OOM killer's FIRST choice
# (oom_score_adj 1000), so it can never take SFT-4 down (R32).
set -u
source /workspace/teacher_env.sh
echo 1000 > /proc/self/oom_score_adj
L=/workspace/data/refe_p2/gate_runner.log; mkdir -p /workspace/data/refe_p2
log() { echo "$(date -u +%FT%T) $*" >> $L; }
log "armed"
until grep -q "ZZPDM_TRAIN_EXIT" /workspace/data/refe_sft2/pdm_train.log 2>/dev/null; do sleep 60; done
MAX=$(cat /sys/fs/cgroup/memory.max)
for i in $(seq 1 240); do
  anon=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat)
  [ $((MAX - anon)) -ge 18000000000 ] && break
  sleep 60
done
log "launch: anon $anon of $MAX"
cd /workspace/refe-p2/run
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
"$DRIVERL_EVAL_PYTHON" p2_gate.py --init-from /workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt \
  --onpolicy /workspace/data/refe_sft4_smoke_sets --pdm-labels /workspace/data/refe_sft2/train_pdm \
  --out /workspace/data/refe_p2/gate > /workspace/data/refe_p2/gate.log 2>&1
echo "ZZEXIT $?" >> /workspace/data/refe_p2/gate.log
log "gate exit: $(tail -1 /workspace/data/refe_p2/gate.log)"

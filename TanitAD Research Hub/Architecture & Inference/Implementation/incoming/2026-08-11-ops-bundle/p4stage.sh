#!/bin/bash
# Pod4 staging + P1 lead-gap rerun (PI decision 2026-08-11: parallel evals on
# pod4 while pod5 grinds the W4r->W7->I4a->p8c queue).
# Source pod5 via DIRECT mapping (C56 recipe); ssh -n ALWAYS (stdin-eat trap).
P5=root@69.30.85.106
PORT=22039
SSHOPTS="-n -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i /root/.ssh/id_ed25519"
LOG=/tmp/p4stage.log
mkdir -p /workspace/data /workspace/experiments /workspace/data/p8_join

echo "[stage] corpus transfer starting $(date -u +%FT%TZ)" >> $LOG
ssh $SSHOPTS $P5 'tar -C /workspace/data -cf - physicalai-val-0c5f7dac3b11-w120-256x640cyl' | tar -C /workspace/data -xf -
echo "CORPUS_TAR_EXIT=$?" >> $LOG
N=$(ls /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl 2>/dev/null | wc -l)
echo "CORPUS_FILES=$N (expect 603)" >> $LOG
if [ "$N" != "603" ]; then echo "P4STAGE_EXIT=CORPUS_INCOMPLETE" >> $LOG; exit 1; fi

echo "[stage] ckpts+join transfer $(date -u +%FT%TZ)" >> $LOG
ssh $SSHOPTS $P5 'tar -C /workspace -cf - experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt experiments/flagship-v5f-w120-30k/probe_vocab.pt experiments/flagship-v5f-w120-30k/config.json experiments/stage-a-predictor/ckpt_stage_a.pt experiments/stage-a-predictor/probe_vocab.pt experiments/stage-a-predictor/config.json experiments/flagship_v4_anchors_dense.pt data/p8_join/combined140.jsonl' | tar -C /workspace -xf -
echo "CKPT_TAR_EXIT=$?" >> $LOG
M=$(md5sum /workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt | cut -d' ' -f1)
echo "CKPT30K_MD5=$M (expect 0b8c92356ce57cb6c7081a35eb2e0559)" >> $LOG
if [ "$M" != "0b8c92356ce57cb6c7081a35eb2e0559" ]; then echo "P4STAGE_EXIT=MD5_MISMATCH" >> $LOG; exit 1; fi

cd /workspace/TanitAD || { echo "P4STAGE_EXIT=NO_CHECKOUT" >> $LOG; exit 1; }
git fetch origin claude/tanitad-resumption-handoff-92zx39 >> $LOG 2>&1
git checkout -B claude/tanitad-resumption-handoff-92zx39 origin/claude/tanitad-resumption-handoff-92zx39 >> $LOG 2>&1
if ! grep -q "lookup_classes" stack/scripts/probe_latent_state.py; then
  echo "P4STAGE_EXIT=SYNC_FAILED_FILTER_MISSING" >> $LOG; exit 1
fi
echo "STACK_SYNC_OK $(git rev-parse --short HEAD)" >> $LOG

cd /workspace/TanitAD/stack
export PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6
mkdir -p /workspace/experiments/p1-rerun-clsfilter
python3 -u scripts/probe_latent_state.py \
  --ckpt /workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 \
  --join-file /workspace/data/p8_join/combined140.jsonl \
  --out /workspace/experiments/p1-rerun-clsfilter >> /tmp/p1_rerun.log 2>&1
echo "P1RERUN_EXIT=$?" >> /tmp/p1_rerun.log
echo "P4STAGE_EXIT=done" >> $LOG

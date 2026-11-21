#!/bin/bash
# flagship-v4 30k gate — MODE B, both goal modes, SEQUENTIAL (one GPU on this pod).
# Card: Project Steering/Gates/flagship-v4-30k.card.json
set -x
export OMP_NUM_THREADS=8          # BLAS oversubscription measured a 9x slowdown
export MKL_NUM_THREADS=8
export PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts:/root/taniteval

CK=/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt
AN=/root/models/flagship-v4-fromscratch-15k/flagship_v4_anchors_dense.pt
VC=/root/valdata/physicalai-val-0c5f7dac3b11
RD=/workspace/_v4gate/results
mkdir -p "$RD"
cd /root/TanitAD/stack/scripts

echo "=========== ORACLE (the primary surface; comparable to 10k/15k history) ==========="
python3 -u eval_flagship_v4.py \
  --ckpt "$CK" --anchors-dense "$AN" --val-cache "$VC" \
  --goal-mode oracle \
  --key flagship-v4-fromscratch-30k-oracle \
  --results-dir "$RD" \
  --out /workspace/_v4gate/gate30k_oracle.json
echo "ORACLE_EXIT=$?"

echo "=========== PRODUCED (goal_head fed back; no future) ==========="
python3 -u eval_flagship_v4.py \
  --ckpt "$CK" --anchors-dense "$AN" --val-cache "$VC" \
  --goal-mode produced \
  --key flagship-v4-fromscratch-30k-produced \
  --results-dir "$RD" \
  --out /workspace/_v4gate/gate30k_produced.json
echo "PRODUCED_EXIT=$?"
echo "ALL_DONE"

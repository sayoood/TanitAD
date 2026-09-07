#!/bin/sh
# INFERENCE-RUN NOISE FLOOR for refcv5-v2's DDIM sampler.
# Same checkpoint, same flags, ONLY --infer-seed moves. Any separated
# difference between two of these rolls is a FALSE POSITIVE by construction.
# The checkpoint is step 15,000 of 40,284 -- MID-TRAINING.
# The LEVEL is NOT a capability number and must never be quoted as one.
# The quantity of interest is the SPREAD ACROSS SEEDS.
set -e
R=/c/Users/Admin/refcv5cmp
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
K=$R/ckpt_v2_15k/ckpt_15000.pt
C=$R/ckpt_v2_15k/config.json
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
export PYTHONPATH="C:\Users\Admin\refcv5cmp\repo\stack;C:\Users\Admin\refcv5cmp\repo\taniteval;C:\Users\Admin\refcv5cmp\repo"

echo "== sync repo off G: =="
"$PY" "$R/sync_repo.py"

for S in 0 1 2; do
  T="v2-15k-inferseed$S"
  echo "== ROLL $T  $(date -u +%H:%M:%SZ) =="
  ( cd "$R/repo" && "$PY" taniteval/tools/refcv3_arm.py \
      --ckpt "$K" --config "$C" \
      --episodes "$R/data/eval" \
      --labels "$R/data/s2_labels_v7.2_eval.jsonl.gz" \
      --nav-source v72 --grid 2s --action-units steer --device cuda \
      --window-stride 5 --with-oracle-sel \
      --lead-block "$R/data/b1_eval_lead_block.npz" \
      --n-boot 2000 --seed 0 --infer-seed "$S" \
      --dump-dir "$R/out/${T}_dump" --out "$R/out/${T}.json" \
      --tiers "os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0" ) \
    > "$R/out/${T}_roll.log" 2>&1
  echo "   done $T $(date -u +%H:%M:%SZ)  json=$(stat -c %s "$R/out/${T}.json" 2>/dev/null)"
done
echo "ALL THREE ROLLS DONE $(date -u +%H:%M:%SZ)"

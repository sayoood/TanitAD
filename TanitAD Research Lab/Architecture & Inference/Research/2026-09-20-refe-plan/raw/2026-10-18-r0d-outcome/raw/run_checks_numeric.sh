#!/bin/bash
source /workspace/teacher_env.sh >/dev/null 2>&1
R=/workspace/data/rls1; PKG=/workspace/refe-plan-rls1
VAL_TSV=$PKG/raw/rl_pool/partition_runs/A_no_trained_list_seed20261012_n16000/rl_val_scenes.tsv
SC=$R/arms/R0d/scores; O=/workspace/ops/r0d_read/raw
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd $PKG/refe
for U in 250 500 750; do
  nice -n 10 $DRIVERL_EVAL_PYTHON $PKG/eval/rl_val_eval.py --pdm $R/pdm_val --terms $R/terms_val --sets $R/sets_val --rl-val-scenes $VAL_TSV --require-full --rule navsim_v1_lane \
    --scores A=$SC/scores_update${U}_A.npz B=$SC/scores_update${U}_B.npz --base A --out $O/rlval_B_vs_ownA_u${U}.json > $O/rlval_B_vs_ownA_u${U}.log 2>&1
  echo "u$U rc=$?"
done
echo CHECKS_DONE

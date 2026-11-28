#!/bin/bash
# SO-G read for R0d: B vs own A (bar G input) + the chain's own vs-base read (which the K3 kill pre-empted). CPU only.
source /workspace/teacher_env.sh >/dev/null 2>&1
R=/workspace/data/rls1; PKG=/workspace/refe-plan-rls1
VAL_TSV=$PKG/raw/rl_pool/partition_runs/A_no_trained_list_seed20261012_n16000/rl_val_scenes.tsv
ARM=$R/arms/R0d; SC=$ARM/scores
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
RV="nice -n 10 $DRIVERL_EVAL_PYTHON $PKG/eval/rl_val_eval.py --pdm $R/pdm_val --terms $R/terms_val --sets $R/sets_val --rl-val-scenes $VAL_TSV --require-full --rule navsim_v1_lane"
cd $PKG/refe
echo "START $(date -u +%FT%TZ) B-vs-ownA"
env -u HF_TOKEN $RV --scores A=$SC/scores_update1000_A.npz B=$SC/scores_update1000_B.npz --base A --out $ARM/rlval_B_vs_ownA.json > $ARM/rlval_B_vs_ownA.log 2>&1
echo "rc_B_vs_ownA=$? $(date -u +%FT%TZ)"
env -u HF_TOKEN $RV --scores A0=$SC/scores_update1000_base.npz CTRL=$SC/scores_update1000_A.npz R=$SC/scores_update1000_B.npz --base A0 --out $ARM/rlval_vs_A0.json > $ARM/rlval.log 2>&1
echo "rc_vs_A0=$? $(date -u +%FT%TZ)"
echo ALLDONE

#!/usr/bin/env bash
# Re-run every smoke/proof on the FINAL code (v7f_r3). Baseline-tree outputs are unchanged (that tree never changed).
set -u
R=C:/Users/Admin/v7f_r3
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES="-1" PYTHONPATH=$R/stack
cd $R
TINY="--in-channels 3 --frame-h 32 --frame-w 32 --patch 16 --enc-dim 32 --enc-depth 1 --enc-heads 2 --readout-grid 4 --readout-dim 8 --pred-dim 32 --pred-depth 1 --pred-heads 2 --window 4 --horizons 1 --d-tac 32 --d-str 16 --d-goal-embed 16 --adapter-hidden 32 --n-candidates 3 --sigreg-slices 8 --dry-steps 3 --dry-batch 4 --dry-k 12"
EVAL=C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz
S=$R/proof/out
echo "== 1. cross-process OFF proof (modified tree)"
OMP_NUM_THREADS=1 PROOF_TREE=$R $PY proof/bit_identity_off.py $S/batches.pt $S/off_v7f_r3.json
echo "== 2. seeded CLI OFF dry-runs (modified tree)"
rm -rf $S/cli_off/v7f_r3
for ST in S-W S-T S-S S-J; do
  OMP_NUM_THREADS=1 $PY proof/seeded_main.py stack/scripts/train_v6_staged.py --stage $ST --dry-run --out $S/cli_off/v7f_r3/$ST $TINY > /dev/null 2>&1; echo "  $ST exit=$?"
done
echo "== 3. CLI ON/OFF dry-runs and refusals"
rm -rf $S/dry_on_ST $S/dry_on_SJ $S/dry_off_ST $S/dry_on_SW $S/dry_on_noml
OMP_NUM_THREADS=2 $PY stack/scripts/train_v6_staged.py --stage S-T --dry-run --out $S/dry_on_ST $TINY --w-tac-label-all 1 --goal-multilabel --s2-labels $EVAL > raw/cli_dry_on_ST.log 2>&1; echo "  ON S-T exit=$?"
OMP_NUM_THREADS=2 $PY stack/scripts/train_v6_staged.py --stage S-J --dry-run --out $S/dry_on_SJ $TINY --w-tac-label-all 1 --goal-multilabel --s2-labels $EVAL > raw/cli_dry_on_SJ.log 2>&1; echo "  ON S-J exit=$?"
OMP_NUM_THREADS=2 $PY stack/scripts/train_v6_staged.py --stage S-T --dry-run --out $S/dry_off_ST $TINY > raw/cli_dry_off_ST.log 2>&1; echo "  OFF S-T exit=$?"
OMP_NUM_THREADS=2 $PY stack/scripts/train_v6_staged.py --stage S-W --dry-run --out $S/dry_on_SW $TINY --w-tac-label-all 1 --goal-multilabel > raw/cli_refusal_on_SW.log 2>&1; echo "  REFUSAL S-W exit=$?"
OMP_NUM_THREADS=2 $PY stack/scripts/train_v6_staged.py --stage S-T --dry-run --out $S/dry_on_noml $TINY --w-tac-label-all 1 > raw/cli_refusal_no_multilabel.log 2>&1; echo "  REFUSAL no-multilabel exit=$?"
echo "== 4. real eval-label smoke"
OMP_NUM_THREADS=2 PROOF_TREE=$R $PY proof/smoke_real_eval_labels.py $EVAL $S/smoke_real_eval_labels.json > raw/smoke_real_eval_labels.log 2>&1; echo "  exit=$?"
echo "== 5. real train() ladder S-W -> S-T ON/OFF"
GEO="--frame-h 256 --frame-w 640 --patch 16 --enc-dim 32 --enc-depth 1 --enc-heads 2 --readout-grid 4 --readout-dim 8 --pred-dim 32 --pred-depth 1 --pred-heads 2 --window 4 --horizons 1 --d-tac 32 --d-str 16 --d-goal-embed 16 --adapter-hidden 32 --n-candidates 3 --sigreg-slices 8"
DATA="--v2-cache $S/cache3 --no-require-parity --exclude-eval-clips none --allow-eval-clips-in-train --i-know-this-arm-predates-nav --device cpu --no-amp"
rm -rf $S/train_smoke
OMP_NUM_THREADS=2 $PY proof/seeded_main.py stack/scripts/train_v6_staged.py --stage S-W --out $S/train_smoke/SW $GEO $DATA --steps 2 --batch 2 --eps-per-batch 2 --log-every 1 --save-every 2 --o1-k 10 --o5-k 12 > raw/train_smoke_SW.log 2>&1; echo "  S-W exit=$?"
SW=$S/train_smoke/SW
LADDER="--init-from $SW/ckpt.pt --prev-gate $SW/stage_gate.json --allow-inconclusive-gate --gate-off-reason CPU_plumbing_smoke_of_R3_v7f_r3_not_a_trained_stage"
RUN="--steps 6 --batch 4 --eps-per-batch 3 --log-every 1 --save-every 6 --o1-k 10 --o5-k 12 --goal-multilabel"
OMP_NUM_THREADS=2 $PY proof/seeded_main.py stack/scripts/train_v6_staged.py --stage S-T --out $S/train_smoke/ST_on $GEO $DATA $LADDER $RUN --s2-labels $EVAL --w-tac-label-all 1 > raw/train_smoke_ST_on.log 2>&1; echo "  S-T ON exit=$?"
OMP_NUM_THREADS=2 $PY proof/seeded_main.py stack/scripts/train_v6_staged.py --stage S-T --out $S/train_smoke/ST_off $GEO $DATA $LADDER $RUN > raw/train_smoke_ST_off.log 2>&1; echo "  S-T OFF exit=$?"
echo "== done"

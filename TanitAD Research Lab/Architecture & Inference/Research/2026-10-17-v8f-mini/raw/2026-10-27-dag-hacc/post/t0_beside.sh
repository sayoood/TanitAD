#!/bin/bash
# BESIDE (PREREG section 2): the four families at T0 for DAG-HACC. The eval harness is run with DAG's own fresh-age T0 command (dagt1/logs/chain.out "start fresh-T0 eval M2b-dag_s0-fresh"),
# only --ckpt / --tag / --out-dir changed. Plain DAG's banked fresh eval is the comparator. Then t0_beside.py pairs them.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PK=E:/Projects/TanitAD-artifacts/daghacc
ART=E:/Projects/TanitAD-artifacts
TREE=$ART/dagt1_tree_b43c7a9a
RAW="$TREE/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TREE/stack")
export TMP=$PK/pt_tmp TEMP=$PK/pt_tmp
mkdir -p $PK/t0
for S in 0 1; do
  $PY $TREE/stack/scripts/v8f_mini_eval.py run --ckpt $PK/ckpt/M2b_s${S}_dag_hacc.pt --tag M2b-daghacc_s${S}-fresh --ages fresh \
    --v9-eval $ART/sfresh_inputs/inputs/v9/v9_labels_eval139.npz --lead-eval $ART/sfresh_inputs/inputs/v9/lead_eval139.npz --speed-eval $ART/sfresh_inputs/inputs/v9/refcv6_speed_max_v8_eval.jsonl \
    --kin-mode yaw --p10 $ART/sfresh_inputs/inputs/p10/p10_prior.pt --out-dir $PK/t0 --eval-cache $ART/sfresh_inputs/cache --tracks $ART/sfresh_inputs/inputs/native_tracks.pt \
    --eval-grid "$RAW/eval_grid_windows.json" --eval-kin "$RAW/eval141_kin.npz" --eval-join "$RAW/join_eval141.npz" --project --families > $PK/logs/t0_daghacc_s$S.out 2>&1 < /dev/null
  echo "t0 s$S rc=$?" >> $PK/logs/t0_daghacc.done
done
$PY -I $PK/post/t0_beside.py > $PK/logs/t0_beside.out 2>&1 < /dev/null
echo "ZZT0-DONE rc=$?" >> $PK/logs/t0_daghacc.done

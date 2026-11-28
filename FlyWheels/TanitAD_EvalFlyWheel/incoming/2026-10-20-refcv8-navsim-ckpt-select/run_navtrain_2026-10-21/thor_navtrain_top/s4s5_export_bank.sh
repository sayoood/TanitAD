#!/bin/bash
# S4 (export AgentInputs, navsim-cpu venv) then S5 (416x1024 frame bank, tanitad-train venv). Serial. End marker QQS45END.
set -u
R=/home/nvidia/r8ns; C=$R/code; N=$R/navtrain; X=$C/navtrain
NAVSIM_PY=/home/nvidia/venvs/navsim-cpu/bin/python; TRAIN_PY=/home/nvidia/venvs/tanitad-train/bin/python
V11=/home/nvidia/navsim-thor/src/navsim-v11; W3=/home/nvidia/navsim-thor/code/w3
YAML=$V11/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml
SAMPLE=$N/NAVTRAIN_SAMPLE_TOKENS.json; INP=$N/inputs/navtrain_sample_inputs.json.gz; BANK=$N/frame_bank; DL=$R/navtrain_dl
export PYTHONDONTWRITEBYTECODE=1
t0=$(date +%s)
cd $X && PYTHONPATH=$V11:/home/nvidia/navsim-thor/code/stubs $NAVSIM_PY export_navtrain_inputs.py --sample $SAMPLE --logs-root $N/navsim_logs/trainval \
    --navsim-tree $V11 --w3-code $W3 --filter-yaml $YAML --out $INP --manifest $N/inputs/manifest.json > $N/s4_export.out 2>&1
t1=$(date +%s); echo "QQS4END secs=$((t1-t0))" >> $N/s4_export.out
grep -q "ZZEXPORT_DONE" $N/s4_export.out || { echo "QQS45END s4_not_done"; exit 4; }
cd $X && TANITAD_REPO=$C/tree PYTHONPATH=$C/tree/stack:$C/tree/taniteval OMP_NUM_THREADS=6 \
  $TRAIN_PY build_navtrain_frames.py --sample $SAMPLE --inputs $INP --shard $DL/navtrain_current_1.tgz --shard $DL/navtrain_history_1.tgz \
    --logs-root $N/navsim_logs/trainval --bank $BANK --threads 6 \
    --frames416 $C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code/frames416.py > $N/s5_bank.out 2>&1
t2=$(date +%s); echo "QQS5END secs=$((t2-t1))" >> $N/s5_bank.out
echo "QQS45END s4=$((t1-t0)) s5=$((t2-t1))"

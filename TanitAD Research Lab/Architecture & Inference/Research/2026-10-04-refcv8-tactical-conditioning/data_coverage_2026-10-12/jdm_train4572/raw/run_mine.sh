#!/bin/bash
# usage: run_mine.sh <tag> <train_manifest> <out_json>
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2
R=C:/Users/Admin/tanitad-data/add203/repo
H=C:/Users/Admin/qland/hold_jdm4572/raw
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY -c "import sys; sys.path.insert(0,'$R/stack'); import tanitad.data.join_label_hygiene as H; print('hygiene file', H.__file__)"
$PY $R/stack/scripts/mine_join_label_defects.py --join $H/b1_train_plus_eval_agents_3d.jsonl.xz \
  --manifest "$2" --manifest C:/Users/Admin/tanitad-data/add203/in/eval139_v2manifest.pt --out "$3" --progress 200000
echo "MINE_${1}_DONE_EXIT_$?"

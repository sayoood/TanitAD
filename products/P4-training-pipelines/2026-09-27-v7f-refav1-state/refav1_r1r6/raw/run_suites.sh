#!/usr/bin/env bash
# Run the refav1 test sets on the R1-R6 tree (and the extra set on the pristine c36b6ddd tree).
R=C:/Users/Admin/refav1_r1r6
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="-1"
export TANITAD_V72_TRAIN_LABELS="$R/data/s2_labels_v7.2_train.jsonl.gz"
# (A) the brief's set: test_refa_v1*.py + test_refav1*.py (37 pre-existing + 6 new) on the NEW tree
cd $R/stack
PYTHONPATH="$R/stack;$R/taniteval" $PY -m pytest -q -rs -p no:cacheprovider tests/test_refa_v1*.py tests/test_refav1*.py > $R/raw/pytest_A_refav1_set_NEW.txt 2>&1
echo "EXIT=$?" >> $R/raw/pytest_A_refav1_set_NEW.txt
# (B) every OTHER stack test that mentions refav1, NEW tree then PRISTINE tree
PYTHONPATH="$R/stack;$R/taniteval" $PY -m pytest -q -rs -p no:cacheprovider $(cat $R/raw/tests_other_refav1_mentions.txt) > $R/raw/pytest_B_other_refav1_mentions_NEW.txt 2>&1
echo "EXIT=$?" >> $R/raw/pytest_B_other_refav1_mentions_NEW.txt
cd $R/stack_base_c36b6ddd_pristine
PYTHONPATH="$R/stack_base_c36b6ddd_pristine;$R/taniteval" $PY -m pytest -q -rs -p no:cacheprovider $(cat $R/raw/tests_other_refav1_mentions.txt) > $R/raw/pytest_B_other_refav1_mentions_PRISTINE.txt 2>&1
echo "EXIT=$?" >> $R/raw/pytest_B_other_refav1_mentions_PRISTINE.txt
# (C) the taniteval tests that mention refav1, NEW tree
cd $R/taniteval
PYTHONPATH="$R/stack;$R/taniteval" $PY -m pytest -q -rs -p no:cacheprovider tests/test_four_families_lateral_undefined.py tests/test_leaderboard.py tests/test_refav1_components_yaw_mask.py > $R/raw/pytest_C_taniteval_refav1_NEW.txt 2>&1
echo "EXIT=$?" >> $R/raw/pytest_C_taniteval_refav1_NEW.txt

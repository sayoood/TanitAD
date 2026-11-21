#!/usr/bin/env bash
# FINAL evidence run on the final R1-R6 tree (CPU only). Each step writes its own log in raw/.
R=C:/Users/Admin/refav1_r1r6
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="-1"
export TANITAD_V72_TRAIN_LABELS="$R/data/s2_labels_v7.2_train.jsonl.gz"
cd $R
# 1. cross-version bit identity: the pristine dump is regenerated too (same script, same inputs)
rm -rf $R/tmp_bitident && mkdir -p $R/tmp_bitident
PYTHONPATH=$R/stack_base_c36b6ddd_pristine $PY raw/bitident/bitident_dump.py --stack $R/stack_base_c36b6ddd_pristine --out $R/tmp_bitident/base.pt --workdir $R/tmp_bitident/base_runs > raw/bitident/dump_base.log 2>&1
echo "EXIT=$?" >> raw/bitident/dump_base.log
PYTHONPATH=$R/stack $PY raw/bitident/bitident_dump.py --stack $R/stack --out $R/tmp_bitident/new.pt --workdir $R/tmp_bitident/new_runs > raw/bitident/dump_new.log 2>&1
echo "EXIT=$?" >> raw/bitident/dump_new.log
$PY raw/bitident/bitident_compare.py $R/tmp_bitident/base.pt $R/tmp_bitident/new.pt > raw/bitident/compare.out.txt 2>&1
echo "EXIT=$?" >> raw/bitident/compare.out.txt
# 2. the real-data known-value checks of the priors
PYTHONPATH=$R/stack $PY raw/checks/kdx_reference_check.py > raw/checks/kdx_reference_check.out.json 2>&1
PYTHONPATH=$R/stack $PY raw/checks/prior_2s_vs_6s_check.py > raw/checks/prior_2s_vs_6s_check.out.json 2>&1
# 3. the test suites on the final tree
cd $R/stack
PYTHONPATH="$R/stack;$R/taniteval" $PY -m pytest -q -rs -p no:cacheprovider tests/test_refa_v1*.py tests/test_refav1*.py > $R/raw/pytest_A_refav1_set_NEW.txt 2>&1
echo "EXIT=$?" >> $R/raw/pytest_A_refav1_set_NEW.txt
PYTHONPATH="$R/stack;$R/taniteval" $PY -m pytest -q -rs -p no:cacheprovider $(cat $R/raw/tests_other_refav1_mentions.txt) > $R/raw/pytest_B_other_refav1_mentions_NEW.txt 2>&1
echo "EXIT=$?" >> $R/raw/pytest_B_other_refav1_mentions_NEW.txt
cd $R/taniteval
PYTHONPATH="$R/stack;$R/taniteval" $PY -m pytest -q -rs -p no:cacheprovider tests/test_four_families_lateral_undefined.py tests/test_leaderboard.py tests/test_refav1_components_yaw_mask.py > $R/raw/pytest_C_taniteval_refav1_NEW.txt 2>&1
echo "EXIT=$?" >> $R/raw/pytest_C_taniteval_refav1_NEW.txt
PYTHONPATH="$R/stack_base_c36b6ddd_pristine;$R/taniteval" $PY -m pytest -q -rs -p no:cacheprovider tests/test_four_families_lateral_undefined.py tests/test_leaderboard.py tests/test_refav1_components_yaw_mask.py > $R/raw/pytest_C_taniteval_refav1_PRISTINE.txt 2>&1
echo "EXIT=$?" >> $R/raw/pytest_C_taniteval_refav1_PRISTINE.txt
# 4. the real-data CPU smokes of refa_v1_train.py (tiny architecture, 3 eval episodes)
cd $R
rm -rf $R/tmp_smoke_real && mkdir -p $R/tmp_smoke_real
PYTHONPATH=$R/stack $PY -X faulthandler raw/smoke/smoke_real.py --stack $R/stack --out $R/tmp_smoke_real/off > raw/smoke/smoke_real_OFF.log 2>&1
echo "EXIT=$?" >> raw/smoke/smoke_real_OFF.log
PYTHONPATH=$R/stack $PY -X faulthandler raw/smoke/smoke_real.py --stack $R/stack --out $R/tmp_smoke_real/on --steps 4 -- --bs 3 --strategic-off --vmax-input --w-goal 0.1 --goal-negatives cot-absence-negative --cot-negative-sidecar $R/raw/sidecars/cot_absence_negative_v7.2_eval.json.gz --w-speed-band 0.1 --w-traj 1.0 > raw/smoke/smoke_real_ON.log 2>&1
echo "EXIT=$?" >> raw/smoke/smoke_real_ON.log
PYTHONPATH=$R/stack $PY -X faulthandler raw/smoke/smoke_real.py --stack $R/stack --out $R/tmp_smoke_real/on_damp50 --steps 2 -- --vmax-input --w-goal 0.1 --w-speed-band 0.1 --w-traj 1.0 --r5-prior damp50 --r5-kappa-source pose_past > raw/smoke/smoke_real_ON_damp50_posepast_strategic_on.log 2>&1
echo "EXIT=$?" >> raw/smoke/smoke_real_ON_damp50_posepast_strategic_on.log
echo FINAL-DONE

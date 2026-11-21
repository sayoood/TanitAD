#!/usr/bin/env bash
# The neighbour suites of every closure-touching restart change, on the COMBINED tree (all four
# patches + the three new test files) and on the LAUNCH tree, one file at a time, JUnit XML per
# file; `diff_junit.py` then compares per-test outcomes. CPU only; OMP 2 (the dev box is shared).
# usage: run_neighbour_suites.sh <launch tree> <combined tree> <out dir>
set -u
BASE="$1"; COMBO="$2"; OUT="$3"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
FILES="test_refcv7_admitted_freeze.py test_v2_cache_pickle_state.py test_refcv6_loader_stamped_queries.py
test_grad_unreachable_declared.py test_declared_vs_built.py test_declared_freeze_preflight.py
test_grad_budget_honesty.py test_route_loss_respects_strategic_bypass.py test_refcv6_trunk.py
test_encoder_lr_multiplier_survives_the_schedule.py test_bands_and_turns.py
test_v7_vocab_reach_census.py test_refcv7_eval_loader.py test_g_box_overfit.py
test_g_box_overfit_near_lift.py test_launch_gate.py test_map_head_hires.py test_map_hires_wiring.py
test_refcv7_box_head.py test_refcv7_box_head_wiring.py test_refcv7_training.py test_refcv7_model.py
test_refcv7_vis1.py test_refc_v3_save_before_eval.py test_refcv6_grad_conflict.py
test_conflict_readings_are_logged.py test_grad_reach_logged.py test_grad_reach_census.py
test_trainer_imports_without_refcv7.py test_v2_dataset.py test_newest_frame_only.py
test_tactical_goal_underpowered_matches_census.py"
mkdir -p "$OUT"
for side in combo base; do
  if [ "$side" = combo ]; then T="$COMBO"; else T="$BASE"; fi
  mkdir -p "$OUT/$side"
  for f in $FILES; do
    if [ ! -f "$T/stack/tests/$f" ]; then echo "$(date +%T) $side $f ABSENT" >> "$OUT/run.log"; continue; fi
    s=$(date +%s)
    ( cd "$T/stack" && PYTHONPATH="$T/stack" "$PY" -m pytest -q -p no:cacheprovider \
        --junitxml="$OUT/$side/${f%.py}.xml" "tests/$f" > "$OUT/$side/${f%.py}.log" 2>&1 )
    rc=$?
    echo "$(date +%T) $side $f rc=$rc $(( $(date +%s) - s ))s $(tail -1 "$OUT/$side/${f%.py}.log")" >> "$OUT/run.log"
  done
done
echo "$(date +%T) SUITES DONE" >> "$OUT/run.log"

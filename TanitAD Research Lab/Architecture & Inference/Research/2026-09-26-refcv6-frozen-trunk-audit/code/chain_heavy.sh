#!/usr/bin/env bash
# Sequential RAM-gated chain: each stage waits until available RAM >= NEED (two polls), then runs.
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
export AUDIT_SCRATCH="${AUDIT_SCRATCH:?set AUDIT_SCRATCH}"
wait_ram() { local need=$1 maxw=$2 t=0 ok=0 a
  while [ $t -lt $maxw ]; do
    a=$($PY -c "import psutil;print('%.2f'%(psutil.virtual_memory().available/2**30))")
    if $PY -c "import sys;sys.exit(0 if float('$a')>=float('$need') else 1)"; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 2 ] && { echo "RAM_OK need=$need avail=$a after ${t}s"; return 0; }
    sleep 30; t=$((t+30)); done
  echo "RAM_TIMEOUT need=$need last=$a"; return 1; }
RAW=../raw
if wait_ram 10.2 21600; then $PY q3_hooks_forward.py > $RAW/q3_hooks_forward.log 2>&1; echo "Q3_EXIT=$?"; fi
if wait_ram 9.0 7200; then $PY q6_guards.py > $RAW/q6_guards.log 2>&1; echo "Q6_EXIT=$?"; fi
if wait_ram 10.0 7200; then
  T="$AUDIT_SCRATCH/patched_tip/stack"
  rm -rf "$T/tests"; mkdir -p "$T/tests"; cp -r C:/Users/Admin/cfull_tip/stack/tests/* "$T/tests/"; find "$T/tests" -name __pycache__ -type d -prune -exec rm -rf {} +
  cp test_refcv6_f3_cascade_reaches_loss.PROPOSED.py "$T/tests/test_refcv6_f3_cascade_reaches_loss.py"
  cd "$T" && PYTHONPATH="$T" timeout 5400 $PY -m pytest -q -p no:cacheprovider \
    tests/test_refcv6_f3_cascade_reaches_loss.py tests/test_built_heads_receive_gradient.py tests/test_cot_negative_policy.py \
    tests/test_goal_point_wiring.py tests/test_hierarchy_rung.py tests/test_refc_v3.py tests/test_refc_v3_nav_from_v7.py \
    tests/test_refc_v3_scale_matrix.py tests/test_refcv6_conflict_wiring.py tests/test_refcv6_diffusion.py \
    tests/test_refcv6_tactical.py tests/test_route_loss_respects_strategic_bypass.py tests/test_tac_goal_trainer_flag.py \
    tests/test_tac_str_labels.py tests/test_tacgoal_eval_target_wiring.py tests/test_v7_vocab_reach_census.py \
    > "$OLDPWD/$RAW/patched_suite.log" 2>&1; echo "SUITE_EXIT=$?"; cd "$OLDPWD"
fi
echo CHAIN_DONE

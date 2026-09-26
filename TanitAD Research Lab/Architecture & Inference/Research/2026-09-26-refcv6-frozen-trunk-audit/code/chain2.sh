#!/usr/bin/env bash
# A16 batch 2 -- the sequential, RAM-gated chain (brief rule 6: nothing starts below 8 GB free).
# Each stage waits until available RAM >= NEED on two consecutive polls, then runs.
#   A  the two NEW tests on the FIXED tree (must pass) and on the unmodified TIP (must go RED)
#   B  the related existing suites on both trees (a new failure on the fixed tree only = regression)
#   C  build the clip-clock sidecar for refcv6's train + eval clips (the fixed builder)
#   D  Q3 hooks + Q1 at the launched configuration (the live checkpoint, CPU)
#   E  Q6 G2 (the pretrained-weights guard's regressions)
set -u
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
: "${AUDIT_SCRATCH:?set AUDIT_SCRATCH}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RAW="$HERE/../raw"
FIX="$AUDIT_SCRATCH/fixtree/stack"
TIP="$AUDIT_SCRATCH/tiptree/stack"
NEW_TESTS="tests/test_refcv6_f3_cascade_reaches_loss.py tests/test_refcv6_label_clock.py"
RELATED="tests/test_built_heads_receive_gradient.py tests/test_cot_negative_policy.py
 tests/test_goal_point_wiring.py tests/test_hierarchy_rung.py tests/test_refc_v3.py
 tests/test_refc_v3_nav_from_v7.py tests/test_refc_v3_scale_matrix.py tests/test_refcv6_conflict_wiring.py
 tests/test_refcv6_diffusion.py tests/test_refcv6_tactical.py tests/test_route_loss_respects_strategic_bypass.py
 tests/test_tac_goal_trainer_flag.py tests/test_tac_str_labels.py tests/test_tacgoal_eval_target_wiring.py
 tests/test_v7_vocab_reach_census.py tests/test_conflict_readings_are_logged.py tests/test_refc_v3_u8_batches.py"

wait_ram() { local need=$1 maxw=$2 t=0 ok=0 a=0
  while [ $t -lt $maxw ]; do
    a=$($PY -c "import psutil;print('%.2f'%(psutil.virtual_memory().available/2**30))")
    if $PY -c "import sys;sys.exit(0 if float('$a')>=float('$need') else 1)"; then ok=$((ok+1)); else ok=0; fi
    if [ $ok -ge 2 ]; then echo "$(date +%H:%M:%S) RAM_OK need=$need avail=$a after ${t}s"; return 0; fi
    sleep 30; t=$((t+30)); done
  echo "$(date +%H:%M:%S) RAM_TIMEOUT need=$need last=$a"; return 1; }

run_pytest() {  # <tree> <log> <files...>
  local tree=$1 log=$2; shift 2
  ( cd "$tree" && PYTHONPATH="$tree" timeout 5400 $PY -m pytest -q -p no:cacheprovider "$@" ) > "$log" 2>&1
  echo "PYTEST_EXIT=$? $(tail -1 "$log")"
}

if wait_ram 9.0 21600; then
  echo "== A: new tests, FIXED tree"; run_pytest "$FIX" "$RAW/fix_new_tests_fixtree.log" $NEW_TESTS
  cp "$FIX/tests/test_refcv6_f3_cascade_reaches_loss.py" "$FIX/tests/test_refcv6_label_clock.py" "$TIP/tests/"
  echo "== A: new tests, unmodified TIP"; run_pytest "$TIP" "$RAW/fix_new_tests_tiptree.log" $NEW_TESTS
  rm -f "$TIP/tests/test_refcv6_f3_cascade_reaches_loss.py" "$TIP/tests/test_refcv6_label_clock.py"
fi
if wait_ram 9.6 21600; then
  echo "== B: related suites, FIXED tree"; run_pytest "$FIX" "$RAW/fix_related_fixtree.log" $RELATED
  echo "== B: related suites, TIP";        run_pytest "$TIP" "$RAW/fix_related_tiptree.log" $RELATED
fi
if wait_ram 8.6 7200; then
  echo "== C: clip-clock sidecar"
  ( PYTHONPATH="$FIX" $PY "$FIX/scripts/build_clip_clock_sidecar.py" \
      --cache D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139 \
      --manifest "$AUDIT_SCRATCH/thor_pull/refcv6-b1-416x1024-train___v2manifest.pt" \
      --out "$RAW/refcv6_clip_clock_sidecar.jsonl" ) > "$RAW/refcv6_clip_clock_sidecar.log" 2>&1
  echo "SIDECAR_EXIT=$? $(tail -1 "$RAW/refcv6_clip_clock_sidecar.log")"
fi
if wait_ram 10.2 21600; then
  echo "== D: Q3 hooks + Q1 as launched"
  ( cd "$HERE" && $PY q3_hooks_forward.py ) > "$RAW/q3_hooks_forward.log" 2>&1; echo "Q3_EXIT=$?"
fi
if wait_ram 8.6 7200; then
  echo "== E: Q6 G2"
  ( cd "$HERE" && $PY q6_guards.py ) > "$RAW/q6_guards.log" 2>&1; echo "Q6_EXIT=$?"
fi
echo "$(date +%H:%M:%S) CHAIN2_DONE"

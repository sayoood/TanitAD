#!/bin/bash
# Does `run_gate.py` render a COMPLETE verdict once the co-primary exists at an
# ADMISSIBLE horizon on v5's frame — and can every rule still FAIL?
#
# ⛔ The arm is `v5frame-CVREF-176x624`: a CONSTANT-VELOCITY reference policy on
# v5's 176x624 cylindrical frame. It is NOT v5 and no verdict here is a v5
# verdict. What is under test is the INSTRUMENT: that the gate consumes a
# corridor block at 20 < K <= 190 rendered through the corrected re-render, and
# that each of its rules can return a failing value.
#
# ⚠️ THE THREE CO-PRIMARY BARS ARE CHOSEN A PRIORI, NOT AFTER SEEING THE NUMBER:
#   0.99  a bar nothing can breach   -> the PASS branch must be reachable
#   0.35  the value the v4/v5 PREP card family uses
#   0.01  a bar nothing can meet     -> the FAIL branch must be reachable
# Registering a bar after reading the value is GATE_PROTOCOL 0.3's forking path;
# these are instrument probes and every card says so in its --note.
#
# K=100 is the HORIZON-HONEST floor (run_gate.HORIZON_HONEST_MIN_K); K=60 is
# admissible but carries the qualifier. Both are exercised.
set -u
ROOT=/workspace/v5gate
STACK=$ROOT/stack
GEOM=$ROOT/geom
export PYTHONPATH=$STACK:$STACK/scripts:$ROOT/taniteval
export TANITEVAL_STACK_OVERRIDE=$STACK
export OMP_NUM_THREADS=6
mkdir -p $GEOM/gates $GEOM/raw
cd "$STACK"

KEY=v5frame-CVREF-176x624
C100=$GEOM/corridor_v5frame_cv_K100.json
C60=$GEOM/corridor_v5frame_cv_K60.json
EVAL=$GEOM/eval_${KEY}_diagnostic.json
LOG=$GEOM/${KEY}_log.jsonl

VOID='{"metric":"nonav_route_beats_majority","original_threshold":">=1","status":"VOID_BY_CONSTRUCTION","adjudication":"INSTRUMENT-FAIL, NEVER MODEL-FAIL","authority":"GATE_PROTOCOL 0.7","reason":"route_target is a LOOKUP of the route input (refb_labels.route_target = _NAV_TO_ROUTE[nav_cmd]), so route_skill is 0.0 BY CONSTRUCTION and the metric measures the label bug, not the model.","re_arms_when":"a run trains with --labels-v2 and --v2-route-from-vision"}'

# ------------------------------------------------------------------ 0. inputs
# The DIAGNOSTIC primary (ade_0_2s) and the log are minted from the SAME rollout
# with the SAME estimator (episode_cluster_bootstrap), so the card's demoted
# metric is a real number about the same policy on the same windows — not
# another arm's ADE borrowed to make `check` proceed.
python3 -u $GEOM/mint_gate_inputs.py --perwindow \
  $GEOM/corridor_v5frame_cv_K100_perwindow_K100.pt \
  --run $KEY --eval-out $EVAL --log-out $LOG
echo "MINT_EXIT=$?"

echo
echo "############ 1. register K=20  -> MUST BE REFUSED (blind horizon) #######"
python3 -u scripts/run_gate.py register --run $KEY --gate-step 0 \
  --primary-metric ade_0_2s --primary-threshold 0.60 \
  --co-primary-threshold 0.35 --co-primary-horizon-K 20 \
  --card "$GEOM/gates/$KEY-K20.card.json" 2>&1 | tail -6
echo "EXIT_K20=${PIPESTATUS[0]}"

echo
echo "############ 2. register K=200 -> MUST BE REFUSED (above ceiling) ######"
python3 -u scripts/run_gate.py register --run $KEY --gate-step 0 \
  --primary-metric ade_0_2s --primary-threshold 0.60 \
  --co-primary-threshold 0.35 --co-primary-horizon-K 200 \
  --card "$GEOM/gates/$KEY-K200.card.json" 2>&1 | tail -6
echo "EXIT_K200=${PIPESTATUS[0]}"

echo
echo "############ 3. register the ADMISSIBLE cards ###########################"
for K in 100 60; do
for T in 0.99 0.35 0.01; do
  TAG=$(echo $T | tr -d '.')
  rm -f "$GEOM/gates/$KEY-K$K-T$TAG.card.json"
  python3 -u scripts/run_gate.py register --run $KEY --gate-step 0 \
    --primary-metric ade_0_2s --primary-threshold 0.60 \
    --co-primary-threshold $T --co-primary-horizon-K $K \
    --co-primary-junction-threshold 0.50 \
    --secondary "wm_canary_ade_2s<=0.55" --secondary "miss_2m<=0.10" \
    --secondary-void "$VOID" \
    --lever-family encoder-geometry --restarts-used 0 \
    --note "INSTRUMENT PROBE, not a v5 gate. Arm is a constant-velocity reference policy on the 176x624 cylindrical frame; bar chosen a priori to exercise the rule." \
    --card "$GEOM/gates/$KEY-K$K-T$TAG.card.json" 2>&1 | tail -2
  echo "REGISTERED K=$K T=$T exit=${PIPESTATUS[0]}"
done
done

run_check () {   # $1 out-tag  $2 card-tag  $3.. extra args
  local tag="$1"; shift
  local ct="$1";  shift
  python3 -u scripts/run_gate.py check \
    --card "$GEOM/gates/$KEY-$ct.card.json" --log "$LOG" \
    --eval-json "$EVAL" "$@" \
    --json "$GEOM/raw/gate_check_$tag.json" 2>&1 | tail -55
  echo "EXIT_$tag=${PIPESTATUS[0]}"
}

echo
echo "##### 4. K=100 card + K=100 corridor + bar 0.99 -> COMPLETE, HORIZON-HONEST"
run_check K100_PASS K100-T099 --corridor-json "$C100" \
  --secondary-value wm_canary_ade_2s=0.42 miss_2m=0.05

echo
echo "##### 5. THE CO-PRIMARY FALSIFIER: bar 0.01 -> must FAIL ################"
run_check K100_FALSIFIER K100-T001 --corridor-json "$C100" \
  --secondary-value wm_canary_ade_2s=0.42 miss_2m=0.05

echo
echo "##### 6. the PREP-family bar 0.35 at K=100 #############################"
run_check K100_T035 K100-T035 --corridor-json "$C100" \
  --secondary-value wm_canary_ade_2s=0.42 miss_2m=0.05

echo
echo "##### 7. SAME card, NO corridor -> must render INCOMPLETE ###############"
run_check K100_NO_CORRIDOR K100-T099 \
  --secondary-value wm_canary_ade_2s=0.42 miss_2m=0.05

echo
echo "##### 8. SECONDARY FALSIFIER (breached values) ##########################"
run_check K100_SECONDARY_FAIL K100-T099 --corridor-json "$C100" \
  --secondary-value wm_canary_ade_2s=9.99 miss_2m=0.99

echo
echo "##### 9. a K=60 corridor against the K=100 card -> must be REFUSED ######"
run_check K100_WITH_K60_CORRIDOR K100-T099 --corridor-json "$C60" \
  --secondary-value wm_canary_ade_2s=0.42 miss_2m=0.05

echo
echo "##### 10. K=60 card + K=60 corridor -> COMPLETE but NOT horizon-honest ##"
run_check K60_PASS K60-T099 --corridor-json "$C60" \
  --secondary-value wm_canary_ade_2s=0.42 miss_2m=0.05

echo ALL_DONE

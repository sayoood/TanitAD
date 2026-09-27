#!/usr/bin/env bash
# The launch gate's CPU evidence on Thor -- authorised by the Master Mind 2026-09-26 ~23:15 Berlin
# ("THOR is AUTHORISED for your CPU evidence"): a fresh dir, md5-verified ship, the tanitad-train
# python with NO installs (pytest from the --no-deps side folder), CUDA_VISIBLE_DEVICES="",
# OMP_NUM_THREADS=2 (the box is shared), one job at a time, and only processes this script starts.
#   nohup bash run_gate_evidence_thor.sh <gate dir> [<out name>] > <gate dir>/runner.out 2>&1 &
# Progress and the end state are FILES: <out>/progress.log, <out>/<step>/..., and <out>/ALLDONE.
# GATE_STEPS (env, comma list; default: all) selects steps by name: closure_probe, unit, tiny_live, model_r6,
# model_r7like, model_r7, clock_r6, arms_r7, arms_r7c, arms_r6, arms_tiny, self_mutation, replay,
# model_r6_eval,
# arms_r6_eval -- a re-check runs only what the code change touches.
set -u
G="${1:?gate dir}"
O="$G/${2:-out}"
PY=/home/nvidia/venvs/tanitad-train/bin/python
PK=/home/nvidia/gate_fix_2224/pytest_pkgs
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONIOENCODING=utf-8
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
Z=0000000000000000000000000000000000000000
KEY=(--key-file "$G/keys/rehearsal.key")
# the clock reference is a gate INPUT bound in every token, so every run names the real file
BASE=(--cpu-only --omp 2 --min-free-gb 8 --ram-wait-s 3600
      --clock-reference "$G/in/q4c_grid_vs_egolog_ALLTRAIN.json")
EV=(--eval-loader "$G/in/refcv6_loader.py" --eval-kit /home/nvidia --eval-stamps "$G/in/refcv6_config.json"
    --eval-remap-overrides "$G/in/eval_remap_overrides_thor.json")
TAU=(--nav-tau-record "$G/in/nav_compliance_tau_train.json")
GATE="$G/cab/stack/scripts/launch_gate.py"
ARMS="$G/cab/stack/scripts/launch_gate_arms.py"
mkdir -p "$O"
mark() { echo "$(date -u +%H:%M:%S) $1" >> "$O/progress.log"; }
# (NOT `STEPS`: the supervisor tests read that name as a manifest variable -- MEASURED
# 2026-09-27; the selector is unset so no child ever sees it)
SEL="${GATE_STEPS:-all}"
unset GATE_STEPS STEPS
want() { [ "$SEL" = all ] || case ",$SEL," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

gate() {  # gate <step> <profile> <checks> <argv file> [extra...]
  local step=$1 prof=$2 checks=$3 argv=$4; shift 4
  want "$step" || { mark "SKIP $step"; return 0; }
  mark "START $step"
  "$PY" "$GATE" run --profile "$prof" --checks "$checks" --tree "$G/cab" --commit "$Z" \
    --argv-file "$argv" --out-dir "$O/$step" "${BASE[@]}" "${KEY[@]}" "$@" > "$O/$step.out" 2>&1
  mark "END $step rc=$?"
}

arms() {  # arms <step> <profile> <argv file> <arms csv> [runner options...] -- [gate extra...]
  local step=$1 prof=$2 argv=$3 list=$4; shift 4
  want "$step" || { mark "SKIP $step"; return 0; }
  mark "START $step"
  "$PY" "$ARMS" --tree "$G/cab" --argv-file "$argv" --profile "$prof" --out-dir "$O/$step" \
    --arms "$list" "$@" > "$O/$step.out" 2>&1
  mark "END $step rc=$?"
}

mark "BEGIN $(cat "$G/tree_note.txt" 2>/dev/null)"
PYTHONPATH="$G/cab/stack:$G/cab/taniteval" "$PY" -c \
  "import tanitad,sys; f=tanitad.__file__; print(f); sys.exit(0 if f.startswith(sys.argv[1]) else 3)" \
  "$G/cab/stack" > "$O/import_check.txt" 2>&1
echo "rc=$?" >> "$O/import_check.txt"
# ---- 1. the gate's unit tests on the clean tree ---------------------------------------------
# ---- 0. SPEC_REFCV7 A11 dry run: the REAL map harness under closure_run.py, judged by the gate --
if want closure_probe; then
mark "START closure_probe"
"$PY" "$G/code/closure_probe_thor.py" "$G" > "$O/closure_probe.out" 2>&1
mark "END closure_probe rc=$?"
fi
if want unit; then
mark "START unit"
mkdir -p "$O/unit"
( cd "$G/cab/stack" && PYTHONPATH="$G/cab/stack:$G/cab/taniteval:$PK" "$PY" -m pytest -q \
    -p no:cacheprovider tests/test_launch_gate.py --junitxml="$O/unit/junit.xml" \
    -o junit_family=xunit2 > "$O/unit/pytest.log" 2>&1 )
mark "END unit rc=$?"
fi
# ---- 2. the CPU rehearsal of the smoke (G-LIVE + G-CKPT) on the tiny rig ---------------------
gate tiny_live refc G-LIVE,G-CKPT "$G/in/argv_tiny_thor.json"
# ---- 3. the model checks on the fixed tip: refcv6's argv, and the refcv7-like argv ------------
gate model_r6 refcv6 G-HYG,G-DVB "$G/in/argv_refcv6_thor.json"
gate model_r7like refcv7 G-HYG,G-DVB "$G/in/argv_refcv7like_thor.json" "${TAU[@]}"
# the CANONICAL launch argv (stack/ops/runs.d/refcv7-r101-s0.argv.json, `--out` moved into the gate
# dir), with the tau record the launch reads (placed on Thor by the Master Mind, sha256 10ca19db...)
TAUC=(--nav-tau-record /home/nvidia/data/refcv7/nav_compliance_tau_train.json)
gate model_r7 refcv7 G-HYG,G-DVB "$G/in/argv_refcv7_canon_thor.json" "${TAUC[@]}"
# ---- 4. the label clock on Thor's REAL train + eval caches ----------------------------------
gate clock_r6 refcv6 G-CLOCK "$G/in/argv_refcv6_thor.json"
# ---- 5. the regression arms (controls reused from 3/4: same binding, same gate) ---------------
arms arms_r7 refcv7 "$G/in/argv_refcv7like_thor.json" \
  unwire_selection_term,required_on_missing,tau_differs,tau_file_missing,drivort_flag,residual_prior_ha0_ext,residual_prior_cv_yawrate \
  --control-evidence "G-DVB=$O/model_r7like/evidence" -- "${BASE[@]}" "${KEY[@]}" "${TAU[@]}"
arms arms_r7c refcv7 "$G/in/argv_refcv7_canon_thor.json" \
  unwire_selection_term,required_on_missing,tau_differs,tau_file_missing,drivort_flag,residual_prior_ha0_ext,residual_prior_cv_yawrate,map_drivable_only_static \
  --control-evidence "G-DVB=$O/model_r7/evidence" -- "${BASE[@]}" "${KEY[@]}" "${TAUC[@]}"
arms arms_r6 refcv6 "$G/in/argv_refcv6_thor.json" \
  drop_fix3_field,undeclared_equalize,missing_hygiene_module,missing_dvb_module,legacy_label_clock,venv_lacks_module \
  --control-evidence "G-DVB=$O/model_r6/evidence" --control-evidence "G-HYG=$O/model_r6/evidence" \
  --control-evidence "G-CLOCK=$O/clock_r6/evidence" -- "${BASE[@]}" "${KEY[@]}"
arms arms_tiny refc "$G/in/argv_tiny_thor.json" \
  no_cascade_passthrough,no_cascade_silent,ga_off_by_one,undeclared_equalize,drop_fix3_field,missing_hygiene_module,missing_dvb_module,ceiling_active_in_training,no_fmap_s8_passthrough,map_class_weight_zero,map_drivable_only_logging,map_iou_not_counts,map_overfit_missing \
  --control-evidence "G-LIVE=$O/tiny_live/evidence" --token-arms --alt-tree "$G/tip" \
  -- "${BASE[@]}" "${KEY[@]}"
# ---- 6. the gate's own mutation audit --------------------------------------------------------
if want self_mutation; then
mark "START self_mutation"
PYTHONPATH="$PK" "$PY" "$G/code/gate_self_mutation.py" --src "$G/cab" --tanitad "$G/cab" \
  --json "$O/gate_self_mutation.json" > "$O/gate_self_mutation.out" 2>&1
mark "END self_mutation rc=$?"
fi
# ---- 7. G-LIVE's term rule on refcv6's REAL run record (a copy; the run dir is not read) ------
if want replay; then
mark "START replay"
"$PY" "$G/code/replay_refcv6_metrics.py" --tree "$G/cab" --config "$G/in/refcv6_config.json" \
  --metrics "$G/in/refcv6_metrics.jsonl" --json "$O/replay_refcv6_metrics.json" > "$O/replay.out" 2>&1
mark "END replay rc=$?"
fi
# ---- 8. G-EVAL (three full-size CPU forwards: the slowest, so last) and its arm ---------------
gate model_r6_eval refcv6 G-EVAL "$G/in/argv_refcv6_thor.json" "${EV[@]}"
arms arms_r6_eval refcv6 "$G/in/argv_refcv6_thor.json" loader_skips_pin \
  --control-evidence "G-EVAL=$O/model_r6_eval/evidence" -- "${BASE[@]}" "${KEY[@]}" "${EV[@]}"
date -u > "$O/ALLDONE"
mark "ALLDONE"

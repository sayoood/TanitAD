#!/usr/bin/env bash
# E1 — exact commands for the NavSim warmup_two_stage reference run (navsim@0a380a9).
# Usage:  bash run_warmup_reference.sh <step>     steps: cache | A1 | R1 | A2 | A1b | A2b | M1
# Every step runs the OFFICIAL devkit entrypoint, unmodified, through code/navsim_win.py
# (Windows loader patch + observation-only hooks + RAM guard). CPU only, 1 worker.
# ⛔ Never read $? through a pipe: each step's status is asserted from its ARTIFACTS
#    (manifest status, row counts) by verify_cache.py / verify_controls.py, not here.
set -u
PKG="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms"
# ⚠️ RUNTIME = the verified C: MIRROR (2026-09-19 11:40-11:58). The D:-resident navsim venv was
# I/O-starved by the concurrent navhard download on the same external drive: `import numpy`
# 23,350 ms (D:) vs 324 ms (C:); the first cache attempt was still importing after 469 s.
# Mirror = venv reinstalled OFFLINE from uv's C: cache (same 200 pins) + devkit copies verified
# blob-for-blob against git (raw/devkit_copy_verify.json) + the pre-existing fcntl shim (sha256
# 75184a48...) + data copies verified by sha256 (raw/data_mirror_verify_warmup.json).
# OUTPUTS stay on the canonical D:-backed path C:/Users/Admin/navsim/exp (junction) that E2 polls.
CR="C:/Users/Admin/navsim-crun"
PY="$CR/venv/Scripts/python.exe"
WRAP="$PKG/code/navsim_win.py"
RAW="$PKG/raw"
EXP="C:/Users/Admin/navsim/exp"
CACHE="$EXP/metric_cache_warmup_two_stage"
DATA="$CR/data/openscene"
SYN_SCENES="$DATA/warmup_two_stage/synthetic_scene_pickles"
SYN_SENSORS="C:/Users/Admin/navsim/data/openscene/warmup_two_stage/sensor_blobs"   # never read: both agents use build_no_sensors()
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" NUPLAN_MAPS_ROOT="$CR/data/maps"
export NAVSIM_EXP_ROOT="$EXP" NAVSIM_DEVKIT_ROOT="$CR/devkit" OPENSCENE_DATA_ROOT="$DATA"
# keep BLAS/OpenMP from fanning out threads next to the live training run
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

COMMON_SCORE=(train_test_split=warmup_two_stage "metric_cache_path=$CACHE"
              "synthetic_sensor_path=$SYN_SENSORS" "synthetic_scenes_path=$SYN_SCENES" worker=sequential)
ONE_STAGE=(train_test_split=warmup_two_stage "metric_cache_path=$CACHE"
           train_test_split.scene_filter.include_synthetic_scenes=false traffic_agents=reactive worker=sequential)

step="${1:?step}"
mkdir -p "$RAW/$step"
case "$step" in
  cache)   # metric cache — the artifact stream E2 polls (CACHE_DONE.json is written by verify_cache.py)
    "$PY" "$WRAP" --script metric_caching --label cache --out-dir "$RAW/cache" -- \
      train_test_split=warmup_two_stage "metric_cache_path=$CACHE" \
      "synthetic_scenes_path=$SYN_SCENES" worker=sequential ;;
  A1)      # constant velocity, official two-stage runner (PRIMARY)
    PYTHONHASHSEED=1 "$PY" "$WRAP" --script pdm_score --label A1 --out-dir "$RAW/A1" --patch-loader \
      --dump-final-scores "$RAW/A1/A1_final_scores_frame.csv" -- \
      "${COMMON_SCORE[@]}" agent=constant_velocity_agent experiment_name=e1_A1_cv_two_stage \
      "output_dir=$EXP/e1/A1_cv_two_stage" ;;
  R1)      # replicate of A1 in a fresh process with a DIFFERENT hash seed (control C7)
    PYTHONHASHSEED=2 "$PY" "$WRAP" --script pdm_score --label R1 --out-dir "$RAW/R1" --patch-loader \
      --dump-final-scores "$RAW/R1/R1_final_scores_frame.csv" -- \
      "${COMMON_SCORE[@]}" agent=constant_velocity_agent experiment_name=e1_R1_cv_two_stage_replicate \
      "output_dir=$EXP/e1/R1_cv_two_stage_replicate" ;;
  A2)      # human agent, official two-stage runner (prediction P-HUM: stage 2 undefined)
    PYTHONHASHSEED=1 "$PY" "$WRAP" --script pdm_score --label A2 --out-dir "$RAW/A2" --patch-loader \
      --dump-final-scores "$RAW/A2/A2_final_scores_frame.csv" -- \
      "${COMMON_SCORE[@]}" agent=human_agent experiment_name=e1_A2_human_two_stage \
      "output_dir=$EXP/e1/A2_human_two_stage" ;;
  A1b)     # constant velocity, official ONE-stage runner, IDM-reactive (control C6)
    PYTHONHASHSEED=1 "$PY" "$WRAP" --script pdm_score_one_stage --label A1b --out-dir "$RAW/A1b" --patch-loader -- \
      "${ONE_STAGE[@]}" agent=constant_velocity_agent experiment_name=e1_A1b_cv_one_stage \
      "output_dir=$EXP/e1/A1b_cv_one_stage" ;;
  A2b)     # human agent, official ONE-stage runner, IDM-reactive (human stage-1 numbers + control C1)
    PYTHONHASHSEED=1 "$PY" "$WRAP" --script pdm_score_one_stage --label A2b --out-dir "$RAW/A2b" --patch-loader -- \
      "${ONE_STAGE[@]}" agent=human_agent experiment_name=e1_A2b_human_one_stage \
      "output_dir=$EXP/e1/A2b_human_one_stage" ;;
  M1)      # MUTATION: A2b with the human penalty filter OFF (gives C1 its teeth)
    PYTHONHASHSEED=1 "$PY" "$WRAP" --script pdm_score_one_stage --label M1 --out-dir "$RAW/M1" --patch-loader -- \
      "${ONE_STAGE[@]}" agent=human_agent scorer.config.human_penalty_filter=false \
      experiment_name=e1_M1_human_one_stage_filter_off "output_dir=$EXP/e1/M1_human_one_stage_filter_off" ;;
  *) echo "unknown step $step"; exit 2 ;;
esac

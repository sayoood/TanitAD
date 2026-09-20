#!/usr/bin/env bash
# E1 priority 2 (orchestrator, PI-authorised 2026-09-19): navhard_two_stage on the OFFICIAL protocol.
# Usage: bash run_navhard.sh <step>   steps: mirror | cache | N1 | N2b
# Same wrapper (code/navsim_win.py), same verified C: runtime as the warmup run.
# Parallelism: the devkit's own worker `single_machine_thread_pool` with use_process_pool=true,
# 2 workers (RAM-limited: ~0.4-0.5 GB per process MEASURED on warmup; system headroom ~1-2 GB).
# ⚠️ Pool CHILDREN do not inherit in-process monkeypatches, so for SCORING the MetricCacheLoader
# separator fix is applied to the C: COPY (code/patches/dataloader_token_separator.diff) — the D:
# devkit is never edited. Caching does not use the loader.
set -u
PKG="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms"
CR="C:/Users/Admin/navsim-crun"
PY="$CR/venv/Scripts/python.exe"
WRAP="$PKG/code/navsim_win.py"
RAW="$PKG/raw/navhard"
EXP="C:/Users/Admin/navsim/exp"
CACHE="$EXP/metric_cache_navhard_two_stage"
DATA="$CR/data/openscene"
SYN_SCENES="$DATA/navhard_two_stage/synthetic_scene_pickles"
SYN_SENSORS="C:/Users/Admin/navsim/data/openscene/navhard_two_stage/sensor_blobs"   # never read (no-sensor agents)
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" NUPLAN_MAPS_ROOT="$CR/data/maps"
export NAVSIM_EXP_ROOT="$EXP" NAVSIM_DEVKIT_ROOT="$CR/devkit" OPENSCENE_DATA_ROOT="$DATA"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
POOL=(worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=2)
step="${1:?step}"
mkdir -p "$RAW/$step"
case "$step" in
  mirror)  # copy the extracted navhard synthetic pickles + the 76 logs to the C: mirror, then sha256-verify
    [ -f "C:/Users/Admin/navsim/data/openscene/navhard_two_stage/EXTRACT_DONE.json" ] || { echo "EXTRACT_DONE.json absent - refusing"; exit 2; }
    powershell.exe -NoProfile -Command "robocopy 'D:\\Archive\\devbox-C\\navsim\\data\\openscene\\navhard_two_stage\\synthetic_scene_pickles' 'C:\\Users\\Admin\\navsim-crun\\data\\openscene\\navhard_two_stage\\synthetic_scene_pickles' /E /MT:8 /R:2 /W:1 /NFL /NDL /NP /NJH /NJS | Out-Null; 'synthetic rc=' + \$LASTEXITCODE"
    "$PY" - <<'PYEOF'
import yaml, shutil, os
sf = yaml.safe_load(open("C:/Users/Admin/navsim-crun/devkit/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"))
src, dst = "D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test", "C:/Users/Admin/navsim-crun/data/openscene/navsim_logs/test"
n = 0
for ln in sf["log_names"]:
    if not os.path.exists(f"{dst}/{ln}.pkl"):
        shutil.copy2(f"{src}/{ln}.pkl", f"{dst}/{ln}.pkl"); n += 1
print("logs copied", n, "of", len(sf["log_names"]))
PYEOF
    rels=(openscene/navhard_two_stage/synthetic_scene_pickles)
    for ln in $("$PY" -c "import yaml;print(' '.join(yaml.safe_load(open('C:/Users/Admin/navsim-crun/devkit/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml'))['log_names']))"); do rels+=("openscene/navsim_logs/test/$ln.pkl"); done
    "$PY" "$PKG/code/verify_mirror.py" "$RAW/mirror/data_mirror_verify_navhard.json" "${rels[@]}" ;;
  cache)
    "$PY" "$WRAP" --script metric_caching --label nh_cache --out-dir "$RAW/cache" -- \
      train_test_split=navhard_two_stage "metric_cache_path=$CACHE" "synthetic_scenes_path=$SYN_SCENES" "${POOL[@]}" ;;
  N1)      # constant velocity, OFFICIAL two-stage runner, navhard
    PYTHONHASHSEED=1 "$PY" "$WRAP" --script pdm_score --label N1 --out-dir "$RAW/N1" --patch-loader \
      --dump-final-scores "$RAW/N1/N1_final_scores_frame.csv" -- \
      train_test_split=navhard_two_stage "metric_cache_path=$CACHE" "synthetic_sensor_path=$SYN_SENSORS" \
      "synthetic_scenes_path=$SYN_SCENES" agent=constant_velocity_agent "${POOL[@]}" \
      experiment_name=e1_N1_cv_navhard "output_dir=$EXP/e1/N1_cv_navhard_two_stage" ;;
  N2b)     # human agent, stage 1 only (the two-stage human is undefined: 0 future frames on synthetic scenes)
    PYTHONHASHSEED=1 "$PY" "$WRAP" --script pdm_score_one_stage --label N2b --out-dir "$RAW/N2b" --patch-loader -- \
      train_test_split=navhard_two_stage "metric_cache_path=$CACHE" train_test_split.scene_filter.include_synthetic_scenes=false \
      traffic_agents=reactive agent=human_agent "${POOL[@]}" \
      experiment_name=e1_N2b_human_navhard_one_stage "output_dir=$EXP/e1/N2b_human_navhard_one_stage" ;;
  *) echo "unknown step $step"; exit 2 ;;
esac

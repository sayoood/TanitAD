#!/usr/bin/env bash
# VARIANT of E1's run_navhard.sh made by the EvalFlyWheel orchestrator 2026-09-19 at the Master Mind's
# request (no load on a training box): EXP moved OFF D: to C:/Users/Admin/navsim-crun/exp, and ONE
# worker instead of two. Nothing else changed; E1's original is untouched.
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
EXP="C:/Users/Admin/navsim-crun/exp"   # OFF D: (MM request 2026-09-19: A8/A7 read D:)
CACHE="$EXP/metric_cache_navhard_two_stage"
DATA="$CR/data/openscene"
SYN_SCENES="$DATA/navhard_two_stage/synthetic_scene_pickles"
SYN_SENSORS="C:/Users/Admin/navsim/data/openscene/navhard_two_stage/sensor_blobs"   # never read (no-sensor agents)
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" NUPLAN_MAPS_ROOT="$CR/data/maps"
export NAVSIM_EXP_ROOT="$EXP" NAVSIM_DEVKIT_ROOT="$CR/devkit" OPENSCENE_DATA_ROOT="$DATA"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
POOL=(worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=1)
step="${1:?step}"
mkdir -p "$RAW/$step"
case "$step" in
  mirror)  # the C: pickle copy is EXTRACTED BY THE ORCHESTRATOR (never written here); E1 only VERIFIES it,
           # and copies + sha256-verifies the 76 logs (not part of that extraction).
    "$PY" - <<'PYEOF'
import json, os, hashlib, random, sys
m = "C:/Users/Admin/navsim-crun/data/openscene/navhard_two_stage/EXTRACT_DONE.json"
if not os.path.exists(m) or not json.load(open(m)).get("ok"):
    print("C: EXTRACT_DONE.json absent or not ok - refusing"); sys.exit(2)
c = "C:/Users/Admin/navsim-crun/data/openscene/navhard_two_stage/synthetic_scene_pickles"
d = "C:/Users/Admin/navsim/data/openscene/navhard_two_stage/synthetic_scene_pickles"
names = sorted(os.listdir(c)); zero = [n for n in names if os.path.getsize(os.path.join(c, n)) == 0]
random.seed(19); sample = random.sample(names, 100)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
mism = [n for n in sample if sha(os.path.join(c, n)) != sha(os.path.join(d, n))]
res = {"n_pickles_C": len(names), "n_zero_byte": len(zero), "sha256_sample": 100, "sample_mismatches": mism,
       "set_equal_to_D": set(names) == set(os.listdir(d)), "ok": len(names) == 5462 and not zero and not mism}
json.dump(res, open("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/raw/navhard/mirror/c_copy_verify.json", "w"), indent=1)
print(res); sys.exit(0 if res["ok"] else 1)
PYEOF
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
    rels=()
    for ln in $("$PY" -c "import yaml;print(' '.join(yaml.safe_load(open('C:/Users/Admin/navsim-crun/devkit/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml'))['log_names']))"); do rels+=("openscene/navsim_logs/test/$ln.pkl"); done
    "$PY" "$PKG/code/verify_mirror.py" "$RAW/mirror/data_mirror_verify_navhard_logs.json" "${rels[@]}" ;;
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

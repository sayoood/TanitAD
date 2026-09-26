#!/bin/bash
# Stage 0 -- run the released DriveRL teacher closed-loop on the nuPlan v1.1 MINI split (dev box, RTX 4060).
#
# Their runner (scripts/run_driverl_nuplan_eval.sh) hard-codes six tasks whose DB paths are
# splits/trainval and splits/test.  It exposes two hooks that let a MINI run through untouched:
#   DRIVERL_EVAL_DB_LINK_ROOT/driverl_val14   -> replaces the DB dir for the val14 task
#   DRIVERL_EVAL_SCENARIO_FILTER_OVERRIDE     -> replaces the val14 token filter (mini has none of those tokens)
# So we run task val14_nr|val14_r with the DB re-pointed at mini and a devkit filter that works on any split.
#
# Usage:  bash run_mini_teacher.sh <MINI_DB_DIR> [nr|r] [limit] [tts_candidates|0]
#   MINI_DB_DIR   directory holding the mini *.db files. Data on D: ONLY (PI 2026-09-20). The DBs LIVE at
#                 D:/Projects/TanitAD/data/nuplan/dblinks/driverl_val14 as a plain directory (the zip's own layout was
#                 data/cache/mini/; renamed on the same volume). No links anywhere: exFAT has no symlinks and an NTFS
#                 junction into exFAT lists empty. Run outputs (logs/videos) stay on C:/Users/Admin/dz/out (short paths)
#                 and are copied into the repo package.
#   NUPLAN_MAPS_ROOT must be the directory that DIRECTLY contains nuplan-maps-v1.0.json + the four city dirs
#                 (…/nuplan-maps/nuplan-maps-v1.0), not its parent — the devkit loads ${map_root}/${map_version}.json.
#   nr|r          non-reactive (log-replay background) or reactive (IDM background)   [default nr]
#   limit         cap on scenarios via limit_total_scenarios                         [default 8]
#   tts           0 = plain Beta mode; N>0 = value-guided TTS with N candidates       [default 0]
# Env:  DRIVERL_EVAL_DRY_RUN=1 validates the whole wiring without data (DB dir must merely exist).
set -euo pipefail
MINI_DB="${1:?MINI_DB_DIR}"; PROTO="${2:-nr}"; LIMIT="${3:-8}"; TTS="${4:-0}"
[[ "$PROTO" == nr || "$PROTO" == r ]] || { echo "proto must be nr or r"; exit 2; }

DZ="C:/Users/Admin/dz/DriveZero/DriveRL"
VPY="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/bd7d00af-b98e-42f1-a53c-cb4113059b0f/scratchpad/driverl-venv/Scripts/python.exe"
LINKS="D:/Projects/TanitAD/data/nuplan/dblinks"   # LINK-FREE: their runner only needs $LINKS/driverl_val14 to EXIST, so it is a plain dir on D:
OUT="C:/dzo"                                       # near the drive root: Windows MAX_PATH, see the arithmetic below
mkdir -p "$OUT"

# 2026-09-20, MEASURED: run #4 died at the simulation-log write with FileNotFoundError on a 263-char
# path -- Windows MAX_PATH (259 usable). The devkit composes, under the output dir, a FIXED tail:
#   <closed_loop_..._driverl_val14_nr:47>/simulation_log/DriveRLNuPlanPlanner/<scenario_type>/<log_name:38>/<token:16>/<token>.msgpack.xz
# (simulation_log_callback.py:147). The longest scenario_type the devkit can emit is 54 chars
# (starting_straight_traffic_light_intersection_traversal), and the six-task suite adds 8 more to the
# task segment -> worst-case tail 231. So the prefix budget is 259-231 = 28 chars, and the old
# C:/Users/Admin/dz/out/mini/notts_nr/mini-nr-l8-notts (52) could never hold it. C:/dzo + a <=7-char
# RUN_ID = 14, leaving 14 chars of headroom. exit_on_failure=true is THEIRS and stays: one over-long
# path aborted the whole suite after 5 of 8 scenarios, which is the behaviour we want to keep.

# 2026-09-20, MEASURED: an NTFS junction (C:) pointing INTO the exFAT volume (D:) reports isdir=True but
# lists EMPTY to Python/os.listdir -> the devkit saw "No log files found!". exFAT holds no symlinks either.
# So the mini DBs simply LIVE at $LINKS/driverl_val14 (renamed there from data/cache/mini, same volume).
LINK="$LINKS/driverl_val14"
[[ -d "$LINK" ]] || { echo "expected the mini DBs as a plain directory at $LINK"; exit 3; }
[[ -n "$(ls "$LINK"/*.db 2>/dev/null | head -1)" ]] || { echo "no *.db in $LINK"; exit 3; }
[[ "$(cd "$MINI_DB" 2>/dev/null && pwd -W 2>/dev/null || echo x)" == "$(cd "$LINK" && pwd -W)" ]] || echo "note: MINI_DB arg ($MINI_DB) differs from the dir actually used ($LINK)"

cd "$DZ"
export DRIVERL_EVAL_ROOT="$PWD"
export DRIVERL_EVAL_NUPLAN_ROOT="$PWD/nuplan-devkit"
export DRIVERL_EVAL_RELEASE_ID=driverl-teacher-u2400
export DRIVERL_EVAL_CONFIG_PATH="$PWD/release/configs/driverl_teacher.yaml"
export DRIVERL_EVAL_CHECKPOINT_PATH="$PWD/release/checkpoints/checkpoint_2400.pt"
export DRIVERL_EVAL_PYTHON="$VPY"
export NUPLAN_DATA_ROOT="D:/Projects/TanitAD/data/nuplan"    # PI 2026-09-20: DATA LIVES ON D: ONLY. Only /nuplan-v1.1/splits/<x> is derived from it; overridden by the junction
export NUPLAN_MAPS_ROOT="D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0" # already local: nuplan-maps-v1.0/{sg-one-north,us-ma-boston,us-nv-las-vegas-strip,us-pa-pittsburgh-hazelwood}
export DRIVERL_EVAL_DB_LINK_ROOT="$LINKS"
export DRIVERL_EVAL_SCENARIO_FILTER_OVERRIDE=one_of_each_scenario_type   # devkit-shipped, split-agnostic
export DRIVERL_EVAL_LIMIT_TOTAL_SCENARIOS="$LIMIT"
export DRIVERL_EVAL_DEVICE=cuda
export DRIVERL_EVAL_TASKS="val14_${PROTO}"
export DRIVERL_EVAL_SAVE_SIMULATION_LOGS=1
export DRIVERL_EVAL_ROUTE_GOAL_HORIZON_S=12.0 DRIVERL_EVAL_ROUTE_GOAL_MIN_SPEED_MPS=5.0 DRIVERL_EVAL_ROUTE_GOAL_PAIR_MODE=legacy
export DRIVERL_EVAL_BICYCLE_MAX_ACCELERATION=4.0 DRIVERL_EVAL_BICYCLE_MAX_STEERING_RATE=0.8
export DRIVERL_EVAL_WORKER_MODE=sequential   # first light: no Ray, one process; switch to ray_local for throughput
export DRIVERL_EVAL_LOCAL_GPUS=1
export DRIVERL_EVAL_SKIP_PREFLIGHT=0
if [[ "$TTS" != 0 ]]; then
  export DRIVERL_EVAL_TTS_ENABLED=1 DRIVERL_EVAL_TTS_NUM_CANDIDATES="$TTS" DRIVERL_EVAL_TTS_SEED=42
  export DRIVERL_EVAL_RUN_ID="m-${PROTO}-${TTS}"
  export DRIVERL_EVAL_OUTPUT_ROOT="$OUT"
else
  export DRIVERL_EVAL_TTS_ENABLED=0
  # Stage-1 route augmentation: DRIVERL_EVAL_ROUTE_LANE_RANK=k picks the k-th nearest lane at the
  # route start (sitecustomize -> route_lane_rank_patch). rank 0/unset installs NOTHING, so the
  # control arm is byte-identical to Stage 0 by construction. Run id keeps <=7 chars (MAX_PATH).
  RANK="${DRIVERL_EVAL_ROUTE_LANE_RANK:-0}"
  # The id MUST encode the limit. MEASURED 2026-09-20: a limit-64 run wrote into the same output
  # root as the Stage-0 limit-8 baseline because the id carried only the rank, so an analysis that
  # globs and takes [-1] silently switched from the 8-scenario run to a 10-scenario one.
  # MAX_PATH budget is 28 chars of prefix and "C:/dzo/" costs 7, so ids may be up to 21 -- the
  # earlier <=7 self-limit was over-cautious. Stage-0's canonical ids are kept at limit 8.
  SUF=""; [[ "$LIMIT" != 8 ]] && SUF="-l${LIMIT}"
  if [[ "$RANK" == 0 ]]; then export DRIVERL_EVAL_RUN_ID="m-${PROTO}-n${SUF}"
  else export DRIVERL_EVAL_RUN_ID="m-${PROTO}-r${RANK}${SUF}"; fi
  export DRIVERL_EVAL_OUTPUT_ROOT="$OUT"
fi
echo "MINI_RUN db=$MINI_DB via $LINK  proto=$PROTO limit=$LIMIT tts=$TTS  out=$DRIVERL_EVAL_OUTPUT_ROOT  dry=${DRIVERL_EVAL_DRY_RUN:-0}"
bash scripts/run_driverl_nuplan_eval.sh eval

#!/bin/bash
# DZ-11 -- the six-task headline reproduction of DriveZero's released teacher on nuPlan val14/test14.
#
# Usage: bash run_headline_suite.sh [task-list]     default runs the CHEAP tasks first (see cost note)
#   e.g. bash run_headline_suite.sh test14hard_nr,test14hard_r
#        bash run_headline_suite.sh val14_nr,val14_r
#
# ============================== READ BEFORE RUNNING ==============================
# 1) *** THE SILENT-WRONG-RESULT TRAP THIS SCRIPT EXISTS TO PREVENT ***
#    Their runner resolves the DB dir as:
#        DB="${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/trainval"     (val14_*)
#        DB="${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/test"         (test14*_*)
#    and then OVERRIDES it if DRIVERL_EVAL_DB_LINK_ROOT is set and $LINK_ROOT/$FILTER exists
#    (run_driverl_nuplan_eval.sh:197-198).
#    Stage 0 created D:/.../data/nuplan/dblinks/driverl_val14 holding the 64 *MINI* DBs.
#    => If DB_LINK_ROOT were inherited here, val14_nr/val14_r would SILENTLY read MINI and report a
#       "val14" score computed on 64 mini logs. It would not error. It would just be wrong.
#    This script therefore UNSETS DB_LINK_ROOT and asserts the real splits by SIZE and COUNT.
#
# 2) MAX_PATH. Output goes to C:/dzo/<short-id>. Worst-case devkit tail for this suite is 230 chars
#    (task segment 54 for closed_loop_nonreactive_agents_driverl_test14random_nr, scenario_type up to
#    54, log_name 38, token twice + .msgpack.xz). Budget is 259, so the prefix must stay <= 29.
#
# 3) OMP_NUM_THREADS. torch spawns ~113 threads per process; with ray_local workers this box makes
#    NO progress and looks hung (TanitAD CLAUDE.md, MEASURED: 7 arms at 0-6% GPU for 50 min).
#
# 4) COST, priced from Stage 0's MEASURED 114 s/scenario sequential on the 4060:
#    (token counts MEASURED from the shipped filter yamls, not estimated)
#       test14hard_nr + _r      272 x 2 =   544 scenarios  ->  ~17.2 h sequential
#       test14random_nr + _r    261 x 2 =   522 scenarios  ->  ~16.5 h sequential
#       val14_nr + _r          1118 x 2 =  2236 scenarios  ->  ~70.8 h sequential
#                                TOTAL  =  3302  => ~104.6 h SEQUENTIAL. Parallelism is mandatory.
#
# 5) LAYOUT. The zips do NOT extract to the path the runner reads. They contain data/cache/<split>/,
#    the runner wants nuplan-v1.1/splits/{trainval,test}, and exFAT cannot link the two. Run
#    code/arrange_splits.py --apply after extraction (same-volume rename, instant).
#    Run the four test14 tasks first: they are the rows DriveZero's headline table reports, and they
#    are 1/3 of the cost.
# 6) MANDATORY POST-CHECK -- the RESULT line's `total` is a coverage assertion, not decoration.
#    The filters carry tokens and `log_names: null`, so a token whose log is absent from the split is
#    silently skipped and the run still succeeds. Assert the totals:
#        val14_nr / val14_r            total == 1118
#        test14hard_nr / test14hard_r  total ==  272
#        test14random_* / _r           total ==  261
#    A smaller total means MISSING DATA, not a teacher failure. Do not score a run that misses it.
# =================================================================================
set -euo pipefail
TASKS="${1:-test14hard_nr,test14hard_r,test14random_nr,test14random_r}"
DZ="C:/Users/Admin/dz/DriveZero/DriveRL"
VPY="${REFE_DRIVERL_PY:-C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe}"
# DZ11_DATA_ROOT exists so the GUARDS can be mutation-tested against a fixture; production leaves it unset.
DATA="${DZ11_DATA_ROOT:-D:/Projects/TanitAD/data/nuplan}"
TRAINVAL="$DATA/nuplan-v1.1/splits/trainval"
TEST="$DATA/nuplan-v1.1/splits/test"

need_split() {  # dir, min_db_count, min_gb, label
  local d="$1" minn="$2" mingb="$3" lab="$4"
  [[ -d "$d" ]] || { echo "GUARD_FAIL $lab: $d does not exist"; return 1; }
  local n gb
  n=$(ls "$d"/*.db 2>/dev/null | wc -l)
  gb=$(du -sb "$d" 2>/dev/null | awk '{printf "%.0f", $1/1073741824}')
  echo "  $lab: $n DBs, ${gb} GB  (require >= $minn DBs and >= ${mingb} GB)"
  [[ "$n" -ge "$minn" ]] || { echo "GUARD_FAIL $lab: only $n DBs -- is this the MINI split?"; return 1; }
  [[ "$gb" -ge "$mingb" ]] || { echo "GUARD_FAIL $lab: only ${gb} GB -- extraction incomplete or MINI"; return 1; }
  return 0
}

echo "== DZ-11 preflight =="
# mini is 64 DBs / 13 GB. Real trainval and test are ~150 GB each at the MEASURED 1.68x ratio.
# Thresholds are overridable ONLY so the guards can be mutation-tested (the DB_LINK_ROOT branch is
# otherwise unreachable behind them). Any override is announced loudly so it cannot be used to
# silently weaken the guard in production.
MIN_DB_TV="${DZ11_MIN_DB_TRAINVAL:-200}"; MIN_GB_TV="${DZ11_MIN_GB_TRAINVAL:-60}"
MIN_DB_TE="${DZ11_MIN_DB_TEST:-100}";     MIN_GB_TE="${DZ11_MIN_GB_TEST:-40}"
if [[ "$MIN_DB_TV" != 200 || "$MIN_GB_TV" != 60 || "$MIN_DB_TE" != 100 || "$MIN_GB_TE" != 40 ]]; then
  echo "  *** GUARD THRESHOLDS OVERRIDDEN (trainval >=${MIN_DB_TV}DB/${MIN_GB_TV}GB, test >=${MIN_DB_TE}DB/${MIN_GB_TE}GB)."
  echo "  *** THIS IS A TEST MODE. Any score produced under it is INADMISSIBLE."
fi
# ⛔ GUARD THE SPLITS THE REQUESTED TASKS ACTUALLY READ -- NOT BOTH, UNCONDITIONALLY.
# MEASURED 2026-09-20: requiring both made the four **test14** tasks -- the rows DriveZero's
# headline table reports, and the ones that read the TEST split only -- exit 3 while the VAL split
# was still downloading. Val is NETWORK-bound at ~0.7-1.6 MB/s (measured against 11.6 MB/s disk and
# a 1.62 MB/s memory-only S3 range GET, so it is the link), i.e. ~40 h. The coupling would have
# stalled the headline reproduction for those 40 h for no reason, and it would have failed at the
# moment the arrange fired rather than now.
# ⭐ THIS DOES NOT WEAKEN THE GUARD. Each task's own split is still checked by DB count AND size,
# which is what stops a MINI directory being scored as val14. What is removed is a split being
# demanded by tasks that never read it. The mutation arms below still go RED.
NEED_TV=0; NEED_TE=0
case ",$TASKS," in (*,val14_nr,*|*,val14_r,*|*val14*) NEED_TV=1;; esac
case ",$TASKS," in (*test14*) NEED_TE=1;; esac
if [[ "$NEED_TV" -eq 0 && "$NEED_TE" -eq 0 ]]; then
  echo "GUARD_FAIL tasks: '$TASKS' names neither a val14 nor a test14 task"; exit 3
fi
if [[ "$NEED_TV" -eq 1 ]]; then
  need_split "$TRAINVAL" "$MIN_DB_TV" "$MIN_GB_TV" "trainval" || exit 3
else
  echo "  trainval NOT required: '$TASKS' names no val14 task"
fi
if [[ "$NEED_TE" -eq 1 ]]; then
  need_split "$TEST"     "$MIN_DB_TE" "$MIN_GB_TE" "test"     || exit 3
else
  echo "  test NOT required: '$TASKS' names no test14 task"
fi
if [[ -n "${DRIVERL_EVAL_DB_LINK_ROOT:-}" ]]; then
  echo "  REFUSING an inherited DRIVERL_EVAL_DB_LINK_ROOT=${DRIVERL_EVAL_DB_LINK_ROOT} -- it would"
  echo "  redirect val14 to the Stage-0 MINI directory and silently mis-report the score."
  exit 4
fi
echo "  DB_LINK_ROOT unset: val14 will read trainval, test14* will read test"

cd "$DZ"
export DRIVERL_EVAL_ROOT="$PWD" DRIVERL_EVAL_NUPLAN_ROOT="$PWD/nuplan-devkit"
export DRIVERL_EVAL_RELEASE_ID=driverl-teacher-u2400
export DRIVERL_EVAL_CONFIG_PATH="$PWD/release/configs/driverl_teacher.yaml"
export DRIVERL_EVAL_CHECKPOINT_PATH="$PWD/release/checkpoints/checkpoint_2400.pt"
export DRIVERL_EVAL_PYTHON="$VPY"
export NUPLAN_DATA_ROOT="$DATA"
export NUPLAN_MAPS_ROOT="$DATA-maps/nuplan-maps-v1.0"
unset DRIVERL_EVAL_DB_LINK_ROOT DRIVERL_EVAL_SCENARIO_FILTER_OVERRIDE DRIVERL_EVAL_LIMIT_TOTAL_SCENARIOS
export DRIVERL_EVAL_DEVICE=cuda DRIVERL_EVAL_TASKS="$TASKS"
export DRIVERL_EVAL_SAVE_SIMULATION_LOGS=1
export DRIVERL_EVAL_ROUTE_GOAL_HORIZON_S=12.0 DRIVERL_EVAL_ROUTE_GOAL_MIN_SPEED_MPS=5.0 DRIVERL_EVAL_ROUTE_GOAL_PAIR_MODE=legacy
export DRIVERL_EVAL_BICYCLE_MAX_ACCELERATION=4.0 DRIVERL_EVAL_BICYCLE_MAX_STEERING_RATE=0.8
export DRIVERL_EVAL_WORKER_MODE="${DRIVERL_EVAL_WORKER_MODE:-ray_local}"
export DRIVERL_EVAL_LOCAL_GPUS=1 DRIVERL_EVAL_SKIP_PREFLIGHT=0
export DRIVERL_EVAL_TTS_ENABLED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"     # MANDATORY -- see note 3
# ⛔ RESPECT AN OVERRIDE. This was an unconditional export, so a REPLICATE launched on the same
# day silently reused the first run's id and wrote into its directory -- the run-id collision this
# package already documented once, reintroduced by a date-derived default. A replicate is the only
# way to put an interval on a score, so the id has to be settable.
export DRIVERL_EVAL_RUN_ID="${DRIVERL_EVAL_RUN_ID:-h$(date +%m%d)}"   # <=7 chars -- see note 2
export DRIVERL_EVAL_OUTPUT_ROOT="C:/dzo"
# ⛔ RAY'S TEMP DIR MUST BE SHORT, AND THE RUNNER REFUSES WITHOUT IT.
# MEASURED 2026-09-20 on the first real launch: "ERROR: set DRIVERL_EVAL_RAY_TEMP_DIR to a short
# writable directory". This is the Windows MAX_PATH family again -- ray builds long session paths
# under its temp root, and a default under the user profile blows the 259-character budget before
# the first worker starts. Same reason DRIVERL_EVAL_OUTPUT_ROOT is C:/dzo and the run id is <= 7
# characters. Set it here so a fresh operator does not rediscover it at launch time.
export DRIVERL_EVAL_RAY_TEMP_DIR="${DRIVERL_EVAL_RAY_TEMP_DIR:-C:/rayt}"
mkdir -p "$DRIVERL_EVAL_RAY_TEMP_DIR"
mkdir -p "$DRIVERL_EVAL_OUTPUT_ROOT"
echo "HEADLINE_RUN tasks=$TASKS run_id=$DRIVERL_EVAL_RUN_ID worker=$DRIVERL_EVAL_WORKER_MODE omp=$OMP_NUM_THREADS"
bash scripts/run_driverl_nuplan_eval.sh eval

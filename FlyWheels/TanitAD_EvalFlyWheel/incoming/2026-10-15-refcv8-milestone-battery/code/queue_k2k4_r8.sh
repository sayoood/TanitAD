#!/bin/sh
# K2 / K4 QUEUE (AMENDMENT A-R8-CLAMP-1 section 3) -- dev box RTX 4060 only; Thor is never touched.   usage: sh queue_k2k4_r8.sh [30000ema]
#   NOT STARTED by its author.  Written 2026-10-18 by the EvalFlyWheel while the raw-30k battery held the shared GPU lock.
#   What it does, in order (assert on ARTIFACTS, never an exit code; opaque markers ZZK2K4...ZZ in chain/k2k4_<tag>.log):
#    0. CPU preconditions: ckpt / config / banked dump / windows exist; the clamp tree passes its PROVENANCE check (r8clamp_common.py check);
#       the episode PREFIX is planned from the banked dump (k2k4_compare.py plan): the first K episodes in dump order, K chosen so >= 10 nav_true rows BIND.
#    1. host-commit wait (>= 6 GB), then the SHARED GPU LOCK exactly as chain_battery_r8.sh takes it (gpu_lock.py acquire; a foreign lock is never
#       broken; the EXIT trap releases).
#    2. K2 roll: tree_de4e15b, flag OFF, seed 0 (full arm set), standalone clock (R8_NO_POSE_SYNC=1 = what the banked *_unc trees were rolled on),
#       on the prefix windows.      -> $KIT/clamp_gpu/<tag>/off/dump_s0
#    3. K4 roll: the same, flag ON.   -> .../on/dump_s0   (its extras also carry traj_precap / plan_cap_binding / plan_v_lim_ms)
#    4. OPTIONAL control (K2K4_CONTROL=1): the LAUNCH tree fb47a58 (roll_seed_r8.py = the banked code path) on the same prefix -> .../ctl/dump_s0.
#       It tells a NON-bit-reproducible roll rig apart from a tree that changes flag-OFF numerics.
#    5. k2k4_compare.py compare (CPU, plus the same vendored function on the GPU) -> PKG/raw/clamp_gpu/K2K4_<tag>.json.
#   Estimated GPU time: ~(342 windows x ~1.8 s) per roll + ~3 min load, two rolls (three with the control): ~30-40 min.   UNVERIFIED until it runs.
#   Every path is overridable by R8_* / K2K4_* env vars ONLY so the control flow can be dry-run in a sandbox with stub scripts (test_queue_k2k4_r8.py).
LABEL="${1:-30000ema}"
TAG="step$LABEL"
PKGW="${R8_PKGW:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
BAT="${R8_BAT:-C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code}"
KITW="${R8_KITW:-E:/refcv8_eval_kit}"
KIT="${R8_KIT:-/e/refcv8_eval_kit}"
PY="${R8_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
TREEW="${R8_CLAMP_TREE:-E:/refcv8_eval_kit/tree_de4e15b}"
LAUNCHW="${K2K4_LAUNCH_TREE:-C:/lgs/Lfb47a58376}"
BANKED="${K2K4_BANKED:-C:/Users/Admin/refcv8_eval_archive/battery/step30000ema_unc}"
CK="${K2K4_CKPT:-$KITW/ckpt/ckpt_ema_30000.pt}"
CFG="${K2K4_CONFIG:-$KITW/ckpt/config.json}"
S2W="${K2K4_S2_WINDOWS:-$KITW/windows/s2_windows.json}"
NEPS="${K2K4_N_EPISODES:-}"
WORKW="$KITW/clamp_gpu/$TAG"
WORKL="$KIT/clamp_gpu/$TAG"
OUTW="$PKGW/raw/clamp_gpu"
OUTJ="$OUTW/K2K4_$TAG.json"
LOG=$KIT/chain/k2k4_$TAG.log
TIM=$KIT/chain/k2k4_timings_$TAG.txt
PP_TREE="$TREEW/stack;$TREEW/taniteval"
PP_LAUNCH="$LAUNCHW/stack;$LAUNCHW/taniteval"
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6
unset R8_NO_POSE_SYNC R8_CLAMP_MODE R8_LAUNCH_TREE
mkdir -p "$KIT/chain" "$WORKL" "$OUTW"
note() { echo "[$(date +%FT%T)] k2k4 queue $TAG: $*" >> $KIT/KIT.log; }
ts() { echo "$(date +%s) $(date +%FT%T) $1" >> $TIM; }
mark() { echo "$1 $(date +%FT%T%z) $2" >> $LOG; }
jget() {   # $1 = json file, $2 = python expression over `j`  (prints nothing on any error)
  $PY -I -c "import json,sys; j=json.load(open(sys.argv[1],encoding='utf-8')); print($2)" "$1" 2>/dev/null
}
echo "ZZK2K4STARTZZ $(date +%FT%T%z) pid $$ tag $TAG" >> $LOG
note "armed (pid $$)"; ts "armed"
# ---- 0. CPU preconditions ---------------------------------------------------------------------------- #
for f in "$CK" "$CFG" "$S2W" "$BANKED/dump_s0/refcv6_extras.npz" "$BANKED/dump_s0/manifest.json" "$BANKED/dump_s0/refcv6_extras.npz.json"; do
  if [ ! -s "$f" ]; then mark ZZK2K4PRECONDFAILZZ "missing or empty: $f"; note "PRECOND FAIL: $f"; exit 2; fi
done
$PY "$PKGW/code/r8clamp_common.py" check --out "$WORKW/provenance.json" >> $LOG 2>&1
if [ "$(jget "$WORKL/provenance.json" "j.get('ok') is True")" != "True" ]; then
  mark ZZK2K4PROVFAILZZ "$(jget "$WORKL/provenance.json" "str(j.get('reason'))[:300]")"; note "PROVENANCE FAIL"; exit 2
fi
mark ZZK2K4PROVOKZZ ""
NARG=""; [ -n "$NEPS" ] && NARG="--n-episodes $NEPS"
PYTHONPATH="$PP_TREE" $PY "$PKGW/code/k2k4_compare.py" plan --banked "$BANKED" --windows-json "$S2W" \
    --out-windows "$WORKW/prefix_windows.json" --out-plan "$OUTW/k2k4_prefix_plan_$TAG.json" $NARG >> $LOG 2>&1
if [ ! -s "$WORKL/prefix_windows.json" ] || [ ! -s "$OUTW/k2k4_prefix_plan_$TAG.json" ]; then
  mark ZZK2K4PLANFAILZZ "see $LOG"; note "PLAN FAIL"; exit 2
fi
mark ZZK2K4PLANOKZZ "$(jget "$OUTW/k2k4_prefix_plan_$TAG.json" "'episodes=%s windows=%s binding=%s' % (j['n_episodes'], j['n_windows'], j['n_binding_posthoc_nav_true_in_prefix'])")"
ts "preconditions + plan done"
# ---- 1. host commit first, the lock second (exactly as chain_battery_r8.sh) --------------------------- #
$PY "$BAT/wait_commit.py" --min-gb 6 --max-wait-s 43200 --out "$WORKW/wait_commit.json" >> $LOG 2>&1
if [ "$(jget "$WORKL/wait_commit.json" "j['met']")" != "True" ]; then mark ZZK2K4NOCOMMITZZ ""; exit 3; fi
JOB="refcv8-k2k4-$TAG"
. "$(dirname "$0")/yield40k_r8.sh"; acquire_y40 "$JOB" "$WORKW/lock.json" "$LOG" 172800   # 2026-10-19: yields to the step-40k EMA battery
if [ "$(jget "$WORKL/lock.json" "j['ok']")" != "True" ]; then mark ZZK2K4NOLOCKZZ ""; exit 3; fi
release() { $PY "$BAT/gpu_lock.py" release --job $JOB --pid $$ >> $LOG 2>&1; }
marker() { $PY "$BAT/chain_marker.py" "$1" --chain $JOB --pid $$ >> $LOG 2>&1; }
STOPW="$KIT/chain/k2k4_gpu_watch_$TAG.stop"; rm -f "$STOPW"
$PY "$BAT/gpu_watch.py" --out "$KITW/chain/k2k4_gpu_watch_$TAG.jsonl" --stop-file "$KITW/chain/k2k4_gpu_watch_$TAG.stop" >> $LOG 2>&1 &
marker create
trap ': > "$STOPW"; release; marker remove' EXIT
mark ZZK2K4LOCKEDZZ ""; note "GPU lock acquired"; ts "lock acquired"
# ---- 2/3/4. the rolls (one subshell each: the environment must not leak between them) --------------- #
roll() {   # $1 = name (off|on|ctl)
  N=$1; mkdir -p "$WORKL/$N"
  case "$N" in
    ctl) SCRIPT="roll_seed_r8.py"; PP="$PP_LAUNCH"; MODE="" ;;
    *)   SCRIPT="roll_seed_r8clamp.py"; PP="$PP_TREE"; MODE="$N" ;;
  esac
  (
    export PYTHONPATH="$PP" R8_NO_POSE_SYNC=1 R8_SEAM_TELE_DIR="$WORKW/$N/seam_tele" R8_CLAMP_STATS_DIR="$WORKW/$N/clamp_stats"
    [ -n "$MODE" ] && export R8_CLAMP_MODE="$MODE"
    $PY "$PKGW/code/$SCRIPT" --ckpt "$CK" --config "$CFG" --seed 0 --dump-dir "$WORKW/$N/dump_s0" --windows-json "$WORKW/prefix_windows.json" \
        --out-json "$WORKW/$N/roll_s0.json"
  ) >> "$KIT/chain/k2k4_roll_${TAG}_$N.log" 2>&1
  if [ -s "$WORKL/$N/dump_s0/refcv6_extras.npz" ] && [ -s "$WORKL/$N/roll_s0.json" ]; then return 0; else return 1; fi
}
ts "roll off start"
if roll off; then mark ZZK2K4OFFDONEZZ ""; else mark ZZK2K4OFFFAILZZ "see chain/k2k4_roll_${TAG}_off.log"; fi
ts "roll off end"
if [ -s "$WORKL/off/dump_s0/refcv6_extras.npz" ]; then
  ts "roll on start"
  if roll on; then mark ZZK2K4ONDONEZZ ""; else mark ZZK2K4ONFAILZZ "see chain/k2k4_roll_${TAG}_on.log"; fi
  ts "roll on end"
fi
CTL=""
if [ "${K2K4_CONTROL:-0}" = "1" ]; then
  ts "roll ctl start"
  if roll ctl; then mark ZZK2K4CTLDONEZZ ""; CTL="--control-dump $WORKW/ctl/dump_s0"; else mark ZZK2K4CTLFAILZZ "see chain/k2k4_roll_${TAG}_ctl.log"; fi
  ts "roll ctl end"
fi
# ---- 5. the comparison (CPU, plus the same vendored function on the still-held GPU) ---------------------- #
if [ -s "$WORKL/off/dump_s0/refcv6_extras.npz" ] && [ -s "$WORKL/on/dump_s0/refcv6_extras.npz" ]; then
  PYTHONPATH="$PP_TREE" $PY "$PKGW/code/k2k4_compare.py" compare --banked "$BANKED" --off-dump "$WORKW/off/dump_s0" --on-dump "$WORKW/on/dump_s0" \
      $CTL --out "$OUTJ" --step "$LABEL" --gpu-posthoc >> $LOG 2>&1
  ts "compare end"
  if [ -s "$OUTJ" ]; then
    mark ZZK2K4VERDICTZZ "K2=$(jget "$OUTJ" "j['K2']['verdict']") K4=$(jget "$OUTJ" "j['K4']['verdict']") OVERALL=$(jget "$OUTJ" "j['OVERALL']['verdict']")"
  else
    mark ZZK2K4NOJSONZZ "see $LOG"
  fi
else
  mark ZZK2K4NOCOMPAREZZ "an extras file is missing (see the roll logs)"
fi
mark ZZK2K4CHAINENDZZ ""
note "chain end"

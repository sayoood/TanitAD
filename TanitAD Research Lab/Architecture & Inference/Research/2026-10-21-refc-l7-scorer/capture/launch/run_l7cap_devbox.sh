#!/bin/bash
# run_l7cap_devbox.sh [seed ...]   (default: 0 1) -- H-L7-1 TRAIN139 fan capture on the DEV-BOX GPU (RTX 4060), AFTER the current dev-box queue.
# WRITTEN, NOT RUN (zero GPU was authorised for this task).
#
# Discipline (CLAUDE.md + the brief):
#   * waits (up to $L7C_QUEUE_WAIT_S, default 24 h) until NO dev-box chain marker exists, then creates the marker itself (atomic, noclobber):
#         C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active   {"chain","pid","started"}   -- the format the queue guardians read
#     and REMOVES it only if it is still ours, on EVERY exit path that bash can trap (normal, `exit N`, INT, TERM, HUP; a SIGKILL cannot be trapped:
#     the guardian's stale-marker sweep covers that, as it did for refcv8-battery-step48852ema).
#   * takes the dev-box GPU lock with gpu_lock.py and a LONG wait (default 48 h); asserts on the lock.json ARTIFACT, never on the exit code;
#     releases it (own job + own pid only) in the same trap. A lock held by anyone else is never broken.
#   * every output goes to $O (C:/Users/Admin/refcv8_pf/l7cap/devbox_out); nothing is staged, committed or pushed.
# Markers (opaque) in $O/devbox.log: ZZL7CD<NAME>ZZ.
# Optional: L7C_MAP_ROOT=<dir of the 139 train SAM3 .sam3mapgt.npz> turns the drivable-crop label bank on (the dev box has none for train;
#           see P1_INPUTS.md for the 610 MB LAN copy from Thor; WITHOUT it the label bank carries agent futures only and DAC abstains).
set -u
SEEDS=${*:-"0 1"}
JOB=l7cap-devbox
P=C:/Users/Admin/refcv8_pf/l7cap; O=${L7C_OUT:-$P/devbox_out}; mkdir -p "$O"; LOG=$O/devbox.log
Q=${L7C_Q:-C:/Users/Admin/qland/work/refcv7}; FLAG=$Q/devbox_gpu.chain_active     # L7C_Q / L7C_GL / L7C_DRY: test hooks only (tests/test_devbox_script.sh)
GL=${L7C_GL:-C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code/gpu_lock.py}
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
HOLDW=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
CK=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt; CFG=E:/refcv8_eval_kit/ckpt/config.json
AGJ=E:/refcv6_eval_kit/data/joins/b1_train_plus_eval_agents.jsonl.xz
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       OMP_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONDONTWRITEBYTECODE=1
unset CUDA_VISIBLE_DEVICES REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION L7CAP_MUTATION
mark() { echo "ZZL7CD$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }

HAVE_FLAG=0; HAVE_LOCK=0; CHILD=0
# every long command runs in the BACKGROUND and is `wait`ed on: bash defers a trap until a FOREGROUND child exits, so a TERM during a 50-minute roll
# would otherwise leave the marker in place for 50 minutes. With `wait` the trap fires at once, kills the child, and the marker/lock are released.
runbg() { "$@" & CHILD=$!; wait $CHILD; local r=$?; CHILD=0; return $r; }
cleanup() {
  rc=$?
  if [ "$CHILD" != 0 ]; then kill "$CHILD" 2>/dev/null; sleep 2; kill -9 "$CHILD" 2>/dev/null; CHILD=0; fi
  if [ $HAVE_LOCK = 1 ]; then "$PY" "$GL" release --job "$JOB" --pid $$ --out "$O/release.json" >/dev/null 2>&1; mark RELEASED "rc=$rc"; HAVE_LOCK=0; fi
  if [ $HAVE_FLAG = 1 ]; then
    if grep -q "\"chain\": \"$JOB\", \"pid\": \"$$\"" "$FLAG" 2>/dev/null; then rm -f "$FLAG"; mark FLAGCLEARED "rc=$rc"; else mark FLAGNOTOURS "rc=$rc"; fi
    HAVE_FLAG=0
  fi
  mark EXIT "rc=$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
mark START "seeds=$SEEDS pid $$"

# ---- 1. wait for the dev-box queue, then claim the chain marker atomically --------------------------------------------------------------
t0=$(date +%s); maxw=${L7C_QUEUE_WAIT_S:-86400}
while :; do
  if ( set -o noclobber; printf '{"chain": "%s", "pid": "%s", "started": "%s"}' "$JOB" "$$" "$(date +%Y-%m-%dT%H:%M:%S%z)" > "$FLAG" ) 2>/dev/null; then HAVE_FLAG=1; break; fi
  [ $(( $(date +%s) - t0 )) -gt "$maxw" ] && { mark QUEUETIMEOUT "waited>${maxw}s"; exit 4; }
  runbg sleep 60
done
mark FLAGSET "waited_s=$(( $(date +%s) - t0 ))"

# ---- 2. the GPU lock, LONG wait; assert on the artifact ---------------------------------------------------------------------------------
runbg "$PY" "$GL" acquire --job "$JOB" --pid $$ --max-wait-s "${L7C_LOCK_WAIT_S:-172800}" --out "$O/lock.json" >/dev/null 2>&1
if [ "$("$PY" -c "import json,sys; print(int(json.load(open(sys.argv[1])).get('ok') is True))" "$O/lock.json" 2>/dev/null)" != 1 ]; then mark NOLOCK ""; exit 5; fi
HAVE_LOCK=1; mark LOCKED ""

if [ "${L7C_DRY:-0}" = 1 ]; then mark DRY "hold ${L7C_DRY_HOLD_S:-0}s"; runbg sleep "${L7C_DRY_HOLD_S:-0}"; mark DONE "dry"; exit 0; fi
# ---- 3. inputs, then the rolls (one process per seed) --------------------------------------------------------------------------------------
[ -s "$CK" ] && [ -s "$CFG" ] && [ -s "$AGJ" ] && [ -s "$P/TRAIN139_WINDOWS.json" ] || { mark INPUTSMISSING ""; exit 6; }
for S in $SEEDS; do
  D=$O/train139_s$S; rm -rf "$D"; mkdir -p "$D"
  MAPARG=""; [ -n "${L7C_MAP_ROOT:-}" ] && MAPARG="--l7-map-gt-root $L7C_MAP_ROOT"
  mark ROLLSTART "seed=$S"
  runbg "$PY" -X utf8 "$P/code/fan_roll_r8_l7cap.py" --mode train139 --pf-code "$HOLDW/code" --ckpt "$CK" --config "$CFG" --seed "$S" \
      --dump-dir "$D/dump" --out-json "$D/roll_s$S.json" --fan-meta-json "$D/fan_roll_s$S.json" --registered-windows "$P/TRAIN139_WINDOWS.json" \
      --l7-agent-join "$AGJ" $MAPARG --device cuda > "$D/roll.log" 2> "$D/roll.err"
  rc=$?
  fan=0; [ -s "$D/dump/fan_extras.npz" ] && fan=1
  mark ROLLEND "seed=$S rc=$rc fan_extras=$fan"
  # CPU post-steps: the S6 GT from the dump itself, then control (b) on the real windows
  if [ $fan = 1 ]; then
    runbg "$PY" "$P/code/build_gt6.py" --dump "$D/dump" --out "$D/gt6" > "$D/gt6.log" 2>&1
    AG=""; [ -s "$D/dump/l7cap_labels.npz" ] && AG="--agents-npz $D/dump/l7cap_labels.npz"
    runbg "$PY" "$P/code/controls_l7cap.py" machinery --dump "$D/dump" --gt6 "$D/gt6" $AG --out "$D/MACHINERY_s$S.json" >> "$D/gt6.log" 2>&1
    mach=0; [ -s "$D/MACHINERY_s$S.json" ] && mach=1
    (cd "$D" && md5sum dump/fan_extras.npz dump/l7cap_scene.npz dump/refcv6_extras.npz fan_roll_s$S.json roll_s$S.json > MD5_L7C$S.txt 2>/dev/null)
    mark POST "seed=$S machinery_json=$mach"
  fi
  # clip-id rule: the dump manifest and the logs carry raw clip ids -> rewrite to clip-sha12-<12 hex>, then assert on the ARTIFACT (CLEAN flag), not on a status
  runbg "$PY" "$P/code/scrub_clip_ids.py" --root "$D" --json "$D/SCRUB_CLIP_IDS.json" >> "$D/scrub.log" 2>&1
  scr=0; [ "$("$PY" -c "import json,sys; print(int(json.load(open(sys.argv[1]))['CLEAN']))" "$D/SCRUB_CLIP_IDS.json" 2>/dev/null)" = 1 ] && scr=1
  mark SCRUB "seed=$S clean=$scr"
done
mark DONE "seeds=$SEEDS"
exit 0

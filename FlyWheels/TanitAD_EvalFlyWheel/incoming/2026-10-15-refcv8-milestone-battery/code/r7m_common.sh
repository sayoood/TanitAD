#!/bin/sh
# r7m_common.sh -- shared definitions for ctl_r7_matched_phase.sh and chain_r7_matched_phase.sh (PREREG_R7_MATCHED_PHASE_TURN.md).
# SOURCED, never run.  NEW file; it edits nothing that the running refcv8 queue/wait scripts call.
# Rules kept: assert on ARTIFACTS (never an exit code, never `$?` through a pipe); markers are opaque ZZR7M...ZZ tokens in a file that only
# THIS chain writes (a monitor must not grep its own command's text); a foreign GPU lock is never broken (gpu_lock.py refuses); the EXIT trap
# releases only a lock this shell acquired (same job AND pid).
PKGW="C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery"
PKGL="/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery"
BAT="C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code"
KITW="E:/refcv8_eval_kit"; KIT="/e/refcv8_eval_kit"
R7KITW="E:/refcv7_eval_kit"
ARCW="C:/Users/Admin/refcv8_eval_archive/battery"; ARCL="/c/Users/Admin/refcv8_eval_archive/battery"
RAWW="$PKGW/raw/r7_matched_phase"; RAWL="$PKGL/raw/r7_matched_phase"
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
CKPT_DIR="$R7KITW/ckpt"; CFG7="$R7KITW/ckpt/config.json"
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6
unset R8_NO_POSE_SYNC
mkdir -p "$KIT/chain" "$RAWL" "$ARCL"
LOG="$KIT/chain/chain_r7_matched_phase.log"
QLOG="$KIT/chain/queue_30000_r8.log"; WLOG="$KIT/chain/wait_30000_r8.log"
mark() { echo "ZZR7M$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
note() { echo "[$(date +%FT%T)] r7_matched_phase: $*" >> "$KIT/KIT.log"; }
jq_ok() {   # $1 = file, $2 = python boolean expression over `j`; true iff the file parses and the expression holds (-X utf8: -I ignores PYTHONIOENCODING)
  _f=$(cygpath -m "$1" 2>/dev/null || echo "$1")
  "$PY" -I -X utf8 -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$_f"
}
md5_of() { md5sum "$1" 2>/dev/null | cut -d' ' -f1; }
expect_md5() { grep " \*\?ckpt_$1\.pt" "$R7KITW/ckpt/MD5SUMS" 2>/dev/null | head -n 1 | cut -d' ' -f1; }
verify_ckpt() {   # $1 = step ; 0 iff md5(file) == the MD5SUMS entry and both are 32 chars
  _e=$(expect_md5 "$1"); _a=$(md5_of "/e/refcv7_eval_kit/ckpt/ckpt_$1.pt")
  [ "${#_e}" -eq 32 ] && [ "${#_a}" -eq 32 ] && [ "$_e" = "$_a" ]
}
# ---- the shared-card lock (the refcv7 battery's; mirrors chain_battery_r8.sh / queue_30000_r8.sh) ------------------- #
. "$(dirname "$0")/yield40k_r8.sh"
gpu_acquire() {   # $1 = job ; 0 iff WE hold the lock
  "$PY" "$BAT/wait_commit.py" --min-gb 6 --max-wait-s 43200 --out "$KITW/chain/r7m_wc_$1.json" >> "$KIT/chain/r7m_lockwait_$1.log" 2>&1
  jq_ok "$KIT/chain/r7m_wc_$1.json" "j.get('met') is True" || { mark NOCOMMIT "$1"; return 3; }
  acquire_y40 "$1" "$KITW/chain/r7m_lock_$1.json" "$KIT/chain/r7m_lockwait_$1.log" 172800   # 2026-10-19: yields to the step-40k EMA battery (yield40k_r8.sh)
  jq_ok "$KIT/chain/r7m_lock_$1.json" "j.get('ok') is True" || { mark NOLOCK "$1"; return 3; }
  "$PY" "$BAT/chain_marker.py" create --chain "$1" --pid $$ >> "$KIT/chain/r7m_lockwait_$1.log" 2>&1
  mark LOCKED "$1"; note "GPU lock acquired: $1"
}
gpu_release() {   # $1 = job
  "$PY" "$BAT/gpu_lock.py" release --job "$1" --pid $$ >> "$KIT/chain/r7m_lockwait_$1.log" 2>&1
  "$PY" "$BAT/chain_marker.py" remove --chain "$1" --pid $$ >> "$KIT/chain/r7m_lockwait_$1.log" 2>&1
  mark RELEASED "$1"
}
# ---- one battery pass on a refcv7 checkpoint (caller holds the lock) ---------------------------------------------- #
r7_pass() {   # $1 = tag (e.g. r7step20000_unc)  $2 = ckpt (Windows path)  rest = extra args (e.g. --max-clips 4)
  _tag=$1; _ck=$2; shift 2
  mkdir -p "$ARCL/seam_tele_$_tag"
  ( export R8_NO_POSE_SYNC=1 R8_SEAM_TELE_DIR="$ARCW/seam_tele_$_tag"
    "$PY" "$PKGW/code/run_battery_r7m.py" --ckpt "$_ck" --config "$CFG7" --no-g0 --tag "$_tag" --infer-seeds 0,1 --lock-held \
        --skip-roll --out-root "$ARCW" "$@" >> "$KIT/chain/r7m_run_$_tag.log" 2>&1 )
}
pass_done() {   # $1 = tag ; true iff the battery summary carries bars + finished + the r8 stamp (a full run), whole-grid unless SMOKE allowed ($2=smoke)
  if [ "$2" = "smoke" ]; then jq_ok "$ARCL/$1/battery_summary.json" "j.get('bars') and j.get('finished') and 'r8' in j"
  else jq_ok "$ARCL/$1/battery_summary.json" "j.get('bars') and j.get('finished') and 'r8' in j and not j.get('SMOKE_RESTRICTED_S2')"; fi
}

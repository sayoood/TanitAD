#!/bin/sh
# run_navfeed.sh -- the GPU WRAPPER of the nav-feed experiment (PREREG_NAV_FEED.md).  usage: sh run_navfeed.sh        (normally called by wait_navfeed.sh)
#   NEW file (2026-10-20, EvalFlyWheel).  Edits NOTHING in the launch tree (C:/lgs/Lfb47a58376) or the live battery package; writes ONLY under
#   $NF_OUTW (default C:/Users/Admin/refcv8_navfeed_out) and, at the end, copies the small result files into the hold's raw/run/.  Nothing is written to G:. No git.
#   Exit codes: 0 all six rolls done + analysis written | 10 YIELDED to the final (48,852) chain (resumable; re-call me) | 11 lock/commit not available (re-call me)
#               12 a precondition no longer holds after the lock was taken (re-call me) | 20 a roll or the smoke FAILED (do NOT re-call without reading the log) | 21 seal bad.
#   Order of work (a killed run still yields the primary comparison):  seal check -> commit gate -> GPU lock -> re-check conditions -> ckpt md5 -> runtime windows ->
#     MINI-SMOKE (clip + perframe_full on 6 windows; asserted on ARTIFACTS by smoke_check.py) -> rolls  (clip,s0) (full,s0) (clip,s1) (full,s1) (token,s0) (token,s1)
#     -> release the lock -> analysis (CPU) -> copy results.   Resumable per roll: a roll whose roll_*.json says done:true is skipped.
#   YIELD: before every roll after the first, the wrapper asks "is the final chain pending?" (the yield48852_r8.sh `bfinal_pending` rule, copied below) and, if so,
#   RELEASES the lock and exits 10 -- the final chain outranks every other GPU job.  A killed roll is never resumed mid-window: the unit is one roll (~12 min).
#   Assert on ARTIFACTS, never on an exit code, never through a pipe. Markers are opaque ZZNF...ZZ lines in $OUT/navfeed.log (never grepped by their own text).
HOLD_CODE="$(cd "$(dirname "$0")" && pwd)"
. "$HOLD_CODE/navfeed_common.sh"
release() { "$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ --out "$OUTW/state/release.json" >> "$LOG" 2>&1; }
HELD=0
on_exit() { if [ "$HELD" = 1 ]; then release; mark RELEASED "gpu lock released (job $JOB pid $$)"; HELD=0; fi; }
trap on_exit EXIT
trap 'exit 143' TERM INT HUP
mark START "wrapper pid $$"
# ---- 0. the seal ------------------------------------------------------------------------------------------------------------------------ #
PRE="$HOLDL/PREREG_NAV_FEED.md"; SEAL="$HOLDL/PREREG_NAV_FEED.md.sha256"
S_ACT=$("$PY" -I -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$(cygpath -m "$PRE")" 2>/dev/null)
S_REC=$(cut -d' ' -f1 < "$SEAL" 2>/dev/null | head -n 1 | tr -d '\r ')
if [ "${#S_ACT}" -ne 64 ] || [ "${#S_REC}" -ne 64 ] || [ "$S_ACT" != "$S_REC" ]; then mark SEALBAD "prereg sha256 '$S_ACT' vs sealed '$S_REC' (an empty string is INCONCLUSIVE, never a match)"; exit 21; fi
# every sealed code file must hash to a value the prereg TEXT contains (the prereg is the register of what was committed)
for f in code/overlay/navfeed_overlay.py code/overlay/navfeed_roll.py code/make_windows_navfeed.py code/build_roll_windows.py code/smoke_check.py code/nav_feed_analyze.py raw/windows_navfeed.json; do
  H=$("$PY" -I -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$(cygpath -m "$HOLDL/$f")" 2>/dev/null)
  if [ "${#H}" -ne 64 ] || ! grep -q "$H" "$PRE"; then mark SEALBAD "$f hash '$H' is not the one the prereg commits"; exit 21; fi
done
mark SEALOK "prereg $S_ACT; sealed code + window files match"
# ---- 1. host commit first, the lock second ---------------------------------------------------------------------------------------------- #
"$PY" "$BAT/wait_commit.py" --min-gb "${NF_MIN_COMMIT_GB:-6}" --max-wait-s "${NF_COMMIT_WAIT_S:-300}" --out "$OUTW/state/commit.json" >> "$LOG" 2>&1
jq_ok "$OUTL/state/commit.json" "j['met'] is True" || { mark NOCOMMIT "free commit < ${NF_MIN_COMMIT_GB:-6} GB"; exit 11; }
conditions_ok || exit 11
rm -f "$OUTL/state/lock.json"
"$PY" "$BAT/gpu_lock.py" acquire --job "$JOB" --pid $$ --max-wait-s "${NF_LOCK_WAIT_S:-300}" --out "$OUTW/state/lock.json" >> "$LOG" 2>&1
jq_ok "$OUTL/state/lock.json" "j.get('ok') is True" || { mark NOLOCK "gpu lock not acquired (a foreign lock is never broken)"; exit 11; }
HELD=1
mark LOCKED "gpu lock acquired (job $JOB pid $$)"
conditions_ok || { mark RECHECK "a condition no longer holds after the lock was taken; releasing"; exit 12; }
# ---- 2. inputs --------------------------------------------------------------------------------------------------------------------------- #
CK_ACT=$(md5sum "$CKPT_L" 2>/dev/null | cut -d' ' -f1)
if [ "${#CK_ACT}" -ne 32 ] || [ "$CK_ACT" != "$CKPT_MD5" ]; then mark CKPTBAD "ckpt md5 '$CK_ACT' != $CKPT_MD5 (empty is INCONCLUSIVE)"; exit 20; fi
mark CKPTOK "ckpt_ema_40000.pt md5 $CK_ACT"
"$PY" "$HOLDW/code/build_roll_windows.py" --windows "$HOLDW/raw/windows_navfeed.json" --archive-dump "$ARCH_DUMP" \
      --out "$OUTW/windows_roll.json" --smoke-out "$OUTW/windows_smoke.json" > "$OUTL/state/windows.out" 2>&1
jq_ok "$OUTL/state/windows.out" "j['n_windows']==1457" || { mark WINDOWSBAD "build_roll_windows did not report 1457 windows (see state/windows.out)"; exit 20; }
SMOKE_EP=$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['smoke_episode_index'])" "$(cygpath -m "$OUTL/state/windows.out")")
roll() {   # $1 arm  $2 seed  $3 windows json  $4 dump dir  $5 rec json  $6 log
  "$PY" "$HOLDW/code/overlay/navfeed_roll.py" --ckpt "$CKPT" --config "$CFG" --nav-feed "$1" --seed "$2" --dump-dir "$4" \
        --windows-json "$3" --out-json "$5" >> "$6" 2>&1
}
# ---- 3. MINI-SMOKE (once per code state) ------------------------------------------------------------------------------------------------- #
SMOKE_OK="$OUTL/state/smoke.ok.json"
SMOKE_KEY=$(cat "$HOLDL/code/overlay/navfeed_overlay.py" "$HOLDL/code/overlay/navfeed_roll.py" "$HOLDL/code/smoke_check.py" | md5sum | cut -d' ' -f1)
if ! jq_ok "$SMOKE_OK" "j.get('ok') is True and j.get('key')=='$SMOKE_KEY'"; then
  mark SMOKESTART "clip + perframe_full on 6 windows of one episode"
  rm -rf "$OUTL/smoke"/dump_* "$OUTL/smoke"/roll_*.json
  roll clip 0 "$OUTW/windows_smoke.json" "$OUTW/smoke/dump_clip_s0" "$OUTW/smoke/roll_clip_s0.json" "$OUTL/logs/smoke_clip_s0.log"
  roll perframe_full 0 "$OUTW/windows_smoke.json" "$OUTW/smoke/dump_perframe_full_s0" "$OUTW/smoke/roll_perframe_full_s0.json" "$OUTL/logs/smoke_perframe_full_s0.log"
  "$PY" "$HOLDW/code/smoke_check.py" --clip-dump "$OUTW/smoke/dump_clip_s0" --full-dump "$OUTW/smoke/dump_perframe_full_s0" \
        --windows "$HOLDW/raw/windows_navfeed.json" --smoke-episode-index "$SMOKE_EP" --out "$OUTW/state/smoke_check.json" >> "$LOG" 2>&1
  if jq_ok "$OUTL/state/smoke_check.json" "j.get('ok') is True"; then
    "$PY" -I -c "import json,sys; json.dump({'ok':True,'key':sys.argv[1]}, open(sys.argv[2],'w'))" "$SMOKE_KEY" "$(cygpath -m "$SMOKE_OK")"
    mark SMOKEOK "the feed is wired: see state/smoke_check.json"
  else
    mark SMOKEFAIL "smoke_check failed: see state/smoke_check.json and logs/smoke_*.log. NO further GPU work. Do not re-arm before reading them."
    exit 20
  fi
fi
# ---- 4. the six rolls ---------------------------------------------------------------------------------------------------------------------- #
roll_done() { jq_ok "$OUTL/roll_$1_s$2.json" "j.get('done') is True and j.get('nav_feed')=='$1' and int(j.get('seed'))==$2 and int(j.get('n_windows'))==1457" && [ -s "$OUTL/dump_$1_s$2/manifest.json" ]; }
FIRST=1
for unit in "clip 0" "perframe_full 0" "clip 1" "perframe_full 1" "perframe_token 0" "perframe_token 1"; do
  set -- $unit; ARM=$1; SEED=$2
  if roll_done "$ARM" "$SEED"; then mark ROLLHAVE "$ARM s$SEED already complete"; continue; fi
  if [ "$FIRST" = 0 ] && bfinal_pending; then mark YIELD "yielding the GPU to the final (48,852) chain before $ARM s$SEED"; exit 10; fi
  FIRST=0
  ATT="$OUTL/state/attempts_${ARM}_s${SEED}"; N=$(cat "$ATT" 2>/dev/null || echo 0); N=$((N + 1)); echo "$N" > "$ATT"
  [ "$N" -le 2 ] || { mark ROLLGIVEUP "$ARM s$SEED failed twice; not retrying"; exit 20; }
  rm -rf "$OUTL/dump_${ARM}_s${SEED}" "$OUTL/roll_${ARM}_s${SEED}.json"
  mark ROLLSTART "$ARM s$SEED attempt $N"
  roll "$ARM" "$SEED" "$OUTW/windows_roll.json" "$OUTW/dump_${ARM}_s${SEED}" "$OUTW/roll_${ARM}_s${SEED}.json" "$OUTL/logs/roll_${ARM}_s${SEED}.log"
  if roll_done "$ARM" "$SEED"; then mark ROLLDONE "$ARM s$SEED"
  else mark ROLLFAIL "$ARM s$SEED did not produce a complete record (see logs/roll_${ARM}_s${SEED}.log)"; exit 20; fi
done
# ---- 5. release the card, then the analysis (CPU) -------------------------------------------------------------------------------------------- #
release; mark RELEASED "gpu lock released after the last roll"; HELD=0
"$PY" -X utf8 "$HOLDW/code/nav_feed_analyze.py" --out-dir "$OUTW" --windows "$HOLDW/raw/windows_navfeed.json" --prereg "$HOLDW/PREREG_NAV_FEED.md" \
      > "$OUTL/logs/analysis.log" 2>&1
if jq_ok "$OUTL/NAV_FEED.json" "j['committed_label']['label'] in ('FEED-FIX-MATTERS','FEED-FIX-NULL','MIXED','NOT-EVALUABLE')"; then
  LBL=$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['committed_label']['label'])" "$(cygpath -m "$OUTL/NAV_FEED.json")")
  mkdir -p "$HOLDL/raw/run"
  cp "$OUTL/NAV_FEED.json" "$OUTL/NAV_FEED.md" "$HOLDL/raw/run/" 2>/dev/null
  for f in "$OUTL"/roll_*.json; do cp "$f" "$HOLDL/raw/run/" 2>/dev/null; done
  cp "$LOG" "$HOLDL/raw/run/navfeed.log" 2>/dev/null
  mark ANALYSIS "NAV_FEED.json written: committed label $LBL"
  mark DONE "all six rolls complete; results in $OUTW and $HOLDW/raw/run"
  exit 0
fi
mark ANALYSISFAIL "NAV_FEED.json missing or malformed: see logs/analysis.log (the dumps are intact: re-run nav_feed_analyze.py)"
exit 20

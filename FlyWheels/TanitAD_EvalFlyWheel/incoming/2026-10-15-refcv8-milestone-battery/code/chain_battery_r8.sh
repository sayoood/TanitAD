#!/bin/sh
# refcv8 BATTERY CHAIN (dev box RTX 4060 only; Thor is never touched). usage: sh chain_battery_r8.sh 5000
#   Armed DETACHED behind the G0 chain (`chain_g0_r8.sh`), which it NEVER edits or signals:
#   1. wait (poll 60 s) until the G0 chain's log shows a TERMINAL marker (or its shell is dead for 5 polls);
#   2. gate on ARTIFACTS: g0.json present, for THIS checkpoint (md5), M1 detected  -> else write the skip marker and exit;
#   3. host-commit wait, then the shared GPU LOCK (waits for the card; a foreign lock is never broken; EXIT trap releases);
#   4. PASS 1 = the STANDALONE pose clock (R8_NO_POSE_SYNC=1): refcv8 GT and model-free arms are bit-identical to refcv7's,
#      so this is the PAIRED read vs refcv7-50,400 (BAR-R8-E1/E3/NR).  Inference seeds 0 (all arms) and 1 (os + controls),
#      S2 + S6, panels, four families; then `pair_read_r8.py`; then the sanitised bank + BATTERY_SUMMARY_<tag>.json.
#   5. PASS 2 = the CORRECTED pose clock (the clock the run trained / was evaluated on), os-only at both seeds, NO banked
#      baselines (their GT is on the other clock): E1/E3 only.  Disable with R8_CORRECTED=0.
#   Optional (default OFF): R8_YIELD=1 turns on the refcv7 evening-yield rule (gpu_yield.py) between GPU stages;
#   R8_PERC=1 appends the raw perception pass (map 10 cm + box, perc_pass_r8.py) after the four-family passes.
#   G0 policy: the battery runs when G0 is FAIL-as-coded provided M1 was detected (the Master Mind's early-read brief);
#   such numbers are stamped PROVISIONAL.  R8_REQUIRE_G0_PASS=1 makes a non-PASS G0 a skip instead.
# Resumable per seed: `--skip-roll` reuses only COMPLETE seed dumps (manifest + roll record); one retry per pass.
# Assert on ARTIFACTS, never on an exit code; markers ZZ...ZZ in chain/battery_<tag>.log (opaque; never grepped by their own text).
STEP="${1:-5000}"
TAG="step$STEP"
# Every path below can be overridden by R8_* env vars ONLY so the control flow can be dry-run in a sandbox with stub scripts
# (test_chain_battery_r8.py); production leaves them unset.
PKGW="${R8_PKGW:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
PKGL="${R8_PKGL:-/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
BAT="${R8_BAT:-C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code}"
KITW="${R8_KITW:-E:/refcv8_eval_kit}"
KIT="${R8_KIT:-/e/refcv8_eval_kit}"
BRW="${R8_BATT_ROOT_W:-$KITW/battery}"; BRL="${R8_BATT_ROOT_L:-$KIT/battery}"   # battery PASS dump root (queue_30000 exports C: archive; default unchanged)
PY="${R8_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
POLL="${R8_POLL_S:-60}"
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6
unset R8_NO_POSE_SYNC
mkdir -p $KIT/chain
LOG=$KIT/chain/battery_$TAG.log
G0LOG="$KITW/chain/chain_$TAG.log"
G0LOGL=$KIT/chain/chain_$TAG.log
OUTW="$KITW/battery/$TAG"                 # the G0 work dir (g0.json lives here)
CK="$KITW/ckpt/ckpt_$STEP.pt"
CFG="$KITW/ckpt/config.json"
TIM=$KIT/chain/battery_timings_$TAG.txt
RAWW="$PKGW/raw/$TAG"
GATEJ="$KITW/chain/battery_gate_$TAG.json"
note() { echo "[$(date +%FT%T)] battery chain $TAG: $*" >> $KIT/KIT.log; }
ts() { echo "$(date +%s) $(date +%FT%T) $1" >> $TIM; }
mark() { echo "$1 $(date +%FT%T%z) $2" >> $LOG; }
echo "ZZR8BATTSTARTZZ $(date +%FT%T%z) pid $$ step $STEP" >> $LOG
note "armed (pid $$), waiting for the G0 chain"
ts "armed"
bank() {   # sanitised bank + BATTERY_SUMMARY (idempotent; re-run after every pass)
  $PY "$PKGW/code/bank_battery_r8.py" "$BRW" "$RAWW" --tag $TAG --passes unc,corr --chain-log "$KITW/chain/battery_$TAG.log" \
      --timings "$KITW/chain/battery_timings_$TAG.txt" --g0-gate "$GATEJ" --status "$1" >> $LOG 2>&1
}
skip() {   # $1 = reason
  mark ZZR8BATTSKIPZZ "$1"
  note "SKIPPED: $1"
  mkdir -p "$PKGL/raw/$TAG"
  $PY -I -c "import json,sys; json.dump({'status':'SKIPPED','reason':sys.argv[1],'tag':'$TAG'}, open(r'$RAWW/BATTERY_SUMMARY_$TAG.json','w',encoding='utf-8'), indent=1)" "$1"
  exit 0
}
# ---- 1. wait for the G0 chain's terminal marker ---------------------------------------------------- #
n=0; dead=0; lastsz=0
while :; do
  S=$($PY -I "$PKGW/code/chain_helpers_r8.py" g0state --log "$G0LOG" --step $STEP)
  case "$S" in ended:*) mark ZZR8BATTG0ENDEDZZ "$S"; break ;; esac
  GPID=${S##*:}
  SZ=$(wc -c < "$G0LOGL" 2>/dev/null || echo 0)
  if [ "$GPID" != "-" ] && ! kill -0 "$GPID" 2>/dev/null && [ "$SZ" = "$lastsz" ]; then dead=$((dead+1)); else dead=0; fi
  lastsz=$SZ
  if [ "$dead" -ge 5 ]; then mark ZZR8BATTG0DEADZZ "G0 chain shell $GPID gone with no terminal marker and a silent log"; break; fi
  n=$((n+1))
  [ $((n % 15)) -eq 1 ] && mark ZZR8BATTWAITZZ "g0 state $S"
  [ "$n" -ge 2880 ] && { mark ZZR8BATTG0TIMEOUTZZ "48 h"; break; }
  # near the end of the G0 chain (artifact written / banking) poll fast: the shared lock is released a moment after its
  # last marker, and a foreign waiter should not win the race by a poll interval
  case "$S" in running:G0DONE:*|running:G0NOJSON:*|running:BANKED:*|running:BANKFAIL:*) sleep 5 ;; *) sleep "$POLL" ;; esac
done
ts "g0 chain ended"
# ---- 2. gate on the artifact ----------------------------------------------------------------------- #
REQ=""; [ "${R8_REQUIRE_G0_PASS:-0}" = "1" ] && REQ="--require-pass"
GATE=$($PY -I "$PKGW/code/chain_helpers_r8.py" g0gate --g0 "$OUTW/g0.json" --ckpt "$CK" --out "$GATEJ" $REQ)
case "$GATE" in
  ok:*) mark ZZR8BATTGATEOKZZ "$GATE" ;;
  *) skip "$GATE" ;;
esac
# ---- 3. host commit first, the lock second ------------------------------------------------------- #
$PY "$BAT/wait_commit.py" --min-gb 6 --max-wait-s 43200 --out "$KITW/chain/battery_wait_commit_$TAG.json" >> $LOG 2>&1
if ! $PY -I -c "import json,sys; sys.exit(0 if json.load(open(r'$KITW/chain/battery_wait_commit_$TAG.json'))['met'] else 1)"; then
  mark ZZR8BATTNOCOMMITZZ ""; exit 3
fi
JOB="refcv8-battery-$TAG"
$PY "$BAT/gpu_lock.py" acquire --job $JOB --pid $$ --max-wait-s 172800 --out "$KITW/chain/battery_lock_$TAG.json" >> $LOG 2>&1
if ! $PY -I -c "import json,sys; sys.exit(0 if json.load(open(r'$KITW/chain/battery_lock_$TAG.json'))['ok'] else 1)"; then
  mark ZZR8BATTNOLOCKZZ ""; exit 3
fi
release() { $PY "$BAT/gpu_lock.py" release --job $JOB --pid $$ >> $LOG 2>&1; }
marker() { $PY "$BAT/chain_marker.py" "$1" --chain $JOB --pid $$ >> $LOG 2>&1; }
STOPW="$KIT/chain/battery_gpu_watch_$TAG.stop"; rm -f "$STOPW"
R8_BAT="$BAT" $PY "$PKGW/code/gpu_watch_r8.py" --out "$KITW/chain/battery_gpu_watch_$TAG.jsonl" --stop-file "$KITW/chain/battery_gpu_watch_$TAG.stop" >> $LOG 2>&1 &
marker create
trap ': > "$STOPW"; release; marker remove' EXIT
mark ZZR8BATTLOCKEDZZ ""; note "GPU lock acquired"; ts "lock acquired"
YARGS=""; [ "${R8_YIELD:-0}" = "1" ] && YARGS="--yield-job $JOB --yield-pid $$ --yield-log $KITW/chain/gpu_yields.jsonl"
# ---- 4/5. one battery PASS (resumable) --------------------------------------------------------------- #
pass_done() {   # $1 = pass suffix; the battery summary carries `bars` + `finished` + the r8 stamp only after a full run
  $PY -I -c "import json,sys; s=json.load(open(r'$BRW/${TAG}_$1/battery_summary.json',encoding='utf-8')); sys.exit(0 if (s.get('bars') and s.get('finished') and 'r8' in s) else 1)" 2>/dev/null
}
run_pass() {    # $1 = suffix, rest = extra run_battery_r8 args
  SUF=$1; shift
  $PY "$PKGW/code/run_battery_r8.py" --ckpt "$CK" --config "$CFG" --g0-json "$OUTW/g0.json" --g0-nonblocking \
      --tag "${TAG}_$SUF" --infer-seeds 0,1 --lock-held --skip-roll --out-root "$BRW" $YARGS "$@" \
      >> "$KIT/chain/battery_run_${TAG}_$SUF.log" 2>&1
}
# The pose-clock policy lives in the ENVIRONMENT of each pass; a subshell keeps it from leaking into the other pass (in POSIX mode
# `VAR=1 function` can persist after the function returns).
pass_unc() ( export R8_NO_POSE_SYNC=1; run_pass unc )
pass_corr() ( unset R8_NO_POSE_SYNC; run_pass corr --no-baselines --os-only-all )
# PASS 1 -- the standalone (refcv7-comparable) clock: the PAIRED read
ts "pass unc start"
pass_unc
if ! pass_done unc; then mark ZZR8BATTRETRYZZ "pass unc incomplete, retrying once (resumes from complete seed dumps)"; pass_unc; fi
ts "pass unc battery end"
if ! pass_done unc; then
  mark ZZR8BATTFAILZZ "pass unc: no complete battery_summary (see chain/battery_run_${TAG}_unc.log)"
  bank "FAILED: pass unc incomplete"
  exit 4
fi
mark ZZR8BATTPASSDONEZZ "unc (battery)"
R8_NO_POSE_SYNC=1 $PY "$PKGW/code/pair_read_r8.py" --root "$BRW/${TAG}_unc" --clock standalone --g0-json "$OUTW/g0.json" --config "$CFG" \
    --wrapfix-json "$RAWW/wrapper_clause_r8fix_gpu.json" --fp64-json "$RAWW/wrapper_fp64/result.json" \
    --out-json "$BRW/${TAG}_unc/PAIRED_READ_${TAG}_unc.json" --out-md "$BRW/${TAG}_unc/PAIRED_READ_${TAG}_unc.md" \
    > "$KIT/chain/pair_read_${TAG}_unc.log" 2>&1
ts "pass unc paired read end"
if [ -s "$BRL/${TAG}_unc/PAIRED_READ_${TAG}_unc.json" ]; then mark ZZR8BATTPAIREDZZ "unc"; else mark ZZR8BATTPAIREDNOJSONZZ "unc (see chain/pair_read_${TAG}_unc.log)"; fi
bank "pass unc complete (corrected-clock pass pending)"
if [ -s "$RAWW/BATTERY_SUMMARY_$TAG.json" ]; then mark ZZR8BATTDONEZZ "primary (standalone-clock, paired) pass banked: $RAWW/BATTERY_SUMMARY_$TAG.json"
else mark ZZR8BATTBANKFAILZZ ""; fi
note "primary pass done"
# PASS 2 -- the corrected clock (os-only at both seeds, no banked baselines): E1/E3
if [ "${R8_CORRECTED:-1}" = "1" ]; then
  ts "pass corr start"
  pass_corr
  if ! pass_done corr; then mark ZZR8BATTRETRYZZ "pass corr incomplete, retrying once"; pass_corr; fi
  ts "pass corr battery end"
  if pass_done corr; then
    mark ZZR8BATTPASSDONEZZ "corr (battery)"
    $PY "$PKGW/code/pair_read_r8.py" --root "$BRW/${TAG}_corr" --clock corrected --g0-json "$OUTW/g0.json" --config "$CFG" \
        --wrapfix-json "$RAWW/wrapper_clause_r8fix_gpu.json" --fp64-json "$RAWW/wrapper_fp64/result.json" \
        --out-json "$BRW/${TAG}_corr/PAIRED_READ_${TAG}_corr.json" --out-md "$BRW/${TAG}_corr/PAIRED_READ_${TAG}_corr.md" \
        > "$KIT/chain/pair_read_${TAG}_corr.log" 2>&1
    ts "pass corr paired read end"
    bank "both passes complete"
    mark ZZR8BATTCORRDONEZZ ""
  else
    mark ZZR8BATTCORRFAILZZ "pass corr incomplete (the primary pass is banked)"
    bank "primary complete; corrected-clock pass INCOMPLETE"
  fi
fi
# OPTIONAL (default OFF; R8_PERC=1): the refcv7 perception pass (map 10 cm + box) on the S2 windows, corrected clock.  RAW outputs only:
# the refcv7 scorer's bars (BAR-M7/B7) are refcv7's, and SPEC_REFCV8 2.3 item 5 refits the decision rules on refcv8's OWN
# TRAIN-DIAG pass, never on eval -- scoring is a separate step.
if [ "${R8_PERC:-0}" = "1" ]; then
  ts "perc start"
  PRIOR=""; [ -s /d/refcv7_eval_kit/prior/prior_train300.npz ] && PRIOR="--prior-npz D:/refcv7_eval_kit/prior/prior_train300.npz"
  mkdir -p "$KIT/perc/$TAG"
  $PY "$PKGW/code/perc_pass_r8.py" --ckpt "$CK" --config "$CFG" --windows-json "$KITW/windows/s2_windows.json"       --out "$KITW/perc/$TAG/perc" $PRIOR >> "$KIT/chain/perc8_$STEP.log" 2>&1
  ts "perc end"
  if [ -s "$KIT/perc/$TAG/perc.json" ]; then mark ZZR8BATTPERCDONEZZ "$KITW/perc/$TAG/perc.json"; else mark ZZR8BATTPERCNOJSONZZ "see chain/perc8_$STEP.log"; fi
fi
mark ZZR8BATTCHAINENDZZ ""
note "chain end"

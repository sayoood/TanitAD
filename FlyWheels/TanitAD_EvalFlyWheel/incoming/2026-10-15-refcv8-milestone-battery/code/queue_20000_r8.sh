#!/bin/sh
# refcv8 step-20,000 queue -- a COPY of queue_after_5000.sh (running; must not be edited): EMA battery first, then pull/G0/wrapfix/raw battery.  Started by wait_20000_r8.sh after the pull.
#   W   wait until the NEWEST segment of chain/battery_step5000.log ends (a terminal marker; or its shell pid gone for 5 polls AND nothing logged)
#   1   wrapfix  : GPU re-run of the G0 wrapper clause with amendment W-R8-1 (wrapper_clause_r8fix_gpu.py)  -> raw/step5000/wrapper_clause_r8fix_gpu.json
#   2   ema      : battery PASS 1 on the EMA weights of step 5,000 (`ckpt_ema_5000.pt`), standalone clock, paired vs refcv7-50,400, both inference
#                  seeds, tag step5000ema_unc, banked to raw/step5000ema/.  No G0 for these weights (--no-g0, reason stamped, PROVISIONAL).
#                  CPU gate first (ema_check_r8.py): strict load through the battery's builder AND weights differ from raw, else the stage STOPS.
#   3   milestone: step ${Q_MILESTONE} (default 10500): thor_pull_r8.sh -> chain_g0_r8.sh -> wrapfix probe -> chain_battery_r8.sh (pass 1; pass 2 only
#                  with Q_CORRECTED=1).  NOTE: there is NO step-10,000 checkpoint pair (the trainer saves ckpt_<N>.pt only at MILESTONES=(5000,15000,
#                  20000,30000), refc_v3_train.py:232); the milestone is the newest rolling ckpt.pt step that was snapshotted (10,500) -- see HANDOFF.
#   Each stage is independent: a failure is LOGGED and the next stage runs.  Every GPU step takes the shared O_EXCL lock (gpu_lock.py; a foreign
#   lock is never broken; the EXIT trap of the stage releases it).  Resumable: re-run the same command; a stage whose artifact already says done
#   is skipped (Q_FORCE=1 reruns).  Markers (opaque) in E:/refcv8_eval_kit/chain/queue_20000_r8.log; terminal marker = the QUEUEEND one.
# Assert on ARTIFACTS, never on an exit code, never through a pipe.  No git; nothing is written to the repo.
Q_STEP5="${Q_EMA_STEP:-20000}"
MS="${Q_MILESTONE:-20000}"
PKGW="${R8_PKGW:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
PKGL="${R8_PKGL:-/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
BAT="${R8_BAT:-C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code}"
KITW="${R8_KITW:-E:/refcv8_eval_kit}"
KIT="${R8_KIT:-/e/refcv8_eval_kit}"
PY="${R8_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
POLL="${R8_POLL_S:-60}"
STAGES="${Q_STAGES:-ema milestone}"
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6
unset R8_NO_POSE_SYNC
mkdir -p "$KIT/chain"
LOG="$KIT/chain/queue_20000_r8.log"
TIM="$KIT/chain/queue_timings.txt"
QH="$PKGW/code/queue_helpers_r8.py"
mark() { echo "ZZQ$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
note() { echo "[$(date +%FT%T)] queue_20000_r8: $*" >> "$KIT/KIT.log"; }
ts() { echo "$(date +%s) $(date +%FT%T) $1" >> "$TIM"; }
jq_ok() {   # $1 = file, $2 = python boolean expression over `j` ; exit 0 iff the file parses and the expression is true
  _f=$(cygpath -m "$1" 2>/dev/null || echo "$1")
  "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$_f"
}
# ---- single instance ------------------------------------------------------------------------------------- #
PIDF="$KIT/chain/queue_20000_r8.pid"
if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then mark DUP "another queue (pid $(cat "$PIDF")) is alive; exiting"; exit 5; fi
echo $$ > "$PIDF"
trap '[ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"' EXIT
mark QUEUESTART "pid $$ stages '$STAGES' milestone $MS"
note "armed (pid $$); stages: $STAGES; milestone $MS"; ts "armed"
# ---- shared-card helpers (the refcv7 battery's lock; mirrors chain_battery_r8.sh) ---------------------------- #
gpu_acquire() {   # $1 = job name ; 0 = lock held
  "$PY" "$BAT/wait_commit.py" --min-gb 6 --max-wait-s 43200 --out "$KITW/chain/queue_wc_$1.json" >> "$KIT/chain/queue_lockwait_$1.log" 2>&1
  jq_ok "$KIT/chain/queue_wc_$1.json" "j.get('met') is True" || { mark NOCOMMIT "$1"; return 3; }
  "$PY" "$BAT/gpu_lock.py" acquire --job "$1" --pid $$ --max-wait-s 172800 --out "$KITW/chain/queue_lock_$1.json" >> "$KIT/chain/queue_lockwait_$1.log" 2>&1
  jq_ok "$KIT/chain/queue_lock_$1.json" "j.get('ok') is True" || { mark NOLOCK "$1"; return 3; }
  "$PY" "$BAT/chain_marker.py" create --chain "$1" --pid $$ >> "$KIT/chain/queue_lockwait_$1.log" 2>&1
  mark LOCKED "$1"; note "GPU lock acquired: $1"; ts "lock $1"
}
gpu_release() {   # $1 = job name
  "$PY" "$BAT/gpu_lock.py" release --job "$1" --pid $$ >> "$KIT/chain/queue_lockwait_$1.log" 2>&1
  "$PY" "$BAT/chain_marker.py" remove --chain "$1" --pid $$ >> "$KIT/chain/queue_lockwait_$1.log" 2>&1
  mark RELEASED "$1"
}
# ---- W: wait for the current chain ----------------------------------------------------------------------- #
wait_chain() {
  n=0; dead=0; lastsz=0
  while :; do
    S=$("$PY" -I "$QH" battstate --log "$KITW/chain/battery_step$Q_STEP5.log" --step $Q_STEP5)
    case "$S" in
      ended:*) mark CHAINENDED "$S"; return 0 ;;
      nostart:*) mark CHAINNOSTART "$S (no battery chain segment to wait for)"; return 0 ;;
    esac
    CPID=${S##*:}
    SZ=$(wc -c < "$KIT/chain/battery_step$Q_STEP5.log" 2>/dev/null || echo 0)
    if [ "$CPID" != "-" ] && ! kill -0 "$CPID" 2>/dev/null && [ "$SZ" = "$lastsz" ]; then dead=$((dead+1)); else dead=0; fi
    lastsz=$SZ
    if [ "$dead" -ge 5 ]; then mark CHAINDEAD "shell $CPID gone, no terminal marker, silent log: $S"; return 0; fi
    n=$((n+1))
    [ $((n % 15)) -eq 1 ] && mark WAIT "battery chain state $S"
    [ "$n" -ge 2880 ] && { mark WAITTIMEOUT "48 h"; return 0; }
    sleep "$POLL"
  done
}
# ---- 1: the wrapper clause on the GPU ----------------------------------------------------------------------- #
wrapfix_one() (   # $1 = step, $2 = ckpt (Windows path), $3 = g0.json for the reproduction control, $4 = out json
  JOB="refcv8-wrapfix-step$1"
  CKM=$(md5sum "$2" | cut -d' ' -f1)
  [ "${#CKM}" -eq 32 ] || { mark WRAPFIXFAIL "cannot md5 $2"; exit 4; }
  if [ "${Q_FORCE:-0}" != "1" ] && jq_ok "$4" "j.get('ok') is True and j.get('CLAUSE') in ('PASS','FAIL','VOID') and j.get('ckpt_md5')=='$CKM'"; then mark WRAPFIXHAVE "$4"; exit 0; fi
  mark WRAPFIXSTART "step $1"; ts "wrapfix $1 start"
  gpu_acquire "$JOB" || exit 3
  trap 'gpu_release "$JOB"' EXIT
  "$PY" "$PKGW/code/wrapper_clause_r8fix_gpu.py" --ckpt "$2" --config "$KITW/ckpt/config.json" --g0-json "$3" --out "$4" > "$KIT/chain/wrapfix_step$1.log" 2>&1
  ts "wrapfix $1 end"
  if jq_ok "$4" "j.get('ok') is True and j.get('ckpt_md5')=='$CKM'"; then
    mark WRAPFIXOK "step $1 CLAUSE=$("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['CLAUSE'])" "$4")"
  else mark WRAPFIXFAIL "step $1 (see chain/wrapfix_step$1.log and $4)"; exit 4; fi
)
# ---- 2: battery pass 1 on the EMA weights of step 5,000 ----------------------------------------------------- #
ema_stage() (
  TAG=step${Q_STEP5}ema; RAWW="$PKGW/raw/$TAG"; RAWL="$PKGL/raw/$TAG"; WORK="$KITW/battery/${TAG}_unc"
  CK="$KITW/ckpt/ckpt_$Q_STEP5.pt"; EMA="$KITW/ckpt/ckpt_ema_$Q_STEP5.pt"; CFG="$KITW/ckpt/config.json"; JOB="refcv8-battery-$TAG"
  mkdir -p "$RAWL"
  if [ "${Q_FORCE:-0}" != "1" ] && jq_ok "$RAWL/BATTERY_SUMMARY_$TAG.json" "j['passes']['unc']['status']=='COMPLETE'"; then mark EMAHAVE "$RAWW"; exit 0; fi
  mark EMASTART "$TAG"; ts "ema start"
  # CPU gate: strict load + differs from raw (STOP if identical)
  GATEJ="$RAWW/EMA_CHECK_step$Q_STEP5.json"
  M_RAW=$(md5sum "$CK" | cut -d' ' -f1); M_EMA=$(md5sum "$EMA" | cut -d' ' -f1)
  [ "${#M_RAW}" -eq 32 ] && [ "${#M_EMA}" -eq 32 ] || { mark EMAGATEFAIL "cannot md5 the checkpoints"; exit 4; }
  PIN="j['md5']['raw']=='$M_RAW' and j['md5']['ema']=='$M_EMA'"
  if ! jq_ok "$GATEJ" "j.get('ok') is True and $PIN"; then
    "$PY" "$PKGW/code/ema_check_r8.py" --raw "$CK" --ema "$EMA" --config "$CFG" --out "$GATEJ" > "$KIT/chain/ema_check_$Q_STEP5.out" 2>&1
  fi
  jq_ok "$GATEJ" "j.get('ok') is True and $PIN and j['tensors']['n_float_differing']>0 and j['strict_load']['missing']==0 and j['strict_load']['unexpected']==0" \
    || { mark EMAGATEFAIL "ema_check not ok: $GATEJ (identical weights or strict-load defect) -- stage STOPPED"; exit 4; }
  mark EMAGATEOK "strict load clean, weights differ"
  gpu_acquire "$JOB" || exit 3
  STOPW="$KIT/chain/queue_gpu_watch_$TAG.stop"; rm -f "$STOPW"
  "$PY" "$BAT/gpu_watch.py" --out "$KITW/chain/queue_gpu_watch_$TAG.jsonl" --stop-file "$KITW/chain/queue_gpu_watch_$TAG.stop" >> "$KIT/chain/queue_lockwait_$JOB.log" 2>&1 &
  trap ': > "$STOPW"; gpu_release "$JOB"' EXIT
  mkdir -p "$KIT/battery/seam_tele_$TAG"
  run_pass() {
    ( export R8_NO_POSE_SYNC=1 R8_SEAM_TELE_DIR="$KITW/battery/seam_tele_$TAG"
      "$PY" "$PKGW/code/run_battery_r8.py" --ckpt "$EMA" --config "$CFG" --no-g0 --tag "${TAG}_unc" --infer-seeds 0,1 --lock-held \
          --skip-roll --out-root "$KITW/battery" >> "$KIT/chain/battery_run_${TAG}_unc.log" 2>&1 )
  }
  pass_done() { jq_ok "$WORK/battery_summary.json" "j.get('bars') and j.get('finished') and 'r8' in j"; }
  ts "ema pass start"
  run_pass
  if ! pass_done; then mark EMARETRY "pass incomplete, retrying once (resumes from complete seed dumps)"; run_pass; fi
  ts "ema pass end"
  bank() { "$PY" "$PKGW/code/bank_battery_r8.py" "$KITW/battery" "$RAWW" --tag "$TAG" --passes unc --chain-log "$KITW/chain/queue_20000_r8.log" \
             --timings "$KITW/chain/queue_timings.txt" --status "$1" >> "$LOG" 2>&1; }
  if ! pass_done; then mark EMAFAIL "no complete battery_summary (see chain/battery_run_${TAG}_unc.log)"; bank "FAILED: EMA pass 1 incomplete"; exit 4; fi
  mark EMAPASSDONE "unc (battery)"
  ( export R8_NO_POSE_SYNC=1
    "$PY" "$PKGW/code/pair_read_r8.py" --root "$WORK" --clock standalone --config "$CFG" \
        --out-json "$WORK/PAIRED_READ_${TAG}_unc.json" --out-md "$WORK/PAIRED_READ_${TAG}_unc.md" > "$KIT/chain/pair_read_${TAG}_unc.log" 2>&1 )
  ts "ema paired read end"
  if [ -s "$KIT/battery/${TAG}_unc/PAIRED_READ_${TAG}_unc.json" ]; then mark EMAPAIRED "unc"; else mark EMAPAIREDNOJSON "see chain/pair_read_${TAG}_unc.log"; fi
  # the stamp the pair-read cannot carry (it only knows "no g0.json")
  "$PY" -I -c "
import json,sys
reason=('EMA weights of the same step; this queue runs the EMA pass FIRST and the G0 of the raw weights at this step AFTER it (milestone stage); at step 5,000 G0 read FAIL as coded, M1 detected, A7; '
        'wrapper clause amended W-R8-1)')
json.dump({'tag':sys.argv[2],'status':'PROVISIONAL','no_g0_reason':reason,'ckpt':sys.argv[3],'raw_ckpt':sys.argv[4],
           'g0_json_of_raw_weights':sys.argv[5],'amendment':'W-R8-1','note':'every number from this pass is PROVISIONAL; no G0 exists for the EMA weights'},
          open(sys.argv[1],'w',encoding='utf-8'),indent=1)" "$RAWW/NO_G0_REASON_$TAG.json" "$TAG" "$EMA" "$CK" "$KITW/battery/step$Q_STEP5/g0.json"
  cp "$RAWL/NO_G0_REASON_$TAG.json" "$KIT/battery/${TAG}_unc/NO_G0_REASON_$TAG.json" 2>/dev/null
  bank "PROVISIONAL: EMA weights, no G0 for them (reason in NO_G0_REASON_$TAG.json); pass unc complete"
  if [ -s "$RAWL/BATTERY_SUMMARY_$TAG.json" ]; then mark EMADONE "banked: $RAWW/BATTERY_SUMMARY_$TAG.json"; else mark EMABANKFAIL ""; fi
  note "EMA battery pass 1 done"
)
# ---- 3: the milestone (pull -> G0 -> wrapper probe -> battery) ------------------------------------------------ #
milestone_stage() (
  S=$MS; TAGM=step$S; RAWL="$PKGL/raw/$TAGM"
  mkdir -p "$RAWL"
  if [ "${Q_FORCE:-0}" != "1" ] && jq_ok "$RAWL/BATTERY_SUMMARY_$TAGM.json" "j['passes']['unc']['status']=='COMPLETE'"; then mark MSHAVE "$RAWL"; exit 0; fi
  mark MSSTART "step $S"; ts "milestone $S start"
  sh "$PKGL/code/thor_pull_r8.sh" "$S" > "$KIT/chain/queue_pull_$S.out" 2>&1
  PR=$(tail -n 1 "$KIT/chain/queue_pull_$S.out")
  case "$PR" in pull:ok:*) mark MSPULLOK "$PR" ;; *) mark MSPULLFAIL "$PR (see chain/thor_pull_step$S.log)"; exit 5 ;; esac
  ts "milestone pull end"
  jq_ok "$RAWL/EMA_CHECK_$TAGM.json" "j.get('ok') is True" || \
    "$PY" "$PKGW/code/ema_check_r8.py" --raw "$KITW/ckpt/ckpt_$S.pt" --ema "$KITW/ckpt/ckpt_ema_$S.pt" --config "$KITW/ckpt/config.json" \
        --out "$PKGW/raw/$TAGM/EMA_CHECK_$TAGM.json" > "$KIT/chain/ema_check_$S.out" 2>&1
  # G0 (own lock, own markers); skipped when this checkpoint already has its G0 artifact
  if ! jq_ok "$KIT/battery/$TAGM/g0.json" "j.get('ckpt_md5') and j.get('verdict')"; then
    sh "$PKGL/code/chain_g0_r8.sh" "$S" > "$KIT/chain/queue_g0_$S.out" 2>&1
  fi
  ts "milestone g0 end"
  if jq_ok "$KIT/battery/$TAGM/g0.json" "j.get('ckpt_md5') and j.get('verdict')"; then
    mark MSG0OK "$("$PY" -I -c "import json,sys; v=json.load(open(sys.argv[1]))['verdict']; print('G0',v.get('G0'),'M1',((v.get('mutation_detection') or {}).get('m1') or {}).get('detected'))" "$KIT/battery/$TAGM/g0.json")"
  else mark MSG0NOJSON "no g0.json for step $S (see chain/chain_$TAGM.log); the battery chain will log its skip"; fi
  # the amended wrapper clause at this step (non-blocking extra: the as-coded probe is blind to refcv8's noise draws)
  if [ "${Q_WRAPFIX_MS:-1}" = "1" ] && jq_ok "$KIT/battery/$TAGM/g0.json" "j.get('ckpt_md5')"; then
    wrapfix_one "$S" "$KITW/ckpt/ckpt_$S.pt" "$KITW/battery/$TAGM/g0.json" "$PKGW/raw/$TAGM/wrapper_clause_r8fix_gpu.json" || mark MSWRAPFIXFAILED "non-blocking"
  fi
  # the battery chain (waits on THIS step's G0 log, gates on THIS step's g0.json, takes the lock, pass 1 (+ pass 2 iff Q_CORRECTED=1))
  mkdir -p "$KIT/battery/seam_tele_$TAGM"
  ( export R8_SEAM_TELE_DIR="$KITW/battery/seam_tele_$TAGM" R8_CORRECTED="${Q_CORRECTED:-0}"
    sh "$PKGL/code/chain_battery_r8.sh" "$S" > "$KIT/chain/queue_battery_$S.out" 2>&1 )
  ts "milestone battery end"
  if jq_ok "$RAWL/BATTERY_SUMMARY_$TAGM.json" "j['passes']['unc']['status']=='COMPLETE'"; then mark MSDONE "banked: $PKGW/raw/$TAGM/BATTERY_SUMMARY_$TAGM.json"
  else mark MSINCOMPLETE "no COMPLETE unc pass in $PKGW/raw/$TAGM/BATTERY_SUMMARY_$TAGM.json (skipped / failed: read it and chain/battery_$TAGM.log)"; fi
)
# ---- drive ------------------------------------------------------------------------------------------------------------ #
[ "${Q_NOWAIT:-0}" = "1" ] || wait_chain
ts "chain ended"
for st in $STAGES; do
  case "$st" in
    wrapfix)   wrapfix_one "$Q_STEP5" "$KITW/ckpt/ckpt_$Q_STEP5.pt" "$KITW/battery/step$Q_STEP5/g0.json" "$PKGW/raw/step$Q_STEP5/wrapper_clause_r8fix_gpu.json" \
                 || mark STAGEFAILED "wrapfix (continuing)" ;;
    ema)       ema_stage || mark STAGEFAILED "ema (continuing)" ;;
    milestone) milestone_stage || mark STAGEFAILED "milestone (continuing)" ;;
    *) mark STAGEUNKNOWN "$st" ;;
  esac
done
mark QUEUEEND "all stages attempted"
note "queue end"

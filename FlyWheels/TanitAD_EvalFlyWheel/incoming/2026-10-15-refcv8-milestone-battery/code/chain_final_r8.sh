#!/bin/sh
# chain_final_r8.sh -- the FINAL-CHECKPOINT chain (refcv8 step 48,852): pull -> weights -> EMA battery -> K-draw rolls -> compliant bars -> frozen-gain + train-fit forwards -> raw battery -> heading -> K-draw read -> combo.   usage: sh chain_final_r8.sh
#   NEW file (2026-10-19, EvalFlyWheel).  Derived from chain_40k_r8.sh (which it does not edit and does not replace: the 40k chain keeps running on its own files).  Written for the registered final reads
#   (kdraw/HANDOFF.md step 3, combo_ens_retime/HANDOFF.md step 2).  Markers (opaque ZZCFN...ZZ) -> E:/refcv8_eval_kit/chain/chain_final.log.  Terminal marker: ZZCFNENDZZ (line says every stage's state) or ZZCFNABORTZZ (stage a or b).
#   Resumable: re-run the same command; a stage whose artifact already says done is skipped.  Assert on ARTIFACTS, never an exit code, never through a pipe.  Thor is touched ONLY by pull_final_r8.sh and
#   thor_pull_r8.sh (read-only: stat / md5sum / grep / cat / scp FROM it).  Nothing is written to G:.  No git.   Plan + input status: raw/step48852/PLAN.md.
#
#   WHAT THE TRAINER WRITES AT THE FINAL STEP (refc_v3_train.py, quoted in pull_final_r8.sh): NO ckpt_48852.pt / ckpt_ema_48852.pt (MILESTONES = 5000, 15000, 20000, 30000, :232); ckpt.pt (full, incl. ['ema']) is
#   written in place at :10822 (and stays: nothing overwrites it after the last step); model_ema.pt at :10840-10842 (only if the loop ran the last step); summary.json at :11039 -- later OVERWRITTEN by the supervisor
#   with a different schema (sup_refcv7.sh:201-215), and written there even if the final save never happened.  So summary.json is never the proof; the step INSIDE ckpt.pt is.
#
#   a  PULL     pull_final_r8.sh (waits for done-marker + train.log's `ckpt step 48852 -> ckpt.pt` line + stat settle; scp; md5 on both ends) -> ckpt_full_48852.pt + sidecar + provenance.  The chain RE-VERIFIES: local md5 ==
#               sidecar == provenance == the DONE marker's md5, all 32 hex (an empty string is INCONCLUSIVE, never a match), and the copied summary.json says done at 48852.
#   b  EXTRACT  extract_ema_r8.py extract --expect-step 48852 (the step INSIDE the file) -> ckpt_48852.pt + ckpt_ema_48852.pt; the chain re-asserts step==48852, ok==true, src_md5 on the provenance ARTIFACT; EMA gate
#               (ema_check_r8.py: strict load clean, weights differ from raw, md5-pinned); XCHECK (xcheck_model_ema_r8.py): the trainer-written model_ema.pt == the derived EMA on every tensor (N/A if the trainer wrote none).
#   c  EMA      battery pass 1 on the EMA weights, tag step48852ema_unc, inference seeds 0,1, standalone clock, shared GPU lock, one retry, pair_read vs refcv7-50,400, bank.   [as chain_40k ema_stage]
#   k  KDRAW    kdraw_queue_r8.sh with KD_STEP=48852 (ctrl + seeds 2..7, os-only, the banked windows byte for byte; one lock hold; RELEASE_R7 NOT rewritten) -> kdraw/step48852ema_unc/.  Needs stage c's banked tree.
#   f  RAW      thor_pull (metrics copy) -> chain_g0_r8.sh -> wrapper-clause probe (non-blocking) -> chain_battery_r8.sh (pass 1 only), tag step48852_unc.   [as chain_40k raw_stage]
#   CPU LANE    background, memory-gated, EMA tree first then RAW tree:  d clamp_posthoc read -> compliant_reread read (A-R8-CLAMP-2 capped plans on ceiling-compliant windows; E1/E3 vs ha0_ext_pose with ha0_ext beside,
#               D-R8-E1-REF)   e the three frozen-gain forwards (30k files UNCHANGED)   g heading_drift (earlier list built from the trees that EXIST)   then the TRAIN-FIT forward when FROZEN_GAINS_RETIME_train.json
#               exists (waits up to CFN_TF_WAIT_S for the train-fit chain)   then render x3 and summarize_40k_r8.py --step 48852.
#   TAIL        run_kdraw_analysis_r8.sh step48852ema_unc (right after stage k) and run_combo_ens_retime_r8.sh step48852ema_unc train (after the lane; refuses on a missing precondition and says which), then final_index_r8.py.
#   G0 POLICY (mirrors 30k/40k, STATED): the EMA weights get NO G0 (`--no-g0`, NO_G0_REASON stamped, PROVISIONAL); the RAW weights get chain_g0_r8.sh and the raw battery runs even on a FAIL-as-coded G0 when M1 was detected.
#   GPU LOCK: the SAME O_EXCL lock and 60 s acquire as every other chain (battery/code/gpu_lock.py).  It has NO priority mechanism and cannot be preempted: this chain queues behind whoever holds it.  No fast-poll
#   acquirer was built (it would race other agents' jobs).  yield48852_r8.sh is the yield for chains that start after this file existed.   Paths are overridable by R8_*/CFN_* env ONLY so the control flow can be dry-run.
STEP="${CFN_STEP:-48852}"
PKGW="${R8_PKGW:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
PKGL="${R8_PKGL:-/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
BAT="${R8_BAT:-C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code}"
KITW="${R8_KITW:-E:/refcv8_eval_kit}"
KIT="${R8_KIT:-/e/refcv8_eval_kit}"
BRW="${R8_BATT_ROOT_W:-C:/Users/Admin/refcv8_eval_archive/battery}"; BRL="${R8_BATT_ROOT_L:-/c/Users/Admin/refcv8_eval_archive/battery}"
export R8_PKGW="$PKGW" R8_PKGL="$PKGL" R8_KITW="$KITW" R8_KIT="$KIT" R8_BATT_ROOT_W="$BRW" R8_BATT_ROOT_L="$BRL" R8_BAT="$BAT"
PY="${R8_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
EXTRACT_PY="${CFN_EXTRACT_PY:-$PKGW/code/extract_ema_r8.py}"
XCHECK_PY="${CFN_XCHECK_PY:-$PKGW/code/xcheck_model_ema_r8.py}"
PULLSH="${CFN_PULL_SH:-$PKGL/code/pull_final_r8.sh}"
KDQ="${CFN_KDQ:-$PKGL/code/kdraw_queue_r8.sh}"
PULLLOG="${CFN_PULL_LOG:-$KIT/chain/pull_final.log}"
FULL="${CFN_FULL:-$KITW/ckpt/ckpt_full_$STEP.pt}"; FULLL="${CFN_FULL_L:-$KIT/ckpt/ckpt_full_$STEP.pt}"
POLL="${CFN_POLL_S:-120}"; LPOLL="${CFN_LANE_POLL_S:-60}"
LANE_MAXWAIT="${CFN_LANE_MAXWAIT_S:-345600}"
CPUNEED="${CFN_CPU_NEED_GB:-5}"; EXNEED="${CFN_EXTRACT_NEED_GB:-5}"; MIN_COMMIT="${CFN_MIN_COMMIT_GB:-4}"
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6
unset R8_NO_POSE_SYNC
CK="$KITW/ckpt/ckpt_$STEP.pt"; EMA="$KITW/ckpt/ckpt_ema_$STEP.pt"; CFG="$KITW/ckpt/config.json"; CKL="$KIT/ckpt"
TAGE="step${STEP}ema"; TAGR="step$STEP"; TE="${TAGE}_unc"; TR="${TAGR}_unc"
RAWE_W="$PKGW/raw/$TAGE"; RAWE_L="$PKGL/raw/$TAGE"; RAWR_W="$PKGW/raw/$TAGR"; RAWR_L="$PKGL/raw/$TAGR"
LOGD_L="$RAWR_L/logs"
LOG="$KIT/chain/chain_final.log"; ST="$KIT/chain/chain_final.state"; TIM="$KIT/chain/chain_final_timings.txt"
mkdir -p "$KIT/chain" "$ST" "$RAWE_L" "$RAWR_L" "$LOGD_L"
mark() { echo "ZZCFN$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
note() { echo "[$(date +%FT%T)] chain_final_r8: $*" >> "$KIT/KIT.log"; }
ts() { echo "$(date +%s) $(date +%FT%T) $1" >> "$TIM"; }
jq_ok() {   # $1 = file, $2 = python boolean expression over `j`; exit 0 iff the file parses and the expression is true
  _f=$(cygpath -m "$1" 2>/dev/null || echo "$1")
  "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$_f"
}
jget() {   # $1 = file, $2 = python expression over `j`; prints it ('' on any error)
  _f=$(cygpath -m "$1" 2>/dev/null || echo "$1")
  "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); print($2)
except Exception: print('')" "$_f"
}
free_gb() {
  if [ -n "$CFN_FREEGB_CMD" ]; then v=$(sh -c "$CFN_FREEGB_CMD" 2>/dev/null)
  else v=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" 2>/dev/null | tr -d '\r ' | tr ',' '.'); fi
  case "$v" in ''|*[!0-9.]*) echo 0 ;; *) echo "$v" ;; esac
}
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit (a+0>=b+0)?0:1}'; }
mem_gate() {   # $1 = GB needed, $2 = job: two checks 20 s apart (the dev box COMMIT limit binds, not RAM); returns 1 after CFN_MEM_MAXWAIT_S
  _t0=$(date +%s); _n=0
  while :; do
    _h1=$(free_gb)
    if ge "$_h1" "$1"; then sleep "${CFN_GATE_GAP_S:-20}"; _h2=$(free_gb); ge "$_h2" "$1" && return 0; fi
    _n=$((_n+1)); [ $((_n % 20)) -eq 1 ] && mark MEMWAIT "$2: free commit ${_h1} GB < $1"
    [ $(( $(date +%s) - _t0 )) -gt "${CFN_MEM_MAXWAIT_S:-43200}" ] && { mark MEMTIMEOUT "$2"; return 1; }
    sleep "${CFN_MEM_POLL_S:-30}"
  done
}
# ---- single instance ------------------------------------------------------------------------------------- #
PIDF="$KIT/chain/chain_final.pid"
if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then mark DUP "another chain_final (pid $(cat "$PIDF")) is alive; exiting"; exit 5; fi
echo $$ > "$PIDF"
trap '[ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"' EXIT
trap 'exit 143' TERM INT HUP
rm -f "$ST"/*.state
abort() { mark ABORT "stage $1: $2"; note "ABORT stage $1: $2"; exit "$3"; }
mark START "pid $$ step $STEP tags $TE $TR"; note "armed (pid $$)"; ts "armed"
# ---- a: pull the final checkpoint ----------------------------------------------------------------------- #
pull_state() {   # none | wait | done | already | fail:<MARKER>   (the LAST ZZPFNSTART segment of pull_final.log)
  awk '/ZZPFNSTARTZZ/ {st="wait"; next}
       /ZZPFNDONEZZ/ {st="done"; next}
       /ZZPFNALREADYZZ/ {if (st != "done") st="already"; next}
       /ZZPFN(SSHDEAD|SUPDEAD|TIMEOUT|WRONGSTEP|HOLE|SETTLEFAIL|NODISK|BADSIZE|INCONCLUSIVE|CHANGED|SIZEMISMATCH|MD5MISMATCH)ZZ/ {match($0, /ZZPFN[A-Z0-9]*ZZ/); st="fail:" substr($0, RSTART, RLENGTH); next}
       END {print (st == "" ? "none" : st)}' "$PULLLOG" 2>/dev/null
}
pull_stage() {
  mark PULLSTART "$PULLSH"
  FIN_KIT="$KIT" FIN_CKPT_DIR="$CKL" FIN_LOG="$PULLLOG" FIN_STEP="$STEP" sh "$PULLSH" > "$KIT/chain/cfn_pull.out" 2>&1
  S=$(pull_state)
  case "$S" in
    done|already) ;;
    fail:*) mark PULLFAIL "pull_final reached ${S#fail:}: ckpt_full_$STEP.pt is NOT trustworthy (or the run did not finish); chain stops"; return 20 ;;
    *) mark PULLFAIL "pull_final ended in state '$S' with no terminal marker (see chain/cfn_pull.out): chain stops"; return 22 ;;
  esac
  mark PULLSEEN "pull_final state $S"
  # the local md5 re-verified against THREE independent records, shape asserted before comparing (an empty string must never read as a match)
  [ -s "$FULLL" ] || { mark MD5BAD "no $FULLL"; return 21; }
  LMD=$(md5sum "$FULLL" | cut -d' ' -f1)
  SIDE=$(cut -d' ' -f1 < "$FULLL.md5" 2>/dev/null | head -n 1)
  PROV=$(jget "$FULLL.provenance.json" "j['md5']")
  PSTEP=$(jget "$FULLL.provenance.json" "j['step']")
  DONEMD=$(grep -a 'ZZPFNDONEZZ' "$PULLLOG" 2>/dev/null | tail -n 1 | sed -n 's/.* ckpt_md5 \([0-9a-f]\{32\}\) .*/\1/p')
  if [ "$S" = already ] && [ -z "$DONEMD" ]; then DONEMD="$PROV"; fi    # ALREADY has no DONE line: provenance stands in (still needs sidecar + provenance + local)
  for m in "$LMD" "$SIDE" "$PROV" "$DONEMD"; do
    [ "${#m}" -eq 32 ] || { mark MD5INCONCLUSIVE "an md5 is not 32 chars (local '$LMD' sidecar '$SIDE' provenance '$PROV' done '$DONEMD'): never a match"; return 21; }
  done
  [ "$PSTEP" = "$STEP" ] || { mark MD5BAD "provenance step '$PSTEP' != $STEP"; return 21; }
  if [ "$LMD" = "$SIDE" ] && [ "$LMD" = "$PROV" ] && [ "$LMD" = "$DONEMD" ]; then SNAPMD5="$LMD"; mark MD5OK "ckpt_full_$STEP.pt md5 $SNAPMD5 (local == sidecar == provenance == DONE marker)"
  else mark MD5BAD "local $LMD sidecar $SIDE provenance $PROV done $DONEMD"; return 21; fi
  # the copied summary.json: the trainer's form has `step`, the supervisor's overwrite has `final_step`.  Necessary, not sufficient (stage b reads the step INSIDE the file).
  if jq_ok "$CKL/summary_thor_$STEP.json" "j.get('done') is True and int(j.get('step', j.get('final_step', -1)))==$STEP"; then mark SUMJSONOK "summary.json done@$STEP"
  else mark SUMJSONBAD "the pulled summary.json does not say done@$STEP (see $CKL/summary_thor_$STEP.json)"; return 21; fi
  return 0
}
pull_stage; RC=$?; [ $RC -eq 0 ] || abort a "final checkpoint not usable" $RC
ts "pull verified"
# ---- b: extract + EMA gate + cross-check ------------------------------------------------------------------ #
EXREC="$KITW/ckpt/ckpt_$STEP.provenance.extract_ema.json"
extract_ok() {   # the pair + a provenance that says ok, step == $STEP (the step INSIDE the file) and the md5 of THIS pull
  [ -s "$CKL/ckpt_$STEP.pt" ] && [ -s "$CKL/ckpt_ema_$STEP.pt" ] && \
    jq_ok "$EXREC" "j.get('ok') is True and int(j.get('step',-1))==$STEP and j.get('src_md5')=='$SNAPMD5'"
}
extract_done() { extract_ok && [ -s "$CKL/ckpt_$STEP.md5" ]; }   # + the md5 record written after the extract
extract_stage() {
  if extract_done; then mark EXTRACTHAVE "pair already derived from this pull"; return 0; fi
  FREEKB=$(df -k "$CKL" | tail -n 1 | awk '{print $4}')
  [ -n "$FREEKB" ] && [ "$FREEKB" -lt "${CFN_MIN_FREE_KB:-3500000}" ] && { mark EXTRACTNODISK "free ${FREEKB} kB on $CKL"; return 24; }
  mem_gate "$EXNEED" extract || return 25
  mark EXTRACTSTART "expect-step $STEP, src $FULL"
  FORCE=""; { [ -e "$CKL/ckpt_$STEP.pt" ] || [ -e "$CKL/ckpt_ema_$STEP.pt" ]; } && FORCE="--force"     # a partial/old attempt: DERIVED files, the pull is the source
  "$PY" "$EXTRACT_PY" extract --src "$FULL" --config "$CFG" --out-dir "$KITW/ckpt" --expect-step "$STEP" --min-commit-gb "$MIN_COMMIT" $FORCE \
      --provenance "thor_ckpt_pt_md5=$SNAPMD5" "snapshot=$FULL" "pull_log=$PULLLOG" "chain=chain_final_r8.sh" > "$KIT/chain/cfn_extract.out" 2>&1
  # the step is re-asserted on the ARTIFACT (not trusted from the exit code or from --expect-step alone)
  if extract_ok; then :; else
    mark EXTRACTFAIL "no ok provenance for step $STEP: $(tail -n 1 "$KIT/chain/cfn_extract.out" | cut -c1-300)"; return 21; fi
  M_RAW=$(md5sum "$CKL/ckpt_$STEP.pt" | cut -d' ' -f1); M_EMA=$(md5sum "$CKL/ckpt_ema_$STEP.pt" | cut -d' ' -f1)
  { [ "${#M_RAW}" -eq 32 ] && [ "${#M_EMA}" -eq 32 ]; } || { mark EXTRACTFAIL "cannot md5 the derived pair"; return 21; }
  { echo "$M_RAW  ckpt_$STEP.pt"; echo "$M_EMA  ckpt_ema_$STEP.pt"; echo "$SNAPMD5  ckpt_full_$STEP.pt (thor final ckpt.pt, step $STEP)"; } > "$CKL/ckpt_$STEP.md5.tmp" && mv -f "$CKL/ckpt_$STEP.md5.tmp" "$CKL/ckpt_$STEP.md5"
  mark EXTRACTOK "ckpt_$STEP.pt $M_RAW ckpt_ema_$STEP.pt $M_EMA"
}
ema_gate() {   # strict load + differs from raw, md5-pinned; the EMA stage STOPS otherwise
  GATEJ="$RAWE_W/EMA_CHECK_step$STEP.json"
  M_RAW=$(md5sum "$CKL/ckpt_$STEP.pt" | cut -d' ' -f1); M_EMA=$(md5sum "$CKL/ckpt_ema_$STEP.pt" | cut -d' ' -f1)
  PIN="j['md5']['raw']=='$M_RAW' and j['md5']['ema']=='$M_EMA'"
  if ! jq_ok "$GATEJ" "j.get('ok') is True and $PIN"; then
    mem_gate "$EXNEED" ema_check || return 25
    "$PY" "$PKGW/code/ema_check_r8.py" --raw "$CK" --ema "$EMA" --config "$CFG" --out "$GATEJ" > "$KIT/chain/cfn_ema_check.out" 2>&1
  fi
  if jq_ok "$GATEJ" "j.get('ok') is True and $PIN and j['tensors']['n_float_differing']>0 and j['strict_load']['missing']==0 and j['strict_load']['unexpected']==0"; then
    cp -f "$RAWE_L/EMA_CHECK_step$STEP.json" "$RAWR_L/EMA_CHECK_step$STEP.json" 2>/dev/null; mark EMAGATEOK "strict load clean, weights differ from raw"; return 0
  fi
  mark EMAGATEFAIL "ema_check not ok / identical weights / strict-load defect: $GATEJ"; return 21
}
xcheck_stage() {   # the trainer-written model_ema.pt == the EMA derived from ckpt.pt['ema'], tensor by tensor.  CFN_XCHECK=strict (default: stop) | warn (record, continue) | off
  TH="$CKL/model_ema_${STEP}_thor.pt"; XJ="$RAWE_W/XCHECK_MODEL_EMA_step$STEP.json"
  [ "${CFN_XCHECK:-strict}" = off ] && { mark XCHECKOFF "CFN_XCHECK=off"; return 0; }
  if [ ! -s "$TH" ]; then mark XCHECKNA "no trainer-written model_ema.pt was pulled (a resume AT step==steps skips that write): ckpt.pt['ema'] is the only EMA record, cross-checked inside extract_ema_r8 only"; return 0; fi
  M_TH=$(md5sum "$TH" | cut -d' ' -f1); M_EMA=$(md5sum "$CKL/ckpt_ema_$STEP.pt" | cut -d' ' -f1)
  { [ "${#M_TH}" -eq 32 ] && [ "${#M_EMA}" -eq 32 ]; } || { mark XCHECKFAIL "cannot md5 the two EMA files"; return 21; }
  PINX="j['md5']['thor']=='$M_TH' and j['md5']['derived']=='$M_EMA'"
  if jq_ok "$XJ" "j.get('ok') is True and $PINX"; then mark XCHECKHAVE "pinned to these two md5s"; return 0; fi
  mem_gate "$EXNEED" xcheck || return 25
  "$PY" "$XCHECK_PY" --thor-ema "$KITW/ckpt/model_ema_${STEP}_thor.pt" --derived "$EMA" --step "$STEP" --out "$XJ" > "$KIT/chain/cfn_xcheck.out" 2>&1
  if jq_ok "$XJ" "j.get('ok') is True and $PINX"; then mark XCHECKOK "trainer model_ema.pt == derived ckpt_ema_$STEP.pt on every tensor ($(jget "$XJ" "j['n_tensors']") tensors)"; return 0; fi
  if [ "${CFN_XCHECK:-strict}" = warn ]; then mark XCHECKWARN "MISMATCH recorded in $XJ; continuing because CFN_XCHECK=warn: every EMA number is then UNVERIFIED against the trainer's own EMA file"; return 0; fi
  mark XCHECKFAIL "the trainer-written EMA file and the derived EMA DISAGREE (or the check could not run): $XJ. Re-run with CFN_XCHECK=warn only after a human has read it"; return 21
}
extract_stage; RC=$?; [ $RC -eq 0 ] || abort b "extract" $RC
ema_gate; RC=$?; [ $RC -eq 0 ] || abort b "ema gate" $RC
xcheck_stage; RC=$?; [ $RC -eq 0 ] || abort b "ema cross-check" $RC
ts "extract + ema gate + xcheck done"
# ---- GPU lock helpers (the SAME lock, the SAME 60 s acquire as every other chain) ---------------------------- #
gpu_acquire() {   # $1 = job name ; 0 = lock held
  rm -f "$KITW/chain/cfn_wc_$1.json" "$KITW/chain/cfn_lock_$1.json" 2>/dev/null     # a stale ok=true must not pass for this acquire
  "$PY" "$BAT/wait_commit.py" --min-gb 6 --max-wait-s 43200 --out "$KITW/chain/cfn_wc_$1.json" >> "$KIT/chain/cfn_lockwait_$1.log" 2>&1
  jq_ok "$KIT/chain/cfn_wc_$1.json" "j.get('met') is True" || { mark NOCOMMIT "$1"; return 3; }
  "$PY" "$BAT/gpu_lock.py" acquire --job "$1" --pid $$ --max-wait-s 172800 --out "$KITW/chain/cfn_lock_$1.json" >> "$KIT/chain/cfn_lockwait_$1.log" 2>&1
  jq_ok "$KIT/chain/cfn_lock_$1.json" "j.get('ok') is True" || { mark NOLOCK "$1"; return 3; }
  "$PY" "$BAT/chain_marker.py" create --chain "$1" --pid $$ >> "$KIT/chain/cfn_lockwait_$1.log" 2>&1
  mark LOCKED "$1"; note "GPU lock acquired: $1"; ts "lock $1"
}
gpu_release() {
  "$PY" "$BAT/gpu_lock.py" release --job "$1" --pid $$ >> "$KIT/chain/cfn_lockwait_$1.log" 2>&1
  "$PY" "$BAT/chain_marker.py" remove --chain "$1" --pid $$ >> "$KIT/chain/cfn_lockwait_$1.log" 2>&1
  mark RELEASED "$1"
}
# ---- c: battery pass 1 on the EMA weights ------------------------------------------------------------------ #
ema_stage() (
  WORK="$BRW/$TE"; JOB="refcv8-battery-$TAGE"
  if jq_ok "$RAWE_L/BATTERY_SUMMARY_$TAGE.json" "j['passes']['unc']['status']=='COMPLETE'"; then mark EMAHAVE "$RAWE_W"; exit 0; fi
  mark EMASTART "$TAGE"; ts "ema start"
  gpu_acquire "$JOB" || exit 3
  STOPW="$KIT/chain/cfn_gpu_watch_$TAGE.stop"; rm -f "$STOPW"
  "$PY" "$BAT/gpu_watch.py" --out "$KITW/chain/cfn_gpu_watch_$TAGE.jsonl" --stop-file "$KITW/chain/cfn_gpu_watch_$TAGE.stop" >> "$KIT/chain/cfn_lockwait_$JOB.log" 2>&1 &
  trap ': > "$STOPW"; gpu_release "$JOB"' EXIT
  mkdir -p "$KIT/battery/seam_tele_$TAGE"
  run_pass() {
    ( export R8_NO_POSE_SYNC=1 R8_SEAM_TELE_DIR="$KITW/battery/seam_tele_$TAGE"
      "$PY" "$PKGW/code/run_battery_r8.py" --ckpt "$EMA" --config "$CFG" --no-g0 --tag "$TE" --infer-seeds 0,1 --lock-held \
          --skip-roll --out-root "$BRW" >> "$KIT/chain/battery_run_$TE.log" 2>&1 )
  }
  pass_done() { jq_ok "$WORK/battery_summary.json" "j.get('bars') and j.get('finished') and 'r8' in j"; }
  ts "ema pass start"
  run_pass
  if ! pass_done; then mark EMARETRY "pass incomplete, retrying once (resumes from complete seed dumps)"; run_pass; fi
  ts "ema pass end"
  bank() { "$PY" "$PKGW/code/bank_battery_r8.py" "$BRW" "$RAWE_W" --tag "$TAGE" --passes unc --chain-log "$KITW/chain/chain_final.log" --timings "$KITW/chain/chain_final_timings.txt" --status "$1" >> "$LOG" 2>&1; }
  if ! pass_done; then mark EMAFAIL "no complete battery_summary (see chain/battery_run_$TE.log)"; bank "FAILED: EMA pass 1 incomplete"; exit 4; fi
  mark EMAPASSDONE "unc (battery)"
  ( export R8_NO_POSE_SYNC=1
    "$PY" "$PKGW/code/pair_read_r8.py" --root "$WORK" --clock standalone --config "$CFG" --out-json "$WORK/PAIRED_READ_$TE.json" --out-md "$WORK/PAIRED_READ_$TE.md" > "$KIT/chain/pair_read_$TE.log" 2>&1 )
  if [ -s "$BRL/$TE/PAIRED_READ_$TE.json" ]; then mark EMAPAIRED "unc"; else mark EMAPAIREDNOJSON "see chain/pair_read_$TE.log"; fi
  "$PY" -I -c "
import json,sys
reason=('EMA weights of the same step; this chain runs the EMA pass FIRST and the G0 of the raw weights at this step AFTER it (the step-30,000 policy); the EMA weights get NO G0 of their own; '
        'the G0 of the raw weights is the nearest instrument and is stated beside, not substituted')
json.dump({'tag':sys.argv[2],'status':'PROVISIONAL','no_g0_reason':reason,'ckpt':sys.argv[3],'raw_ckpt':sys.argv[4],'g0_json_of_raw_weights':sys.argv[5],
           'note':'every number from this pass is PROVISIONAL; no G0 exists for the EMA weights'}, open(sys.argv[1],'w',encoding='utf-8'),indent=1)" \
     "$RAWE_W/NO_G0_REASON_$TAGE.json" "$TAGE" "$EMA" "$CK" "$KITW/battery/$TAGR/g0.json"
  cp "$RAWE_L/NO_G0_REASON_$TAGE.json" "$BRL/$TE/NO_G0_REASON_$TAGE.json" 2>/dev/null
  bank "PROVISIONAL: EMA weights, no G0 for them (reason in NO_G0_REASON_$TAGE.json); pass unc complete"
  if jq_ok "$RAWE_L/BATTERY_SUMMARY_$TAGE.json" "j['passes']['unc']['status']=='COMPLETE'"; then mark EMADONE "banked: $RAWE_W/BATTERY_SUMMARY_$TAGE.json"; else mark EMABANKFAIL ""; exit 4; fi
)
# ---- k: the K-draw rolls (registered read #3) --------------------------------------------------------------------- #
KDR="$KIT/kdraw/$TE"
kd_all_done() {
  _s=2; while [ "$_s" -le 7 ]; do jq_ok "$KDR/dump_s$_s/KDRAW_VERIFY.json" "j.get('ok') is True" || return 1; _s=$((_s+1)); done
  jq_ok "$KDR/KDRAW_CONTROL.json" "'verdict' in j and len(j.get('cases',{}))>=3"
}
kd_stage() (
  if kd_all_done; then mark KDHAVE "seeds 2..7 + control already rolled for $TE"; exit 0; fi
  # the preconditions kdraw_roll_r8 itself enforces, asserted HERE so a skip names its reason: a complete banked battery tree for the EMA weights, with the same ckpt md5, the windows file and dump_s0
  M_EMA=$(md5sum "$CKL/ckpt_ema_$STEP.pt" | cut -d' ' -f1)
  if ! jq_ok "$BRL/$TE/battery_summary.json" "j.get('finished') and j.get('ckpt_md5')=='$M_EMA'" || [ ! -s "$BRL/$TE/windows_s0.json" ] || [ ! -d "$BRL/$TE/dump_s0" ]; then
    mark KDSKIP "no complete banked EMA battery tree (battery_summary.json with ckpt_md5 == $M_EMA, windows_s0.json, dump_s0) under $BRW/$TE: the K-draw rolls need it"; exit 5; fi
  _n=0; while [ -s "$KIT/chain/kdraw_queue.pid" ] && kill -0 "$(cat "$KIT/chain/kdraw_queue.pid")" 2>/dev/null; do   # kdraw_queue_r8.sh is single-instance
    _n=$((_n+1)); [ $((_n % 15)) -eq 1 ] && mark KDBUSY "another kdraw queue (pid $(cat "$KIT/chain/kdraw_queue.pid")) is alive; waiting"
    [ $((_n * POLL)) -ge "${CFN_KD_BUSY_MAXWAIT_S:-21600}" ] && { mark KDFAIL "another kdraw queue stayed alive for ${CFN_KD_BUSY_MAXWAIT_S:-21600} s"; exit 4; }
    sleep "$POLL"
  done
  mark KDSTART "KD_STEP=$STEP tree $TE (ctrl + seeds 2..7); queue log $KIT/chain/kdraw_queue_final.log"; ts "kdraw start"
  # KD_NO_RELEASE: /c/lgs/RELEASE_R7 is the 30k queue's signal to the deferred refcv7 chains; the final tree must not re-trigger it.  KD_SEG_MAX 8: ONE lock hold for all 7 jobs (the final read is not interleaved).
  KD_STEP="$STEP" KD_NO_RELEASE=1 KD_LOG="$KIT/chain/kdraw_queue_final.log" KD_SEG_MAX="${CFN_KD_SEG_MAX:-8}" KD_LOCK_MAXWAIT_S=172800 KD_MEM_MAXWAIT_S="${CFN_KD_MEM_MAXWAIT_S:-21600}" \
      sh "$KDQ" < /dev/null > "$KIT/chain/cfn_kdraw.out" 2>&1
  ts "kdraw end"
  if kd_all_done; then mark KDDONE "seeds 2..7 verified + control written under $KDR"; else mark KDFAIL "not every roll verified (see $KIT/chain/kdraw_queue_final.log and $KIT/chain/kdraw_queue_py.log); the read will be PENDING"; exit 4; fi
)
# ---- f: the raw milestone (metrics copy -> G0 -> wrapper probe -> battery) ------------------------------------------ #
metrics_have() { "$PY" -I "$PKGW/code/pick_metrics.py" "$KITW/thor_reads" $STEP; }
pull_metrics() {   # read-only on Thor (thor_pull_r8.sh finds ckpt_$STEP.md5 -> mode 'have' -> only the metrics copy; the final eval row exists because the trainer evals at step == args.steps)
  [ -n "$(metrics_have)" ] && { mark METRICSHAVE "metrics copy with one eval row at $STEP exists"; return 0; }
  sh "$PKGL/code/thor_pull_r8.sh" "$STEP" > "$KIT/chain/cfn_metrics_pull.out" 2>&1
  PR=$(tail -n 1 "$KIT/chain/cfn_metrics_pull.out")
  case "$PR" in pull:ok:*) mark METRICSOK "$PR" ;; *) mark METRICSFAIL "$PR (see chain/thor_pull_step$STEP.log)"; return 5 ;; esac
}
wrapfix_one() (
  JOB="refcv8-wrapfix-$TAGR"; OUTJ="$RAWR_W/wrapper_clause_r8fix_gpu.json"
  CKM=$(md5sum "$CK" | cut -d' ' -f1); [ "${#CKM}" -eq 32 ] || { mark WRAPFIXFAIL "cannot md5 $CK"; exit 4; }
  if jq_ok "$OUTJ" "j.get('ok') is True and j.get('CLAUSE') in ('PASS','FAIL','VOID') and j.get('ckpt_md5')=='$CKM'"; then mark WRAPFIXHAVE "$OUTJ"; exit 0; fi
  mark WRAPFIXSTART "$TAGR"
  gpu_acquire "$JOB" || exit 3
  trap 'gpu_release "$JOB"' EXIT
  "$PY" "$PKGW/code/wrapper_clause_r8fix_gpu.py" --ckpt "$CK" --config "$CFG" --g0-json "$KITW/battery/$TAGR/g0.json" --out "$OUTJ" > "$KIT/chain/cfn_wrapfix.log" 2>&1
  if jq_ok "$OUTJ" "j.get('ok') is True and j.get('ckpt_md5')=='$CKM'"; then mark WRAPFIXOK "CLAUSE=$(jget "$OUTJ" "j['CLAUSE']")"; else mark WRAPFIXFAIL "see chain/cfn_wrapfix.log"; exit 4; fi
)
raw_stage() (
  if jq_ok "$RAWR_L/BATTERY_SUMMARY_$TAGR.json" "j['passes']['unc']['status']=='COMPLETE'"; then mark RAWHAVE "$RAWR_W"; exit 0; fi
  mark RAWSTART "$TAGR"; ts "raw start"
  pull_metrics || exit 5
  if ! jq_ok "$KIT/battery/$TAGR/g0.json" "j.get('ckpt_md5') and j.get('verdict')"; then sh "$PKGL/code/chain_g0_r8.sh" "$STEP" > "$KIT/chain/cfn_g0.out" 2>&1; fi
  ts "raw g0 end"
  if jq_ok "$KIT/battery/$TAGR/g0.json" "j.get('ckpt_md5') and j.get('verdict')"; then
    mark RAWG0OK "$(jget "$KIT/battery/$TAGR/g0.json" "'G0 '+str(j['verdict'].get('G0'))+' M1 '+str(((j['verdict'].get('mutation_detection') or {}).get('m1') or {}).get('detected'))")"
  else mark RAWG0NOJSON "no g0.json for step $STEP (see chain/chain_$TAGR.log); the battery chain will log its skip"; fi
  if [ "${CFN_WRAPFIX:-1}" = "1" ] && jq_ok "$KIT/battery/$TAGR/g0.json" "j.get('ckpt_md5')"; then wrapfix_one || mark RAWWRAPFIXFAILED "non-blocking"; fi
  mkdir -p "$KIT/battery/seam_tele_$TAGR"
  ( export R8_SEAM_TELE_DIR="$KITW/battery/seam_tele_$TAGR" R8_CORRECTED=0
    sh "$PKGL/code/chain_battery_r8.sh" "$STEP" > "$KIT/chain/cfn_battery.out" 2>&1 )
  ts "raw battery end"
  if jq_ok "$RAWR_L/BATTERY_SUMMARY_$TAGR.json" "j['passes']['unc']['status']=='COMPLETE'"; then mark RAWDONE "banked: $RAWR_W/BATTERY_SUMMARY_$TAGR.json"
  else mark RAWINCOMPLETE "no COMPLETE unc pass in $RAWR_W/BATTERY_SUMMARY_$TAGR.json (skipped / failed: read it and chain/battery_$TAGR.log)"; exit 4; fi
)
# ---- the CPU lane: d (compliant bars), e (frozen-gain forwards), g (heading), train-fit forward ------------------- #
cpu_run() {   # $1 name $2 cwd $3 logfile, rest = command; CPU only, memory-gated, one at a time
  _n=$1; _d=$2; _l=$3; shift 3
  mem_gate "$CPUNEED" "$_n" || return 3
  ( cd "$_d" && env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKGW/code" "$@" ) >> "$_l" 2>&1
}
paired_ready() {   # the battery's PAIRED_READ for tree $1 is complete (finished + bars) -- the tree_complete rule of compliant_reread_r8
  jq_ok "$BRL/$1/PAIRED_READ_$1.json" "j.get('finished') and (j.get('bars') or {}).get('BAR-R8-E1') and all('rows' in j['surfaces'][s]['seeds'][k] for s in ('S2','S6') for k in ('0','1'))"
}
earlier_args() {   # $1 = label prefix, $2 = tree suffix ('ema' or ''): the earlier refcv8 points that EXIST (a banked battery_summary.json), most recent first
  _o=""
  for _p in 40000:40k 30000:30k 20000:20k 15000:15k 10500:10.5k 5000:5k; do
    _s=${_p%%:*}; _l=${_p#*:}
    [ -s "$BRL/step${_s}$2_unc/battery_summary.json" ] && _o="$_o $1-$_l=$BRW/step${_s}$2_unc"
  done
  echo "$_o"
}
tf_gains_ready() { [ -s "$PKGL/raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json" ] && [ -s "$PKGL/raw/retime_trainfit/FROZEN_GAINS_RETIME_train.sha256" ]; }
lane_tree() (   # $1 = tree, $2 = state name (ema|raw), $3 = heading label prefix, $4 = suffix of the earlier trees ('ema' or ''), $5 = heading tag
  T=$1; STN=$2; LAB=$3; SUF=$4; HTAG=$5
  _n=0
  while [ ! -s "$ST/$STN.state" ]; do _n=$((_n+1)); [ $((_n % 60)) -eq 1 ] && mark LANEWAIT "$T waits for the $STN GPU stage"; [ $((_n * LPOLL)) -ge "$LANE_MAXWAIT" ] && { mark LANETIMEOUT "$T"; exit 4; }; sleep "$LPOLL"; done
  if [ "$(cat "$ST/$STN.state")" != ok ] || ! paired_ready "$T"; then mark LANESKIP "$T: GPU stage state '$(cat "$ST/$STN.state")' / PAIRED_READ not complete -- d, e, g for this tree NOT run"; exit 4; fi
  LG="$LOGD_L/lane_$T.log"
  # d -- BOTH are needed, in this order: compliant_reread's C1b compares the capped read with CLAMP_POSTHOC's read_ON; with no clamp_posthoc tree it WITHHOLDS every compliant number
  if [ -s "$PKGL/raw/clamp_posthoc/trees/$T.json" ]; then mark CLAMPHAVE "$T"; else
    cpu_run clamp_posthoc "$PKGW/code" "$LG" "$PY" -X utf8 clamp_posthoc_r8.py read --tree "$T"
    if [ -s "$PKGL/raw/clamp_posthoc/trees/$T.json" ]; then mark CLAMPOK "$T"; else mark CLAMPFAIL "$T (see $LG)"; fi
  fi
  CRJ="$PKGL/raw/compliant_reread/trees/$T.json"
  if jq_ok "$CRJ" "j.get('controls',{}).get('ALL_PASS') is True and j['bars'].get('status','').startswith('WRITTEN')"; then mark COMPLHAVE "$T"; else
    cpu_run compliant "$PKGW/code" "$LG" "$PY" -X utf8 compliant_reread_r8.py read --tree "$T" --force
    if jq_ok "$CRJ" "j.get('controls',{}).get('ALL_PASS') is True and j['bars'].get('status','').startswith('WRITTEN')"; then mark COMPLOK "$T capped-on-compliant bars WRITTEN, controls C1/C1v/C2/C3 ALL_PASS"
    else mark COMPLFAIL "$T controls did not pass or no tree json: bars WITHHELD (see $LG)"; fi
  fi
  # e -- the frozen-gain forwards (each script's frozen 30k file UNCHANGED; forward needs the compliant tree json above)
  for pr in "a0k0_headroom_r8.py:raw/a0k0_headroom/FORWARD_$T.json" "a0k0_headroom_lon_r8.py:raw/a0k0_lon/FORWARD_LON_$T.json" "a0k0_headroom_retime_r8.py:raw/a0k0_retime/FORWARD_RETIME_$T.json"; do
    S=${pr%%:*}; A=${pr#*:}
    if jq_ok "$PKGL/$A" "j.get('controls',{}).get('ALL_PASS') is True"; then mark FWDHAVE "$S $T"; continue; fi
    cpu_run "fwd_$S" "$PKGW/code" "$LG" "$PY" -X utf8 "$S" forward --tree "$T"
    if jq_ok "$PKGL/$A" "j.get('controls',{}).get('ALL_PASS') is True"; then mark FWDOK "$S $T -> $A"
    elif [ -s "$PKGL/$A" ]; then mark FWDCTRLFAIL "$S $T wrote $A but its controls are not ALL_PASS: read before quoting"; else mark FWDFAIL "$S $T (see $LG)"; fi
  done
  # g -- heading drift, as post30k_heading_r8.sh; the earlier list is built from the trees that exist
  OUTF="$PKGL/raw/heading_drift/$HTAG.json"
  if [ -s "$OUTF" ]; then mark HEADHAVE "$HTAG"; else
    EARL=$(earlier_args "$LAB" "$SUF")
    cpu_run heading "$PKGW" "$LG" "$PY" -X utf8 raw/heading_drift/heading_drift.py --new-root "$BRW/$T" --label-new "$LAB-final" \
        --earlier $EARL --out-dir raw/heading_drift --tag "$HTAG"
    if [ -s "$OUTF" ]; then mark HEADDONE "$HTAG -> raw/heading_drift/$HTAG.md (earlier:$EARL)"; else mark HEADFAIL "$HTAG wrote no json (see $LG)"; fi
  fi
  mark LANETREEDONE "$T"
)
tf_tree() (   # $1 = tree: the TRAIN-FIT retime forward (frozen train gains, PREREG_TRAINFIT); needs the same compliant machinery as the frozen forwards
  T=$1; LG="$LOGD_L/lane_$T.log"; TFJ="$PKGL/raw/retime_trainfit/FORWARD_TRAINFIT_$T.json"
  tf_ok() { jq_ok "$TFJ" "j.get('controls',{}).get('ALL_PASS') is True and (j.get('const') or {}).get('ALL_PASS') is True"; }
  if tf_ok; then mark TFHAVE "$T"; exit 0; fi
  if ! tf_gains_ready; then mark TFSKIP "$T: raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json (+ .sha256) does not exist (the train-fit chain has not finished): re-run this chain, or: python retime_trainfit_r8.py forward --tree $T"; exit 5; fi
  cpu_run tf_forward "$PKGW/code" "$LG" "$PY" -X utf8 retime_trainfit_r8.py forward --tree "$T"
  if tf_ok; then mark TFOK "$T -> $TFJ (its 'role' text is hard-coded to the 30k wording: read the tree name, not that string)"
  elif [ -s "$PKGL/raw/retime_trainfit/WITHHELD_$T.json" ]; then mark TFWITHHELD "$T: a known-value control failed, no forward file written (see WITHHELD_$T.json)"
  else mark TFFAIL "$T (see $LG)"; fi
)
lane() {
  lane_tree "$TE" ema EMA ema "HEADING_DRIFT_${TAGE}"
  lane_tree "$TR" raw RAW "" "HEADING_DRIFT_${TAGR}raw"
  # the train-fit gains come from the train-fit chain (a GPU queue member); wait for them once, bounded, then try both trees
  _w=0; while ! tf_gains_ready && [ "$_w" -lt "${CFN_TF_WAIT_S:-14400}" ]; do [ $((_w % 1800)) -eq 0 ] && mark TFWAIT "waiting ${_w}s for FROZEN_GAINS_RETIME_train.json"; sleep "${CFN_TF_POLL_S:-120}"; _w=$((_w + ${CFN_TF_POLL_S:-120})); done
  for t in "$TE" "$TR"; do
    if jq_ok "$BRL/$t/battery_summary.json" "j.get('finished')"; then tf_tree "$t"; else mark TFSKIP "$t: no finished battery tree"; fi
  done
  for t in a0k0_headroom_r8.py:a0k0_headroom:A0K0_HEADROOM.md a0k0_headroom_lon_r8.py:a0k0_lon:A0K0_LON.md a0k0_headroom_retime_r8.py:a0k0_retime:A0K0_RETIME.md; do   # render: re-render each .md from its JSONs (backup once)
    s=${t%%:*}; r=${t#*:}; d=${r%%:*}; m=${r#*:}
    [ -e "$PKGL/raw/$d/$m.preFinal" ] || cp "$PKGL/raw/$d/$m" "$PKGL/raw/$d/$m.preFinal" 2>/dev/null
    cpu_run render "$PKGW/code" "$LOGD_L/lane_render.log" "$PY" -X utf8 "$s" render || mark RENDERFAIL "$s (non-blocking)"
  done
  if [ -s "$PKGL/code/summarize_40k_r8.py" ]; then "$PY" -I "$PKGW/code/summarize_40k_r8.py" --pkg "$PKGW" --out-dir "$RAWR_W" --step "$STEP" > "$LOGD_L/summarize.log" 2>&1
    if [ -s "$RAWR_L/READ_40K.json" ]; then cp -f "$RAWR_L/READ_40K.json" "$RAWR_L/READ_FINAL.json"; cp -f "$RAWR_L/READ_40K.md" "$RAWR_L/READ_FINAL.md" 2>/dev/null; mark SUMMARYOK "$RAWR_W/READ_FINAL.md (summarize_40k_r8.py names its output READ_40K.*; copied)"; else mark SUMMARYFAIL "see $LOGD_L/summarize.log"; fi; fi
  mark LANEEND ""
}
# ---- tail: the K-draw read and the combo ---------------------------------------------------------------------------------- #
kd_analysis() {   # CPU, 3-4 s; works on partial seeds (PENDING), so it runs whatever stage k delivered -- but not without a banked EMA tree (nothing to read)
  [ -s "$BRL/$TE/battery_summary.json" ] || { mark KDANALYSISSKIP "no banked EMA battery tree"; return 0; }
  KAJ="$PKGL/raw/kdraw/$TE/KDRAW_E1.json"
  # skip when every roll is done and the read is newer than every roll's verify record (a resumed chain must not redo it; a read made on PARTIAL seeds is stale once a later roll lands)
  if kd_all_done && [ -s "$KAJ" ] && [ -z "$(find "$KDR" -name KDRAW_VERIFY.json -newer "$KAJ" 2>/dev/null | head -n 1)" ]; then mark KDANALYSISHAVE "$TE"; return 0; fi
  cpu_run kdraw_analysis "$PKGW/code" "$LOGD_L/tail.log" sh "$PKGL/code/run_kdraw_analysis_r8.sh" "$TE"
  if [ -s "$KAJ" ]; then mark KDANALYSIS "$(jget "$KAJ" "'overall='+str(j.get('overall'))+' void='+str(j.get('void_controls_failed'))")"; else mark KDANALYSISFAIL "no $KAJ (see $LOGD_L/tail.log)"; fi
}
combo_stage() {
  CJ="$PKGL/raw/combo_ens_retime/${TE}_train/COMBO_${TE}_train.json"
  if jq_ok "$CJ" "j.get('status') is not None and not j.get('void_controls_failed')"; then mark COMBOHAVE "$TE"; return 0; fi
  _miss=""
  kd_all_done || _miss="$_miss kdraw-rolls"
  jq_ok "$BRL/$TE/battery_summary.json" "j.get('finished')" || _miss="$_miss ema-battery-tree"
  tf_gains_ready || _miss="$_miss train-fit-gains"
  if [ -n "$_miss" ]; then mark COMBOSKIP "$TE: missing preconditions:$_miss (the script would refuse too; re-run the chain or: sh code/run_combo_ens_retime_r8.sh $TE train)"; return 5; fi
  cpu_run combo "$PKGW/code" "$LOGD_L/tail.log" sh "$PKGL/code/run_combo_ens_retime_r8.sh" "$TE" train
  if [ -s "$CJ" ]; then mark COMBODONE "$(jget "$CJ" "'status='+str(j.get('status'))+' void='+str(j.get('void_controls_failed'))")"; else mark COMBOFAIL "no $CJ (see $LOGD_L/tail.log)"; return 4; fi
}
# ---- drive ------------------------------------------------------------------------------------------------------------------ #
pull_metrics || mark METRICSDEFER "metrics copy not available yet; the raw stage retries it"
( lane ) &
LANEPID=$!
echo "$LANEPID" > "$KIT/chain/chain_final_lane.pid"
mark LANESPAWNED "lane pid $LANEPID"
ema_stage; RC=$?
if [ $RC -eq 0 ]; then echo ok > "$ST/ema.state"; else echo fail > "$ST/ema.state"; mark STAGEFAILED "ema rc $RC (continuing: the K-draw skips, the raw stage is independent)"; fi
kd_stage; RC=$?
if [ $RC -eq 0 ]; then echo ok > "$ST/kd.state"; else echo fail > "$ST/kd.state"; mark STAGEFAILED "kd rc $RC (continuing: the raw stage is independent)"; fi
kd_analysis
raw_stage; RC=$?
if [ $RC -eq 0 ]; then echo ok > "$ST/raw.state"; else echo fail > "$ST/raw.state"; mark STAGEFAILED "raw rc $RC"; fi
wait "$LANEPID"
combo_stage; echo "rc $?" > "$ST/combo.state"
if [ -s "$PKGL/code/final_index_r8.py" ]; then
  "$PY" -I "$PKGW/code/final_index_r8.py" --pkg "$PKGW" --kit "$KITW" --arch "$BRW" --step "$STEP" --chain-log "$KITW/chain/chain_final.log" --out-dir "$RAWR_W" > "$LOGD_L/index.log" 2>&1
  if [ -s "$RAWR_L/FINAL_INDEX.json" ]; then mark INDEXOK "$RAWR_W/FINAL_INDEX.md"; else mark INDEXFAIL "see $LOGD_L/index.log"; fi
fi
mark END "ema=$(cat "$ST/ema.state") kd=$(cat "$ST/kd.state") raw=$(cat "$ST/raw.state") combo=$(cat "$ST/combo.state")"
note "chain end"
# the index was written BEFORE the END marker (it could not show it); write it once more, silently, so the page a human opens carries the terminal line
[ -s "$RAWR_L/FINAL_INDEX.json" ] && "$PY" -I "$PKGW/code/final_index_r8.py" --pkg "$PKGW" --kit "$KITW" --arch "$BRW" --step "$STEP" --chain-log "$KITW/chain/chain_final.log" --out-dir "$RAWR_W" >> "$LOGD_L/index.log" 2>&1

#!/bin/sh
# chain_40k_r8.sh -- the STEP-40,000 chain: snapshot -> weights -> EMA battery -> compliant bars -> frozen-gain forward tests -> raw battery -> heading.  usage: sh chain_40k_r8.sh
#   NEW file (2026-10-19).  Mirrors queue_30000_r8.sh (EMA battery first, raw behind it), compliant_reread (PI-ruled bars), the three pre-registered frozen-gain forwards and post30k_heading_r8.sh.
#   Markers (opaque ZZC40...ZZ) -> E:/refcv8_eval_kit/chain/chain_40k.log.  Terminal marker of the whole chain: ZZC40ENDZZ (or ZZC40ABORTZZ with the stage).  Resumable: re-run the same command;
#   a stage whose artifact already says done is skipped.  Assert on ARTIFACTS, never an exit code, never through a pipe.  Thor is touched ONLY by thor_pull_r8.sh (read-only scp of metrics.jsonl).
#
#   a  SNAP     wait for the snap40k_r8.sh terminal marker (chain/snap40k.log, last ZZS40START segment).  Proceed ONLY on ZZS40DONEZZ (or ZZS40ALREADYZZ with a provenance.json) AND the local
#               md5 of ckpt_full_40000.pt == the .md5 sidecar == provenance.json md5 == the DONE marker's md5, all four 32 hex chars.  Any other terminal marker -> ZZC40SNAPFAILZZ, stop.
#   b  EXTRACT  extract_ema_r8.py extract --expect-step 40000 (the step INSIDE the file; snap40k does not check it) -> ckpt_40000.pt + ckpt_ema_40000.pt; the chain re-asserts step==40000 and
#               ok==true on the provenance artifact; writes ckpt_40000.md5; then the EMA gate (ema_check_r8.py: strict load clean, weights differ from raw, md5-pinned).  A 39,500 file stops HERE.
#   c  EMA      battery pass 1 on the EMA weights, tag step40000ema_unc, inference seeds 0,1, standalone clock, shared GPU lock, one retry, pair_read vs refcv7-50,400, bank.   [as queue_30000 ema_stage]
#   f  RAW      thor_pull (metrics copy) -> chain_g0_r8.sh 40000 -> wrapper-clause probe (non-blocking) -> chain_battery_r8.sh 40000 (pass 1 only, R8_CORRECTED=0), tag step40000_unc.
#   CPU LANE    a background subshell (CPU only, memory-gated on FreeVirtualMemory) that follows c and f without waiting for the other lane:
#               d  clamp_posthoc read + compliant_reread read (A-R8-CLAMP-2: capped plans on ceiling-compliant windows -- the read that decides)   e  the three frozen-gain forwards
#               (a0k0_headroom / _lon / _retime, each `forward --tree <tree>` with its frozen file UNCHANGED)   g  heading_drift (the H-R8-TURN-UNDERCORRECT-1 read, as post30k_heading_r8.sh)
#               Order inside the lane: EMA tree first (d, e, g), then the RAW tree.  Deviation from the brief's letter order: g(EMA) runs BEFORE the raw battery finishes (CPU, minutes, same inputs).
#
#   G0 POLICY (mirrors step 30,000, STATED): the EMA weights get NO G0 (`--no-g0`, NO_G0_REASON stamped, every EMA number PROVISIONAL).  The RAW weights get chain_g0_r8.sh; the raw battery runs
#   even when that G0 reads FAIL-as-coded provided M1 was detected (chain_battery_r8.sh --g0-nonblocking), stamped PROVISIONAL; R8_REQUIRE_G0_PASS=1 would turn that into a skip (off).
#   GPU LOCK: the SAME O_EXCL lock and the SAME 60 s acquire as every other chain (battery/code/gpu_lock.py).  NO priority mechanism exists in it (see HANDOFF): this chain can queue behind
#   kdraw / refcv7 / K2K4 and cannot preempt.  It deliberately does not poll faster than they do.  Paths are overridable by R8_*/C40_* env ONLY so the control flow can be dry-run (test_chain_40k_r8.py).
STEP="${C40_STEP:-40000}"
PKGW="${R8_PKGW:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
PKGL="${R8_PKGL:-/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
BAT="${R8_BAT:-C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code}"
KITW="${R8_KITW:-E:/refcv8_eval_kit}"
KIT="${R8_KIT:-/e/refcv8_eval_kit}"
BRW="${R8_BATT_ROOT_W:-C:/Users/Admin/refcv8_eval_archive/battery}"; BRL="${R8_BATT_ROOT_L:-/c/Users/Admin/refcv8_eval_archive/battery}"
export R8_PKGW="$PKGW" R8_PKGL="$PKGL" R8_KITW="$KITW" R8_KIT="$KIT" R8_BATT_ROOT_W="$BRW" R8_BATT_ROOT_L="$BRL" R8_BAT="$BAT"
PY="${R8_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
EXTRACT_PY="${C40_EXTRACT_PY:-$PKGW/code/extract_ema_r8.py}"
SNAPLOG="${C40_SNAP_LOG:-$KIT/chain/snap40k.log}"
FULL="${C40_SNAP_DEST:-$KITW/ckpt/ckpt_full_$STEP.pt}"; FULLL="${C40_SNAP_DEST_L:-$KIT/ckpt/ckpt_full_$STEP.pt}"
POLL="${C40_POLL_S:-120}"; LPOLL="${C40_LANE_POLL_S:-60}"
SNAP_MAXWAIT="${C40_SNAP_MAXWAIT_S:-259200}"; LANE_MAXWAIT="${C40_LANE_MAXWAIT_S:-259200}"
CPUNEED="${C40_CPU_NEED_GB:-5}"; EXNEED="${C40_EXTRACT_NEED_GB:-5}"; MIN_COMMIT="${C40_MIN_COMMIT_GB:-4}"
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6
unset R8_NO_POSE_SYNC
CK="$KITW/ckpt/ckpt_$STEP.pt"; EMA="$KITW/ckpt/ckpt_ema_$STEP.pt"; CFG="$KITW/ckpt/config.json"; CKL="$KIT/ckpt"
TAGE="step${STEP}ema"; TAGR="step$STEP"; TE="${TAGE}_unc"; TR="${TAGR}_unc"
RAWE_W="$PKGW/raw/$TAGE"; RAWE_L="$PKGL/raw/$TAGE"; RAWR_W="$PKGW/raw/$TAGR"; RAWR_L="$PKGL/raw/$TAGR"
LOGD_L="$RAWR_L/logs"
LOG="$KIT/chain/chain_40k.log"; ST="$KIT/chain/chain_40k.state"; TIM="$KIT/chain/chain_40k_timings.txt"
mkdir -p "$KIT/chain" "$ST" "$RAWE_L" "$RAWR_L" "$LOGD_L"
mark() { echo "ZZC40$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
note() { echo "[$(date +%FT%T)] chain_40k_r8: $*" >> "$KIT/KIT.log"; }
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
  if [ -n "$C40_FREEGB_CMD" ]; then v=$(sh -c "$C40_FREEGB_CMD" 2>/dev/null)
  else v=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" 2>/dev/null | tr -d '\r ' | tr ',' '.'); fi
  case "$v" in ''|*[!0-9.]*) echo 0 ;; *) echo "$v" ;; esac
}
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit (a+0>=b+0)?0:1}'; }
mem_gate() {   # $1 = GB needed, $2 = job: two checks 20 s apart (the dev box COMMIT limit binds, not RAM); returns 1 after C40_MEM_MAXWAIT_S
  _t0=$(date +%s); _n=0
  while :; do
    _h1=$(free_gb)
    if ge "$_h1" "$1"; then sleep "${C40_GATE_GAP_S:-20}"; _h2=$(free_gb); ge "$_h2" "$1" && return 0; fi
    _n=$((_n+1)); [ $((_n % 20)) -eq 1 ] && mark MEMWAIT "$2: free commit ${_h1} GB < $1"
    [ $(( $(date +%s) - _t0 )) -gt "${C40_MEM_MAXWAIT_S:-43200}" ] && { mark MEMTIMEOUT "$2"; return 1; }
    sleep "${C40_MEM_POLL_S:-30}"
  done
}
# ---- single instance ------------------------------------------------------------------------------------- #
PIDF="$KIT/chain/chain_40k.pid"
if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then mark DUP "another chain_40k (pid $(cat "$PIDF")) is alive; exiting"; exit 5; fi
echo $$ > "$PIDF"
trap '[ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"' EXIT
trap 'exit 143' TERM INT HUP
rm -f "$ST"/*.state
abort() { mark ABORT "stage $1: $2"; note "ABORT stage $1: $2"; exit "$3"; }
mark START "pid $$ step $STEP tags $TE $TR"; note "armed (pid $$)"; ts "armed"
# ---- a: wait for the snapshot -------------------------------------------------------------------------- #
snap_state() {   # prints none | wait | done | already | fail:<MARKER>   (the LAST ZZS40START segment of the snap40k log)
  awk '/ZZS40STARTZZ/ {st="wait"; next}
       /ZZS40DONEZZ/ {st="done"; next}
       /ZZS40ALREADYZZ/ {if (st != "done") st="already"; next}
       /ZZS40(ABORTLATER|SSHDEAD|TIMEOUT|NODISK|BADSIZE|INCONCLUSIVE|CHANGED|SIZEMISMATCH|MD5MISMATCH|TOOLATE)ZZ/ {match($0, /ZZS40[A-Z]*ZZ/); st="fail:" substr($0, RSTART, RLENGTH); next}
       END {print (st == "" ? "none" : st)}' "$SNAPLOG" 2>/dev/null
}
snap_stage() {
  _t0=$(date +%s); _n=0; _dead=0
  while :; do
    S=$(snap_state)
    case "$S" in
      done|already) break ;;
      fail:*) mark SNAPFAIL "snap40k reached ${S#fail:}: ckpt_full_$STEP.pt is NOT trustworthy (or the window is lost); chain stops"; return 20 ;;
    esac
    SPID=$(grep -a 'ZZS40STARTZZ' "$SNAPLOG" 2>/dev/null | tail -n 1 | sed -n 's/.* pid \([0-9][0-9]*\) .*/\1/p')
    if [ -n "$SPID" ] && ! kill -0 "$SPID" 2>/dev/null; then _dead=$((_dead+1)); else _dead=0; fi
    if [ "$_dead" -ge "${C40_DEAD_POLLS:-3}" ]; then mark SNAPDEAD "snap40k pid $SPID is gone with no terminal marker (state $S): the 40,000 window may be lost; a human must look"; return 22; fi
    _n=$((_n+1)); [ $((_n % 15)) -eq 1 ] && mark SNAPWAIT "snap40k state $S (pid ${SPID:-?})"
    [ $(( $(date +%s) - _t0 )) -ge "$SNAP_MAXWAIT" ] && { mark SNAPTIMEOUT "waited $SNAP_MAXWAIT s"; return 23; }
    sleep "$POLL"
  done
  mark SNAPSEEN "snap40k state $S"
  # the local md5 re-verified against THREE independent records, shape asserted before comparing (an empty string must never read as a match)
  [ -s "$FULLL" ] || { mark MD5BAD "no $FULLL"; return 21; }
  LMD=$(md5sum "$FULLL" | cut -d' ' -f1)
  SIDE=$(cut -d' ' -f1 < "$FULLL.md5" 2>/dev/null | head -n 1)
  PROV=$(jget "$FULLL.provenance.json" "j['md5']")
  DONEMD=$(grep -a 'ZZS40DONEZZ' "$SNAPLOG" 2>/dev/null | tail -n 1 | sed -n 's/.* md5 \([0-9a-f]\{32\}\) size .*/\1/p')
  if [ "$S" = already ] && [ -z "$DONEMD" ]; then DONEMD="$PROV"; fi    # ALREADY has no DONE line: provenance stands in (still needs sidecar + provenance + local)
  for m in "$LMD" "$SIDE" "$PROV" "$DONEMD"; do
    [ "${#m}" -eq 32 ] || { mark MD5INCONCLUSIVE "an md5 is not 32 chars (local '$LMD' sidecar '$SIDE' provenance '$PROV' done '$DONEMD'): never a match"; return 21; }
  done
  if [ "$LMD" = "$SIDE" ] && [ "$LMD" = "$PROV" ] && [ "$LMD" = "$DONEMD" ]; then SNAPMD5="$LMD"; mark MD5OK "ckpt_full_$STEP.pt md5 $SNAPMD5 (local == sidecar == provenance == DONE marker)"
  else mark MD5BAD "local $LMD sidecar $SIDE provenance $PROV done $DONEMD"; return 21; fi
  return 0
}
snap_stage; RC=$?; [ $RC -eq 0 ] || abort a "snapshot not usable" $RC
ts "snapshot verified"
# ---- b: extract + EMA gate ------------------------------------------------------------------------------- #
EXREC="$KITW/ckpt/ckpt_$STEP.provenance.extract_ema.json"
extract_ok() {   # the pair + a provenance that says ok, step == $STEP (the step INSIDE the file) and the md5 of THIS snapshot
  [ -s "$CKL/ckpt_$STEP.pt" ] && [ -s "$CKL/ckpt_ema_$STEP.pt" ] && \
    jq_ok "$EXREC" "j.get('ok') is True and int(j.get('step',-1))==$STEP and j.get('src_md5')=='$SNAPMD5'"
}
extract_done() { extract_ok && [ -s "$CKL/ckpt_$STEP.md5" ]; }   # + the md5 record written after the extract
extract_stage() {
  if extract_done; then mark EXTRACTHAVE "pair already derived from this snapshot"; return 0; fi
  FREEKB=$(df -k "$CKL" | tail -n 1 | awk '{print $4}')
  [ -n "$FREEKB" ] && [ "$FREEKB" -lt "${C40_MIN_FREE_KB:-3500000}" ] && { mark EXTRACTNODISK "free ${FREEKB} kB on $CKL"; return 24; }
  mem_gate "$EXNEED" extract || return 25
  mark EXTRACTSTART "expect-step $STEP, src $FULL"
  FORCE=""; { [ -e "$CKL/ckpt_$STEP.pt" ] || [ -e "$CKL/ckpt_ema_$STEP.pt" ]; } && FORCE="--force"     # a partial/old attempt: DERIVED files, the snapshot is the source
  "$PY" "$EXTRACT_PY" extract --src "$FULL" --config "$CFG" --out-dir "$KITW/ckpt" --expect-step "$STEP" --min-commit-gb "$MIN_COMMIT" $FORCE \
      --provenance "thor_ckpt_pt_md5=$SNAPMD5" "snapshot=$FULL" "snap40k_log=$SNAPLOG" "chain=chain_40k_r8.sh" > "$KIT/chain/c40_extract.out" 2>&1
  # the step is re-asserted on the ARTIFACT (not trusted from the exit code or from --expect-step alone)
  if extract_ok; then :; else
    mark EXTRACTFAIL "no ok provenance for step $STEP: $(tail -n 1 "$KIT/chain/c40_extract.out" | cut -c1-300)"; return 21; fi
  M_RAW=$(md5sum "$CKL/ckpt_$STEP.pt" | cut -d' ' -f1); M_EMA=$(md5sum "$CKL/ckpt_ema_$STEP.pt" | cut -d' ' -f1)
  { [ "${#M_RAW}" -eq 32 ] && [ "${#M_EMA}" -eq 32 ]; } || { mark EXTRACTFAIL "cannot md5 the derived pair"; return 21; }
  { echo "$M_RAW  ckpt_$STEP.pt"; echo "$M_EMA  ckpt_ema_$STEP.pt"; echo "$SNAPMD5  ckpt_full_$STEP.pt (thor rolling ckpt.pt, step $STEP)"; } > "$CKL/ckpt_$STEP.md5.tmp" && mv -f "$CKL/ckpt_$STEP.md5.tmp" "$CKL/ckpt_$STEP.md5"
  mark EXTRACTOK "ckpt_$STEP.pt $M_RAW ckpt_ema_$STEP.pt $M_EMA"
}
ema_gate() {   # strict load + differs from raw, md5-pinned; the EMA stage STOPS otherwise
  GATEJ="$RAWE_W/EMA_CHECK_step$STEP.json"
  M_RAW=$(md5sum "$CKL/ckpt_$STEP.pt" | cut -d' ' -f1); M_EMA=$(md5sum "$CKL/ckpt_ema_$STEP.pt" | cut -d' ' -f1)
  PIN="j['md5']['raw']=='$M_RAW' and j['md5']['ema']=='$M_EMA'"
  if ! jq_ok "$GATEJ" "j.get('ok') is True and $PIN"; then
    mem_gate "$EXNEED" ema_check || return 25
    "$PY" "$PKGW/code/ema_check_r8.py" --raw "$CK" --ema "$EMA" --config "$CFG" --out "$GATEJ" > "$KIT/chain/c40_ema_check.out" 2>&1
  fi
  if jq_ok "$GATEJ" "j.get('ok') is True and $PIN and j['tensors']['n_float_differing']>0 and j['strict_load']['missing']==0 and j['strict_load']['unexpected']==0"; then
    cp -f "$RAWE_L/EMA_CHECK_step$STEP.json" "$RAWR_L/EMA_CHECK_step$STEP.json" 2>/dev/null; mark EMAGATEOK "strict load clean, weights differ from raw"; return 0
  fi
  mark EMAGATEFAIL "ema_check not ok / identical weights / strict-load defect: $GATEJ"; return 21
}
extract_stage; RC=$?; [ $RC -eq 0 ] || abort b "extract" $RC
ema_gate; RC=$?; [ $RC -eq 0 ] || abort b "ema gate" $RC
ts "extract + ema gate done"
# ---- GPU lock helpers (the SAME lock, the SAME 60 s acquire as every other chain) ---------------------------- #
gpu_acquire() {   # $1 = job name ; 0 = lock held
  rm -f "$KITW/chain/c40_wc_$1.json" "$KITW/chain/c40_lock_$1.json" 2>/dev/null     # a stale ok=true must not pass for this acquire
  "$PY" "$BAT/wait_commit.py" --min-gb 6 --max-wait-s 43200 --out "$KITW/chain/c40_wc_$1.json" >> "$KIT/chain/c40_lockwait_$1.log" 2>&1
  jq_ok "$KIT/chain/c40_wc_$1.json" "j.get('met') is True" || { mark NOCOMMIT "$1"; return 3; }
  "$PY" "$BAT/gpu_lock.py" acquire --job "$1" --pid $$ --max-wait-s 172800 --out "$KITW/chain/c40_lock_$1.json" >> "$KIT/chain/c40_lockwait_$1.log" 2>&1
  jq_ok "$KIT/chain/c40_lock_$1.json" "j.get('ok') is True" || { mark NOLOCK "$1"; return 3; }
  "$PY" "$BAT/chain_marker.py" create --chain "$1" --pid $$ >> "$KIT/chain/c40_lockwait_$1.log" 2>&1
  mark LOCKED "$1"; note "GPU lock acquired: $1"; ts "lock $1"
}
gpu_release() {
  "$PY" "$BAT/gpu_lock.py" release --job "$1" --pid $$ >> "$KIT/chain/c40_lockwait_$1.log" 2>&1
  "$PY" "$BAT/chain_marker.py" remove --chain "$1" --pid $$ >> "$KIT/chain/c40_lockwait_$1.log" 2>&1
  mark RELEASED "$1"
}
# ---- c: battery pass 1 on the EMA weights ------------------------------------------------------------------ #
ema_stage() (
  WORK="$BRW/$TE"; JOB="refcv8-battery-$TAGE"
  if jq_ok "$RAWE_L/BATTERY_SUMMARY_$TAGE.json" "j['passes']['unc']['status']=='COMPLETE'"; then mark EMAHAVE "$RAWE_W"; exit 0; fi
  mark EMASTART "$TAGE"; ts "ema start"
  gpu_acquire "$JOB" || exit 3
  STOPW="$KIT/chain/c40_gpu_watch_$TAGE.stop"; rm -f "$STOPW"
  "$PY" "$BAT/gpu_watch.py" --out "$KITW/chain/c40_gpu_watch_$TAGE.jsonl" --stop-file "$KITW/chain/c40_gpu_watch_$TAGE.stop" >> "$KIT/chain/c40_lockwait_$JOB.log" 2>&1 &
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
  bank() { "$PY" "$PKGW/code/bank_battery_r8.py" "$BRW" "$RAWE_W" --tag "$TAGE" --passes unc --chain-log "$KITW/chain/chain_40k.log" --timings "$KITW/chain/chain_40k_timings.txt" --status "$1" >> "$LOG" 2>&1; }
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
# ---- f: the raw milestone (metrics copy -> G0 -> wrapper probe -> battery) ------------------------------------------ #
metrics_have() { "$PY" -I "$PKGW/code/pick_metrics.py" "$KITW/thor_reads" $STEP; }
pull_metrics() {   # read-only on Thor (thor_pull_r8.sh finds ckpt_$STEP.md5 -> mode 'have' -> only the metrics copy)
  [ -n "$(metrics_have)" ] && { mark PULLHAVE "metrics copy with one eval row at $STEP exists"; return 0; }
  sh "$PKGL/code/thor_pull_r8.sh" "$STEP" > "$KIT/chain/c40_pull.out" 2>&1
  PR=$(tail -n 1 "$KIT/chain/c40_pull.out")
  case "$PR" in pull:ok:*) mark PULLOK "$PR" ;; *) mark PULLFAIL "$PR (see chain/thor_pull_step$STEP.log)"; return 5 ;; esac
}
wrapfix_one() (
  JOB="refcv8-wrapfix-$TAGR"; OUTJ="$RAWR_W/wrapper_clause_r8fix_gpu.json"
  CKM=$(md5sum "$CK" | cut -d' ' -f1); [ "${#CKM}" -eq 32 ] || { mark WRAPFIXFAIL "cannot md5 $CK"; exit 4; }
  if jq_ok "$OUTJ" "j.get('ok') is True and j.get('CLAUSE') in ('PASS','FAIL','VOID') and j.get('ckpt_md5')=='$CKM'"; then mark WRAPFIXHAVE "$OUTJ"; exit 0; fi
  mark WRAPFIXSTART "$TAGR"
  gpu_acquire "$JOB" || exit 3
  trap 'gpu_release "$JOB"' EXIT
  "$PY" "$PKGW/code/wrapper_clause_r8fix_gpu.py" --ckpt "$CK" --config "$CFG" --g0-json "$KITW/battery/$TAGR/g0.json" --out "$OUTJ" > "$KIT/chain/c40_wrapfix.log" 2>&1
  if jq_ok "$OUTJ" "j.get('ok') is True and j.get('ckpt_md5')=='$CKM'"; then mark WRAPFIXOK "CLAUSE=$(jget "$OUTJ" "j['CLAUSE']")"; else mark WRAPFIXFAIL "see chain/c40_wrapfix.log"; exit 4; fi
)
raw_stage() (
  if jq_ok "$RAWR_L/BATTERY_SUMMARY_$TAGR.json" "j['passes']['unc']['status']=='COMPLETE'"; then mark RAWHAVE "$RAWR_W"; exit 0; fi
  mark RAWSTART "$TAGR"; ts "raw start"
  pull_metrics || exit 5
  if ! jq_ok "$KIT/battery/$TAGR/g0.json" "j.get('ckpt_md5') and j.get('verdict')"; then sh "$PKGL/code/chain_g0_r8.sh" "$STEP" > "$KIT/chain/c40_g0.out" 2>&1; fi
  ts "raw g0 end"
  if jq_ok "$KIT/battery/$TAGR/g0.json" "j.get('ckpt_md5') and j.get('verdict')"; then
    mark RAWG0OK "$(jget "$KIT/battery/$TAGR/g0.json" "'G0 '+str(j['verdict'].get('G0'))+' M1 '+str(((j['verdict'].get('mutation_detection') or {}).get('m1') or {}).get('detected'))")"
  else mark RAWG0NOJSON "no g0.json for step $STEP (see chain/chain_$TAGR.log); the battery chain will log its skip"; fi
  if [ "${C40_WRAPFIX:-1}" = "1" ] && jq_ok "$KIT/battery/$TAGR/g0.json" "j.get('ckpt_md5')"; then wrapfix_one || mark RAWWRAPFIXFAILED "non-blocking"; fi
  mkdir -p "$KIT/battery/seam_tele_$TAGR"
  ( export R8_SEAM_TELE_DIR="$KITW/battery/seam_tele_$TAGR" R8_CORRECTED=0
    sh "$PKGL/code/chain_battery_r8.sh" "$STEP" > "$KIT/chain/c40_battery.out" 2>&1 )
  ts "raw battery end"
  if jq_ok "$RAWR_L/BATTERY_SUMMARY_$TAGR.json" "j['passes']['unc']['status']=='COMPLETE'"; then mark RAWDONE "banked: $RAWR_W/BATTERY_SUMMARY_$TAGR.json"
  else mark RAWINCOMPLETE "no COMPLETE unc pass in $RAWR_W/BATTERY_SUMMARY_$TAGR.json (skipped / failed: read it and chain/battery_$TAGR.log)"; exit 4; fi
)
# ---- the CPU lane: d (compliant bars), e (frozen-gain forwards), g (heading) --------------------------------------- #
cpu_run() {   # $1 name $2 cwd $3 logfile, rest = command; CPU only, memory-gated, one at a time
  _n=$1; _d=$2; _l=$3; shift 3
  mem_gate "$CPUNEED" "$_n" || return 3
  ( cd "$_d" && env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKGW/code" "$@" ) >> "$_l" 2>&1
}
paired_ready() {   # the battery's PAIRED_READ for tree $1 is complete (finished + bars) -- the tree_complete rule of compliant_reread_r8
  jq_ok "$BRL/$1/PAIRED_READ_$1.json" "j.get('finished') and (j.get('bars') or {}).get('BAR-R8-E1') and all('rows' in j['surfaces'][s]['seeds'][k] for s in ('S2','S6') for k in ('0','1'))"
}
lane_tree() (   # $1 = tree, $2 = state name (ema|raw), $3 = heading label prefix, $4 = suffix of the earlier trees ('ema' or ''), $5 = heading tag
  T=$1; STN=$2; LAB=$3; SUF=$4; HTAG=$5
  _n=0
  while [ ! -s "$ST/$STN.state" ]; do _n=$((_n+1)); [ $((_n % 60)) -eq 1 ] && mark LANEWAIT "$T waits for the $STN GPU stage"; [ $((_n * LPOLL)) -ge "$LANE_MAXWAIT" ] && { mark LANETIMEOUT "$T"; exit 4; }; sleep "$LPOLL"; done
  if [ "$(cat "$ST/$STN.state")" != ok ] || ! paired_ready "$T"; then mark LANESKIP "$T: GPU stage state '$(cat "$ST/$STN.state")' / PAIRED_READ not complete -- d, e, g for this tree NOT run"; exit 4; fi
  LG="$LOGD_L/lane_$T.log"; CP="$PKGW/raw/clamp_posthoc/trees/$T.json"; CRJ="$PKGL/raw/compliant_reread/trees/$T.json"
  # d -- BOTH are needed, in this order: compliant_reread's C1b compares the capped read with CLAMP_POSTHOC's read_ON; with no clamp_posthoc tree it WITHHOLDS every compliant number
  #      (measured at step30000_unc on 2026-10-18: C1=False until the clamp_posthoc tree existed)
  if [ -s "$PKGL/raw/clamp_posthoc/trees/$T.json" ]; then mark CLAMPHAVE "$T"; else
    cpu_run clamp_posthoc "$PKGW/code" "$LG" "$PY" -X utf8 clamp_posthoc_r8.py read --tree "$T"
    if [ -s "$PKGL/raw/clamp_posthoc/trees/$T.json" ]; then mark CLAMPOK "$T"; else mark CLAMPFAIL "$T (see $LG)"; fi
  fi
  if jq_ok "$CRJ" "j.get('controls',{}).get('ALL_PASS') is True and j['bars'].get('status','').startswith('WRITTEN')"; then mark COMPLHAVE "$T"; else
    cpu_run compliant "$PKGW/code" "$LG" "$PY" -X utf8 compliant_reread_r8.py read --tree "$T" --force
    if jq_ok "$CRJ" "j.get('controls',{}).get('ALL_PASS') is True and j['bars'].get('status','').startswith('WRITTEN')"; then mark COMPLOK "$T capped-on-compliant bars WRITTEN, controls C1/C1v/C2/C3 ALL_PASS"
    else mark COMPLFAIL "$T controls did not pass or no tree json: bars WITHHELD (see $LG)"; fi
  fi
  # e -- the frozen-gain forwards (each script's frozen file unchanged; forward needs the compliant tree json above)
  for pr in "a0k0_headroom_r8.py:raw/a0k0_headroom/FORWARD_$T.json" "a0k0_headroom_lon_r8.py:raw/a0k0_lon/FORWARD_LON_$T.json" "a0k0_headroom_retime_r8.py:raw/a0k0_retime/FORWARD_RETIME_$T.json"; do
    S=${pr%%:*}; A=${pr#*:}
    if jq_ok "$PKGL/$A" "j.get('controls',{}).get('ALL_PASS') is True"; then mark FWDHAVE "$S $T"; continue; fi
    cpu_run "fwd_$S" "$PKGW/code" "$LG" "$PY" -X utf8 "$S" forward --tree "$T"
    if jq_ok "$PKGL/$A" "j.get('controls',{}).get('ALL_PASS') is True"; then mark FWDOK "$S $T -> $A"
    elif [ -s "$PKGL/$A" ]; then mark FWDCTRLFAIL "$S $T wrote $A but its controls are not ALL_PASS: read before quoting"; else mark FWDFAIL "$S $T (see $LG)"; fi
  done
  # g -- heading drift, as post30k_heading_r8.sh (the earlier list now starts at 30k)
  OUTF="$PKGL/raw/heading_drift/$HTAG.json"
  if [ -s "$OUTF" ]; then mark HEADHAVE "$HTAG"; else
    cpu_run heading "$PKGW" "$LG" "$PY" -X utf8 raw/heading_drift/heading_drift.py --new-root "$BRW/$T" --label-new "$LAB-40k" \
        --earlier "$LAB-30k=$BRW/step30000${SUF}_unc" "$LAB-20k=$BRW/step20000${SUF}_unc" "$LAB-15k=$BRW/step15000${SUF}_unc" "$LAB-10.5k=$BRW/step10500${SUF}_unc" "$LAB-5k=$BRW/step5000${SUF}_unc" \
        --out-dir raw/heading_drift --tag "$HTAG"
    if [ -s "$OUTF" ]; then mark HEADDONE "$HTAG -> raw/heading_drift/$HTAG.md"; else mark HEADFAIL "$HTAG wrote no json (see $LG)"; fi
  fi
  mark LANETREEDONE "$T"
)
lane() {
  lane_tree "$TE" ema EMA ema "HEADING_DRIFT_${TAGE}"
  lane_tree "$TR" raw RAW "" "HEADING_DRIFT_${TAGR}raw"
  for t in a0k0_headroom_r8.py:a0k0_headroom:A0K0_HEADROOM.md a0k0_headroom_lon_r8.py:a0k0_lon:A0K0_LON.md a0k0_headroom_retime_r8.py:a0k0_retime:A0K0_RETIME.md; do   # render: re-render each .md from its JSONs (backup once)
    s=${t%%:*}; r=${t#*:}; d=${r%%:*}; m=${r#*:}
    [ -e "$PKGL/raw/$d/$m.pre40k" ] || cp "$PKGL/raw/$d/$m" "$PKGL/raw/$d/$m.pre40k" 2>/dev/null
    cpu_run render "$PKGW/code" "$LOGD_L/lane_render.log" "$PY" -X utf8 "$s" render || mark RENDERFAIL "$s (non-blocking)"
  done
  if [ -s "$PKGL/code/summarize_40k_r8.py" ]; then "$PY" -I "$PKGW/code/summarize_40k_r8.py" --pkg "$PKGW" --out-dir "$RAWR_W" > "$LOGD_L/summarize.log" 2>&1
    if [ -s "$RAWR_L/READ_40K.json" ]; then mark SUMMARYOK "$RAWR_W/READ_40K.md"; else mark SUMMARYFAIL "see $LOGD_L/summarize.log"; fi; fi
  mark LANEEND ""
}
# ---- drive ------------------------------------------------------------------------------------------------------------------ #
pull_metrics || mark PULLDEFER "metrics copy not available yet; the raw stage retries it"
( lane ) &
LANEPID=$!
echo "$LANEPID" > "$KIT/chain/chain_40k_lane.pid"
mark LANESPAWNED "lane pid $LANEPID"
ema_stage; RC=$?
if [ $RC -eq 0 ]; then echo ok > "$ST/ema.state"; else echo fail > "$ST/ema.state"; mark STAGEFAILED "ema rc $RC (continuing: the raw stage is independent)"; fi
raw_stage; RC=$?
if [ $RC -eq 0 ]; then echo ok > "$ST/raw.state"; else echo fail > "$ST/raw.state"; mark STAGEFAILED "raw rc $RC"; fi
wait "$LANEPID"
mark END "ema=$(cat "$ST/ema.state") raw=$(cat "$ST/raw.state")"
note "chain end"

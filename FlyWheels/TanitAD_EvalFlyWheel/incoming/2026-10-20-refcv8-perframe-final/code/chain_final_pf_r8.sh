#!/bin/sh
# chain_final_pf_r8.sh -- the BINDING per-frame final read of refcv8 at step 48,852 (SHIP-R8 AMENDMENT 3).   usage: sh chain_final_pf_r8.sh as=chain_final_r8
#   NEW file (EvalFlyWheel 2026-10-20, hold_r8pf).  It edits no file of the battery package or the launch tree; it runs the package's own tools (pull, extract, EMA gate, xcheck,
#   battery, pair_read, bank, the CPU-lane instruments) re-pointed, and three NEW pieces from this hold: run_battery_r8_pf.py (the roll child fed --nav-feed perframe_full),
#   pf_lane.py (the CPU lane re-pointed at the REGISTERED per-frame table) and feed_parity_pf.py + ship_r8_verdict_pf.py (clause 1b and the Amendment-3 verdict).
#   Markers (opaque ZZPF...ZZ) -> E:/refcv8_eval_kit/chain/chain_final_pf.log.  Terminal: ZZPFENDZZ (states every stage) or ZZPFABORTZZ.  Resumable: re-run the same command.
#   Assert on ARTIFACTS, never an exit code, never through a pipe.  Thor is touched ONLY by pull_final_r8.sh (read-only).  Nothing is written to G:.  No git.
#
#   a  PULL     pull_final_r8.sh -- the SAME script, log (chain/pull_final.log) and destination as chain_final_r8.sh, re-verified the same way (local md5 == sidecar == provenance == DONE).
#   b  EXTRACT  extract_ema_r8.py -- the SAME call and the SAME provenance/md5 records as chain_final_r8.sh (so its restart reads EXTRACTHAVE); EMA gate + xcheck re-run, json into THIS hold.
#   R  REGISTER the eval rolls REFUSE to start until C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256 holds 64 hex == sha256(FROZEN_GAINS_RETIME_train_pf.json) == its .sha256
#               record == its provenance (waits PF_REG_MAXWAIT_S, default 24 h); then REGISTRATION_GATE.json is written (its time precedes every pf roll: the verdict checks it).
#   c  GPU      ONE lock hold: run_battery_r8_pf.py, tag step48852ema_pf, seeds 0,1, standalone clock, --no-g0 (NO_G0_REASON stamped, as the per-clip EMA pass), one retry, then halt;
#               pair_read_r8.py vs refcv7-50,400 inside the hold; release; bank_battery_r8.py (sanitised) into THIS hold.
#   P  PER-CLIP chain_final_r8.sh is RESTARTED here, the moment stage c is terminal (success or give-up) -- and by the EXIT trap on every other exit path.
#   L  CPU LANE pf_lane.py install (REGISTERED table) -> clamp -> compliant -> TRAIN-FIT forward -> DEPLOY8 -> DEPLOY -> TAC -> prior parity; feed_parity_pf.py on both seeds' dumps
#               (per-clip token reference: the CLIP-FED step40000ema_unc dumps of the SAME windows -- the clip token does not depend on the checkpoint); ship_r8_verdict_pf.py.
#
#   PF-FIRST GUARANTEE (structural): chain_final_r8.sh is STOPPED before this chain starts (wait_final_pf_r8.sh refuses to launch it otherwise, and stage a refuses while its pid file is
#   alive) and is started again only in stage P, i.e. after the pf GPU stage is terminal.  The per-clip EMA battery therefore cannot hold the lock before the pf battery.  Chains that
#   honour yield48852_r8.sh's rule (ps `[c]hain_final_r8` alive + the per-clip log open + ckpt_full banked) yield to THIS chain too: its argv carries the token `as=chain_final_r8`
#   on purpose.  A job that ignores the rule can still win the lock (gpu_lock.py has no priority and is never broken): this chain then QUEUES, it does not preempt.
STEP=48852
TAG=step48852ema_pf
HOLDW="${PF_HOLDW:-C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final}"
HOLDL="${PF_HOLDL:-/c/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final}"
PKGW="${PF_R8PKGW:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
PKGL="${PF_R8PKGL:-/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
BAT="${PF_BAT:-C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code}"
KITW="${PF_KITW:-E:/refcv8_eval_kit}"; KIT="${PF_KIT:-/e/refcv8_eval_kit}"
BRW="${PF_BRW:-C:/Users/Admin/refcv8_eval_archive/battery}"; BRL="${PF_BRL:-/c/Users/Admin/refcv8_eval_archive/battery}"
PY="${PF_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
REGF="${PF_REGF:-C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256}"
FROZEN="${PF_FROZEN:-$HOLDW/raw/retime_trainfit_pf/FROZEN_GAINS_RETIME_train_pf.json}"
PFPKG="$HOLDW/pfpkg"
REF_CLIP="${PF_REF_CLIP:-C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc}"
EV9="${PF_EV9:-E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz}"; EV9MD5="${PF_EV9MD5:-6b5c7f207cffc3b7eb3cd527fd433599}"
PULLSH="${PF_PULL_SH:-$PKGL/code/pull_final_r8.sh}"; PULLLOG="${PF_PULL_LOG:-$KIT/chain/pull_final.log}"
EXTRACT_PY="${PF_EXTRACT_PY:-$PKGW/code/extract_ema_r8.py}"; XCHECK_PY="${PF_XCHECK_PY:-$PKGW/code/xcheck_model_ema_r8.py}"; EMACHK_PY="${PF_EMACHK_PY:-$PKGW/code/ema_check_r8.py}"
BATT_PY="${PF_BATT_PY:-$HOLDW/code/run_battery_r8_pf.py}"; PAIR_PY="${PF_PAIR_PY:-$PKGW/code/pair_read_r8.py}"; BANK_PY="${PF_BANK_PY:-$PKGW/code/bank_battery_r8.py}"
LANE_PY="${PF_LANE_PY:-$HOLDW/code/pf_lane.py}"; PAR_PY="${PF_PAR_PY:-$HOLDW/code/feed_parity_pf.py}"; VERD_PY="${PF_VERD_PY:-$HOLDW/code/ship_r8_verdict_pf.py}"
CFN_PIDF="${PF_CFN_PIDF:-$KIT/chain/chain_final.pid}"
CFN_RESTART="${PF_CFN_RESTART:-cd $PKGL && nohup sh code/chain_final_r8.sh < /dev/null >> $KIT/chain/chain_final.out 2>&1 &}"
STATE="${PF_STATE:-/c/Users/Admin/refcv8_pf/state}"
LOG="${PF_LOG:-$KIT/chain/chain_final_pf.log}"
POLL="${PF_POLL_S:-60}"; REG_MAXWAIT="${PF_REG_MAXWAIT_S:-86400}"
EXNEED="${PF_EXTRACT_NEED_GB:-5}"; MIN_COMMIT="${PF_MIN_COMMIT_GB:-4}"; LANENEED="${PF_LANE_NEED_GB:-5}"
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6
unset R8_NO_POSE_SYNC
CK="$KITW/ckpt/ckpt_$STEP.pt"; EMA="$KITW/ckpt/ckpt_ema_$STEP.pt"; CFG="$KITW/ckpt/config.json"; CKL="$KIT/ckpt"
FULL="$KITW/ckpt/ckpt_full_$STEP.pt"; FULLL="$KIT/ckpt/ckpt_full_$STEP.pt"
RAWW="$HOLDW/raw/$TAG"; RAWL="$HOLDL/raw/$TAG"; WORK="$BRW/$TAG"; WORKL="$BRL/$TAG"
mkdir -p "$KIT/chain" "$STATE" "$RAWL" "$HOLDL/raw/feed_parity" "$HOLDL/raw/ship_r8" "$STATE/logs"
mark() { echo "ZZPF$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
ts() { echo "$(date +%s) $(date +%FT%T) $1" >> "$STATE/chain_final_pf_timings.txt"; }
jq_ok() {
  _f=$(cygpath -m "$1" 2>/dev/null || echo "$1")
  "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$_f"
}
jget() {
  _f=$(cygpath -m "$1" 2>/dev/null || echo "$1")
  "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); print($2)
except Exception: print('')" "$_f"
}
sha256_of() { _s=$(sha256sum "$1" 2>/dev/null | cut -d' ' -f1); [ "${#_s}" -eq 64 ] && echo "$_s" || echo ""; }
free_gb() {
  if [ -n "$PF_FREEGB_CMD" ]; then v=$(sh -c "$PF_FREEGB_CMD" 2>/dev/null)
  else v=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" 2>/dev/null | tr -d '\r ' | tr ',' '.'); fi
  case "$v" in ''|*[!0-9.]*) echo 0 ;; *) echo "$v" ;; esac
}
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit (a+0>=b+0)?0:1}'; }
mem_gate() {
  _t0=$(date +%s); _n=0
  while :; do
    _h1=$(free_gb)
    if ge "$_h1" "$1"; then sleep "${PF_GATE_GAP_S:-20}"; _h2=$(free_gb); ge "$_h2" "$1" && return 0; fi
    _n=$((_n+1)); [ $((_n % 20)) -eq 1 ] && mark MEMWAIT "$2: free commit ${_h1} GB < $1"
    [ $(( $(date +%s) - _t0 )) -gt "${PF_MEM_MAXWAIT_S:-43200}" ] && { mark MEMTIMEOUT "$2"; return 1; }
    sleep "${PF_MEM_POLL_S:-30}"
  done
}
cfn_alive() { [ -s "$CFN_PIDF" ] && kill -0 "$(cat "$CFN_PIDF" 2>/dev/null)" 2>/dev/null; }
restart_cfn() {   # idempotent: never starts a second per-clip chain; its own pid file refuses a duplicate anyway (ZZCFNDUPZZ)
  [ -e "$STATE/cfn_restarted" ] && return 0
  if cfn_alive; then mark CFNALIVE "chain_final_r8 (pid $(cat "$CFN_PIDF")) is already running: not restarted"; : > "$STATE/cfn_restarted"; return 0; fi
  sh -c "$CFN_RESTART"
  : > "$STATE/cfn_restarted"
  mark CFNRESTART "$1: chain_final_r8.sh restarted (per-clip reads, never binding: Amendment 3 (iv))"
}
# ---- single instance ---------------------------------------------------------------------------------------- #
PIDF="$STATE/chain_final_pf.pid"
if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then mark DUP "another chain_final_pf (pid $(cat "$PIDF")) is alive; exiting"; exit 5; fi
echo $$ > "$PIDF"
GPU_TERMINAL=0; ABORTED=0
on_exit() {
  [ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"
  # the per-clip chain comes back on every TERMINAL path (stage c done or failed, or an ABORT before it).  A non-terminal exit (killed) leaves it to the guardian,
  # which relaunches this resumable chain once and otherwise restarts chain_final_r8 itself once the pf lock is gone.
  if [ "$GPU_TERMINAL" = 1 ] || [ "$ABORTED" = 1 ]; then restart_cfn "exit path"; else mark CFNHOLD "non-terminal exit: the guardian relaunches this chain or restarts chain_final_r8"; fi
}
trap on_exit EXIT
trap 'exit 143' TERM INT HUP
rm -f "$STATE/cfn_restarted" "$STATE/gpu_started"
abort() { ABORTED=1; mark ABORT "stage $1: $2"; exit "$3"; }
mark START "pid $$ step $STEP tag $TAG argv '$*'"; ts armed
# ---- a: pull (the per-clip chain must not be pulling at the same time) --------------------------------------- #
cfn_alive && abort a "chain_final_r8 (pid $(cat "$CFN_PIDF")) is alive: two pullers would race on the same .part; stop it first (wait_final_pf_r8.sh)" 30
pull_state() {
  awk '/ZZPFNSTARTZZ/ {st="wait"; next}
       /ZZPFNDONEZZ/ {st="done"; next}
       /ZZPFNALREADYZZ/ {if (st != "done") st="already"; next}
       /ZZPFN(SSHDEAD|SUPDEAD|TIMEOUT|WRONGSTEP|HOLE|SETTLEFAIL|NODISK|BADSIZE|INCONCLUSIVE|CHANGED|SIZEMISMATCH|MD5MISMATCH)ZZ/ {match($0, /ZZPFN[A-Z0-9]*ZZ/); st="fail:" substr($0, RSTART, RLENGTH); next}
       END {print (st == "" ? "none" : st)}' "$PULLLOG" 2>/dev/null
}
pull_stage() {
  mark PULLSTART "$PULLSH"
  FIN_KIT="$KIT" FIN_CKPT_DIR="$CKL" FIN_LOG="$PULLLOG" FIN_STEP="$STEP" sh "$PULLSH" > "$STATE/logs/pull.out" 2>&1
  S=$(pull_state)
  case "$S" in
    done|already) ;;
    fail:*) mark PULLFAIL "pull_final reached ${S#fail:}"; return 20 ;;
    *) mark PULLFAIL "pull_final ended in state '$S' with no terminal marker"; return 22 ;;
  esac
  [ -s "$FULLL" ] || { mark MD5BAD "no $FULLL"; return 21; }
  LMD=$(md5sum "$FULLL" | cut -d' ' -f1); SIDE=$(cut -d' ' -f1 < "$FULLL.md5" 2>/dev/null | head -n 1)
  PROV=$(jget "$FULLL.provenance.json" "j['md5']"); PSTEP=$(jget "$FULLL.provenance.json" "j['step']")
  DONEMD=$(grep -a 'ZZPFNDONEZZ' "$PULLLOG" 2>/dev/null | tail -n 1 | sed -n 's/.* ckpt_md5 \([0-9a-f]\{32\}\) .*/\1/p')
  if [ "$S" = already ] && [ -z "$DONEMD" ]; then DONEMD="$PROV"; fi
  for m in "$LMD" "$SIDE" "$PROV" "$DONEMD"; do
    [ "${#m}" -eq 32 ] || { mark MD5INCONCLUSIVE "an md5 is not 32 chars (local '$LMD' sidecar '$SIDE' provenance '$PROV' done '$DONEMD')"; return 21; }
  done
  [ "$PSTEP" = "$STEP" ] || { mark MD5BAD "provenance step '$PSTEP' != $STEP"; return 21; }
  if [ "$LMD" = "$SIDE" ] && [ "$LMD" = "$PROV" ] && [ "$LMD" = "$DONEMD" ]; then SNAPMD5="$LMD"; mark MD5OK "ckpt_full_$STEP.pt md5 $SNAPMD5"
  else mark MD5BAD "local $LMD sidecar $SIDE provenance $PROV done $DONEMD"; return 21; fi
  jq_ok "$CKL/summary_thor_$STEP.json" "j.get('done') is True and int(j.get('step', j.get('final_step', -1)))==$STEP" || { mark SUMJSONBAD "summary_thor_$STEP.json not done@$STEP"; return 21; }
  return 0
}
pull_stage; RC=$?; [ $RC -eq 0 ] || abort a "final checkpoint not usable" $RC
ts "pull verified"
# ---- b: extract (same records as the per-clip chain) + EMA gate + xcheck into this hold ---------------------- #
EXREC="$KITW/ckpt/ckpt_$STEP.provenance.extract_ema.json"
extract_ok() { [ -s "$CKL/ckpt_$STEP.pt" ] && [ -s "$CKL/ckpt_ema_$STEP.pt" ] && jq_ok "$EXREC" "j.get('ok') is True and int(j.get('step',-1))==$STEP and j.get('src_md5')=='$SNAPMD5'"; }
extract_stage() {
  if extract_ok && [ -s "$CKL/ckpt_$STEP.md5" ]; then mark EXTRACTHAVE "pair already derived from this pull"; return 0; fi
  FREEKB=$(df -k "$CKL" | tail -n 1 | awk '{print $4}')
  [ -n "$FREEKB" ] && [ "$FREEKB" -lt "${PF_MIN_FREE_KB:-3500000}" ] && { mark EXTRACTNODISK "free ${FREEKB} kB on $CKL"; return 24; }
  mem_gate "$EXNEED" extract || return 25
  mark EXTRACTSTART "expect-step $STEP"
  FORCE=""; { [ -e "$CKL/ckpt_$STEP.pt" ] || [ -e "$CKL/ckpt_ema_$STEP.pt" ]; } && FORCE="--force"
  "$PY" "$EXTRACT_PY" extract --src "$FULL" --config "$CFG" --out-dir "$KITW/ckpt" --expect-step "$STEP" --min-commit-gb "$MIN_COMMIT" $FORCE \
      --provenance "thor_ckpt_pt_md5=$SNAPMD5" "snapshot=$FULL" "pull_log=$PULLLOG" "chain=chain_final_pf_r8.sh (the per-clip chain_final_r8.sh reads this record as EXTRACTHAVE)" > "$STATE/logs/extract.out" 2>&1
  extract_ok || { mark EXTRACTFAIL "no ok provenance for step $STEP: $(tail -n 1 "$STATE/logs/extract.out" | cut -c1-300)"; return 21; }
  M_RAW=$(md5sum "$CKL/ckpt_$STEP.pt" | cut -d' ' -f1); M_EMA=$(md5sum "$CKL/ckpt_ema_$STEP.pt" | cut -d' ' -f1)
  { [ "${#M_RAW}" -eq 32 ] && [ "${#M_EMA}" -eq 32 ]; } || { mark EXTRACTFAIL "cannot md5 the derived pair"; return 21; }
  { echo "$M_RAW  ckpt_$STEP.pt"; echo "$M_EMA  ckpt_ema_$STEP.pt"; echo "$SNAPMD5  ckpt_full_$STEP.pt (thor final ckpt.pt, step $STEP)"; } > "$CKL/ckpt_$STEP.md5.tmp" && mv -f "$CKL/ckpt_$STEP.md5.tmp" "$CKL/ckpt_$STEP.md5"
  mark EXTRACTOK "ckpt_$STEP.pt $M_RAW ckpt_ema_$STEP.pt $M_EMA"
}
ema_gate() {
  GATEJ="$RAWW/EMA_CHECK_step$STEP.json"
  M_RAW=$(md5sum "$CKL/ckpt_$STEP.pt" | cut -d' ' -f1); M_EMA=$(md5sum "$CKL/ckpt_ema_$STEP.pt" | cut -d' ' -f1)
  PIN="j['md5']['raw']=='$M_RAW' and j['md5']['ema']=='$M_EMA'"
  if ! jq_ok "$GATEJ" "j.get('ok') is True and $PIN"; then
    mem_gate "$EXNEED" ema_check || return 25
    "$PY" "$EMACHK_PY" --raw "$CK" --ema "$EMA" --config "$CFG" --out "$GATEJ" > "$STATE/logs/ema_check.out" 2>&1
  fi
  if jq_ok "$GATEJ" "j.get('ok') is True and $PIN and j['tensors']['n_float_differing']>0 and j['strict_load']['missing']==0 and j['strict_load']['unexpected']==0"; then
    mark EMAGATEOK "strict load clean, weights differ from raw"; return 0; fi
  mark EMAGATEFAIL "ema_check not ok: $GATEJ"; return 21
}
xcheck_stage() {
  TH="$CKL/model_ema_${STEP}_thor.pt"; XJ="$RAWW/XCHECK_MODEL_EMA_step$STEP.json"
  if [ ! -s "$TH" ]; then mark XCHECKNA "no trainer-written model_ema.pt was pulled: ckpt.pt['ema'] is the only EMA record"; return 0; fi
  M_TH=$(md5sum "$TH" | cut -d' ' -f1); M_EMA=$(md5sum "$CKL/ckpt_ema_$STEP.pt" | cut -d' ' -f1)
  { [ "${#M_TH}" -eq 32 ] && [ "${#M_EMA}" -eq 32 ]; } || { mark XCHECKFAIL "cannot md5 the two EMA files"; return 21; }
  PINX="j['md5']['thor']=='$M_TH' and j['md5']['derived']=='$M_EMA'"
  if jq_ok "$XJ" "j.get('ok') is True and $PINX"; then mark XCHECKHAVE ""; return 0; fi
  mem_gate "$EXNEED" xcheck || return 25
  "$PY" "$XCHECK_PY" --thor-ema "$KITW/ckpt/model_ema_${STEP}_thor.pt" --derived "$EMA" --step "$STEP" --out "$XJ" > "$STATE/logs/xcheck.out" 2>&1
  if jq_ok "$XJ" "j.get('ok') is True and $PINX"; then mark XCHECKOK "trainer model_ema.pt == derived EMA on every tensor"; return 0; fi
  mark XCHECKFAIL "the trainer-written EMA and the derived EMA DISAGREE (or the check could not run): $XJ"; return 21
}
extract_stage; RC=$?; [ $RC -eq 0 ] || abort b "extract" $RC
ema_gate; RC=$?; [ $RC -eq 0 ] || abort b "ema gate" $RC
xcheck_stage; RC=$?; [ $RC -eq 0 ] || abort b "ema cross-check" $RC
ts "extract + ema gate + xcheck"
# ---- R: the registration gate (Amendment 3 (ii)): no pf eval roll before the per-frame table is registered ------ #
GATEJSON="$HOLDW/raw/REGISTRATION_GATE.json"
reg_check() {   # 0 iff registered == frozen file == its .sha256 record == provenance, all 64 hex (an empty string is never a match)
  R=$(head -n 1 "$REGF" 2>/dev/null | awk '{print $1}' | tr -cd '0-9a-fA-F' | tr 'A-F' 'a-f')     # first token, hex only: a UTF-8 BOM / CRLF / "sha  file" form all read the same
  F=$(sha256_of "$FROZEN"); D=$(cut -d' ' -f1 < "${FROZEN%.json}.sha256" 2>/dev/null | head -n 1 | tr -d '\r\n ')
  P=$(jget "${FROZEN%.json}.provenance.json" "j['sha256']")
  for x in "$R" "$F" "$D" "$P"; do [ "${#x}" -eq 64 ] || return 1; done
  [ "$R" = "$F" ] && [ "$R" = "$D" ] && [ "$R" = "$P" ]
}
if jq_ok "$GATEJSON" "len(j.get('registered_sha256',''))==64" && reg_check; then mark REGHAVE "$(jget "$GATEJSON" "j['registered_sha256']")"
else
  _t0=$(date +%s); _said=0
  until reg_check; do
    [ "$_said" = 0 ] && { mark REGWAIT "waiting for $REGF == sha256($FROZEN) (cap ${REG_MAXWAIT} s): NO pf eval roll before the per-frame table is registered"; _said=1; }
    [ $(( $(date +%s) - _t0 )) -gt "$REG_MAXWAIT" ] && abort R "REGISTERED_PF_TABLE.sha256 never matched the frozen per-frame table: SHIP-R8 (Amendment 3) INCOMPLETE" 40
    sleep "$POLL"
  done
  "$PY" -I -c "import json,sys,time
json.dump({'registered_sha256':sys.argv[2],'registration_file':sys.argv[3],'frozen_table':sys.argv[4],'passed_epoch':time.time(),'passed_iso':time.strftime('%FT%T%z'),
           'rule':'SHIP-R8 AMENDMENT 3 (ii): the per-frame table sha256 is registered BEFORE any 48,852 pf eval roll; this record is written before the GPU stage'},
          open(sys.argv[1],'w',encoding='utf-8'),indent=1)" "$GATEJSON" "$(sha256_of "$FROZEN")" "$REGF" "$FROZEN"
  jq_ok "$GATEJSON" "len(j.get('registered_sha256',''))==64" || abort R "could not write $GATEJSON" 41
  mark REGOK "$(jget "$GATEJSON" "j['registered_sha256']")"
fi
ts "registration gate"
# ---- c: the pf battery (ONE lock hold for both seeds + the paired read) ------------------------------------------ #
JOB="refcv8-battery-$TAG"
pass_done() { jq_ok "$WORK/battery_summary.json" "j.get('bars') and j.get('finished') and 'r8' in j and (j.get('pf') or {}).get('all_seeds_fed') is True"; }
gpu_stage() (
  if pass_done && [ -s "$WORKL/PAIRED_READ_$TAG.json" ]; then mark GPUHAVE "$WORK"; exit 0; fi
  rm -f "$KITW/chain/pf_wc.json" "$KITW/chain/pf_lock.json"
  "$PY" "$BAT/wait_commit.py" --min-gb 6 --max-wait-s 43200 --out "$KITW/chain/pf_wc.json" >> "$STATE/logs/lockwait.log" 2>&1
  jq_ok "$KIT/chain/pf_wc.json" "j.get('met') is True" || { mark NOCOMMIT "$JOB"; exit 3; }
  mark LOCKWAIT "$JOB (a foreign holder is never broken; no priority: this chain queues)"
  "$PY" "$BAT/gpu_lock.py" acquire --job "$JOB" --pid $$ --max-wait-s 172800 --out "$KITW/chain/pf_lock.json" >> "$STATE/logs/lockwait.log" 2>&1
  jq_ok "$KIT/chain/pf_lock.json" "j.get('ok') is True" || { mark NOLOCK "$JOB"; exit 3; }
  "$PY" "$BAT/chain_marker.py" create --chain "$JOB" --pid $$ >> "$STATE/logs/lockwait.log" 2>&1
  trap '"$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$STATE/logs/lockwait.log" 2>&1; "$PY" "$BAT/chain_marker.py" remove --chain "$JOB" --pid $$ >> "$STATE/logs/lockwait.log" 2>&1; mark RELEASED "$JOB"' EXIT
  trap 'exit 143' TERM INT HUP                                  # a subshell does not inherit the parent's traps: a TERM must still run the release above
  mark LOCKED "$JOB"; ts "lock"
  mkdir -p "$KIT/battery/seam_tele_$TAG"
  run_pass() {
    ( export R8_NO_POSE_SYNC=1 R8_SEAM_TELE_DIR="$KITW/battery/seam_tele_$TAG"
      "$PY" "$BATT_PY" --ckpt "$EMA" --config "$CFG" --no-g0 --tag "$TAG" --infer-seeds 0,1 --lock-held --skip-roll --out-root "$BRW" >> "$STATE/logs/battery_run_$TAG.log" 2>&1 )
  }
  run_pass
  if ! pass_done; then mark BATTRETRY "pass incomplete, retrying once (resumes from complete seed dumps)"; run_pass; fi
  pass_done || { mark BATTFAIL "no complete, fed battery_summary after one retry (see $STATE/logs/battery_run_$TAG.log)"; exit 4; }
  mark BATTDONE "$WORK (all seeds fed perframe_full)"; ts "battery"
  ( export R8_NO_POSE_SYNC=1
    "$PY" "$PAIR_PY" --root "$WORK" --clock standalone --config "$CFG" --out-json "$WORK/PAIRED_READ_$TAG.json" --out-md "$WORK/PAIRED_READ_$TAG.md" > "$STATE/logs/pair_read_$TAG.log" 2>&1 )
  if [ ! -s "$WORKL/PAIRED_READ_$TAG.json" ]; then
    mark PAIRRETRY "no PAIRED_READ json, retrying once"
    ( export R8_NO_POSE_SYNC=1
      "$PY" "$PAIR_PY" --root "$WORK" --clock standalone --config "$CFG" --out-json "$WORK/PAIRED_READ_$TAG.json" --out-md "$WORK/PAIRED_READ_$TAG.md" >> "$STATE/logs/pair_read_$TAG.log" 2>&1 )
  fi
  [ -s "$WORKL/PAIRED_READ_$TAG.json" ] || { mark PAIRFAIL "see $STATE/logs/pair_read_$TAG.log"; exit 4; }
  mark PAIRED "$WORK/PAIRED_READ_$TAG.json"
  exit 0
)
: > "$STATE/gpu_started"
gpu_stage; GRC=$?
GPU_TERMINAL=1
ts "gpu stage rc $GRC"
# ---- P: the per-clip chain comes back NOW (after the pf GPU stage is terminal, whatever its result) -------------- #
restart_cfn "after the pf GPU stage (rc $GRC)"
[ $GRC -eq 0 ] || abort c "pf battery / paired read did not complete (rc $GRC)" $GRC
"$PY" -I -c "
import json,sys
json.dump({'tag':sys.argv[2],'status':'PROVISIONAL','no_g0_reason':'EMA weights of the final step, per-frame fed (Amendment 3); the per-clip EMA pass gets no G0 either (the step-30,000 policy); the G0 of the raw weights at this step is the nearest instrument, stated beside, not substituted',
           'ckpt':sys.argv[3]}, open(sys.argv[1],'w',encoding='utf-8'),indent=1)" "$RAWW/NO_G0_REASON_$TAG.json" "$TAG" "$EMA"
cp "$RAWL/NO_G0_REASON_$TAG.json" "$WORKL/NO_G0_REASON_$TAG.json" 2>/dev/null
"$PY" "$BANK_PY" "$BRW" "$RAWW" --tag step48852ema --passes pf --chain-log "$KITW/chain/chain_final_pf.log" --status "PROVISIONAL: EMA weights, per-frame fed, no G0 (NO_G0_REASON_$TAG.json)" > "$STATE/logs/bank.out" 2>&1
if jq_ok "$RAWL/BATTERY_SUMMARY_step48852ema.json" "j['passes']['pf']['status']=='COMPLETE'"; then mark BANKED "$RAWW/BATTERY_SUMMARY_step48852ema.json"
else mark BANKFAIL "see $STATE/logs/bank.out (the archive tree $WORK is intact; the lane continues)"; fi
# ---- L: the CPU lane on the REGISTERED per-frame table ---------------------------------------------------------- #
lane() {   # $1 = stage, $2 = artifact that proves it (skipped if present); one retry
  if [ -s "$2" ]; then mark "LANEHAVE_$1" ""; return 0; fi
  mem_gate "$LANENEED" "lane_$1" || return 25
  _try=0
  while [ "$_try" -lt 2 ]; do
    env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PF_R8_CODE="$PKGW/code" PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKGW/code" \
        "$PY" -X utf8 "$LANE_PY" "$1" --pkg "$PFPKG" --tree "$TAG" --registered "$REGF" >> "$STATE/logs/lane_$1.log" 2>&1
    [ -s "$2" ] && { mark "LANEOK_$1" "$2"; return 0; }
    _try=$((_try+1))
  done
  mark "LANEFAIL_$1" "no $2 (see $STATE/logs/lane_$1.log)"; return 4
}
if [ -s "$PFPKG/PF_LANE_INSTALL.json" ] && jq_ok "$PFPKG/PF_LANE_INSTALL.json" "j.get('installed_sha256')=='$(sha256_of "$FROZEN")' and j.get('rs_table_loads_and_matches') is True"; then mark LANEINSTALLHAVE ""
else
  env CUDA_VISIBLE_DEVICES=-1 PF_R8_CODE="$PKGW/code" PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKGW/code" \
      "$PY" -X utf8 "$LANE_PY" install --pkg "$PFPKG" --table "$FROZEN" --registered "$REGF" > "$STATE/logs/lane_install.log" 2>&1
  jq_ok "$PFPKG/PF_LANE_INSTALL.json" "j.get('rs_table_loads_and_matches') is True" && mark LANEINSTALLOK "" || abort L "pf_lane install failed (see $STATE/logs/lane_install.log)" 42
fi
R="$PFPKG/raw"
lane clamp "$R/clamp_posthoc/trees/$TAG.json"
lane compliant "$R/compliant_reread/trees/$TAG.json"
lane tf_forward "$R/retime_trainfit/FORWARD_TRAINFIT_$TAG.json"
lane d8 "$R/retime_deploy/FORWARD_DEPLOY8_$TAG.json"
lane df "$R/retime_deploy/FORWARD_DEPLOY_$TAG.json"
lane tac "$R/retime_deploy/TAC_$TAG.json"
lane parity "$R/retime_deploy/PRIOR_PARITY.json"
# ---- 1b: feed parity on the tree's own dumps --------------------------------------------------------------------- #
for s in 0 1; do
  PJ="$HOLDW/raw/feed_parity/FEED_PARITY_${TAG}_s$s.json"
  if [ ! -s "$HOLDL/raw/feed_parity/FEED_PARITY_${TAG}_s$s.json" ]; then
    "$PY" -X utf8 "$PAR_PY" --pf-dump "$WORK/dump_s$s" --clip-dump "$REF_CLIP/dump_s$s" --v9 "$EV9" --v9-md5 "$EV9MD5" --out "$PJ" \
        --label "$TAG s$s vs the clip-fed step40000ema_unc dump of the same windows (clip token is checkpoint-independent)" --known-values 3 > "$STATE/logs/parity_s$s.log" 2>&1
  fi
  mark "PARITY_s$s" "$(jget "$PJ" "j['status']")"
done
# ---- verdict ----------------------------------------------------------------------------------------------------- #
"$PY" -X utf8 "$VERD_PY" --tree "$TAG" --pf-pkg "$PFPKG" --registered "$REGF" --frozen "$FROZEN" --parity-dir "$HOLDW/raw/feed_parity" \
    --gate-json "$GATEJSON" --out-dir "$HOLDW/raw/ship_r8" > "$STATE/logs/verdict.log" 2>&1
OV=$(jget "$HOLDL/raw/ship_r8/SHIP_R8_$TAG.json" "j['overall']")
[ -n "$OV" ] && mark "VERDICT_$OV" "$HOLDW/raw/ship_r8/SHIP_R8_$TAG.json" || mark VERDICTMISSING "see $STATE/logs/verdict.log"
mark END "pull MD5OK, extract OK, registration $(jget "$GATEJSON" "j['registered_sha256'][:12]"), battery fed, paired, verdict ${OV:-MISSING}"
ts end

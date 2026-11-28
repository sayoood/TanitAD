#!/bin/sh
# chain_trainfit_pf.sh -- SHIP-R8 AMENDMENT 3 (ii): the per-frame TRAIN-FIT rolls (EMA-30k, A6 train139, inference seeds 0 and 1, os-only, --nav-feed perframe_full)
#   -> FEED PARITY on the roll artifacts -> panels -> FREEZE (the UNCHANGED fit code, NEW dir) -> TRAINFIT_PF_FROZEN.md.   NEW file (2026-10-20, EvalFlyWheel, hold_r8pf).
#   start (detached, ONCE):  nohup sh <hold>/code/chain_trainfit_pf.sh < /dev/null >> C:/Users/Admin/refcv8_pf/logs/chain_trainfit_pf.out 2>&1 &
#   alive?  ps -ef | grep -c '[c]hain_trainfit_pf'      progress:  grep -a ZZTFPF E:/refcv8_eval_kit/chain/trainfit_pf.log | tail
#   Resumable: a seed whose roll json reads ok (pf.ok true, 139 episodes, the clip-fed roll's window count, cuda) is skipped; parity / panels / freeze are re-run
#   only while their artifact is missing.  The frozen table is NEVER re-frozen in place (trainfit_pf.py refuses).
#   GPU: the shared O_EXCL lock (battery/code/gpu_lock.py), ONE hold for both seeds (ESTIMATED ~2.2 GPU-h: the clip-fed rolls took 4,183 s + 3,627 s, MEASURED
#   2026-10-19 trainfit_chain.log); a foreign holder is NEVER broken.  The job name carries the 48,852 tag on purpose: this table is a precondition of the binding
#   48,852 per-frame read, and the nav-feed waiter's rule (navfeed_common.sh final_py_children) yields to python processes naming that tag.
#   One retry per seed, then HALT (no freeze).  Assert on ARTIFACTS only, never an exit code, never through a pipe.  Nothing is written to G:.  No git.
#   Env overrides (TFPF_*) exist ONLY so the control flow can be dry-run with stub tools (test_chain_pf_flow.py).
HOLDW="${TFPF_HOLDW:-C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final}"
HOLDL="${TFPF_HOLDL:-/c/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final}"
BAT="${TFPF_BAT:-C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code}"
R8CODE="${TFPF_R8CODE:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code}"
PY="${TFPF_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
ROOTW="${TFPF_ROOTW:-C:/Users/Admin/refcv8_pf}"; ROOTL="${TFPF_ROOTL:-/c/Users/Admin/refcv8_pf}"
TREE="step30000ema_train139_pf"
TFW="$ROOTW/trainfit/$TREE"; TFL="$ROOTL/trainfit/$TREE"
CLIPREF_L="${TFPF_CLIPREF_L:-/e/refcv8_eval_kit/trainfit/step30000ema_train139}"; CLIPREF_W="${TFPF_CLIPREF_W:-E:/refcv8_eval_kit/trainfit/step30000ema_train139}"
CKPT="${TFPF_CKPT:-E:/refcv8_eval_kit/ckpt/ckpt_ema_30000.pt}"; CKPT_L="${TFPF_CKPT_L:-/e/refcv8_eval_kit/ckpt/ckpt_ema_30000.pt}"
CKMD5F="${TFPF_CKMD5F:-/e/refcv8_eval_kit/ckpt/ckpt_30000.md5}"
CFG="${TFPF_CFG:-E:/refcv8_eval_kit/ckpt/config.json}"
V9="${TFPF_V9:-E:/refcv8_eval_kit/data/refcv8_v9labels/release_train4572/v9_labels_train.npz}"
V9MD5="83e30bd3791f0767707eade5f4671759"      # == the launch argv's --r8-v9-md5 (E:/refcv8_eval_kit/ckpt/config.json argv); Thor md5sum identical (MEASURED 2026-10-20)
ROLLPY="${TFPF_ROLLPY:-$HOLDW/code/roll_seed_r8_train_pf.py}"
PARPY="${TFPF_PARPY:-$HOLDW/code/feed_parity_pf.py}"
FITPY="${TFPF_FITPY:-$HOLDW/code/trainfit_pf.py}"
OUTW="$HOLDW/raw/retime_trainfit_pf"; OUTL="$HOLDL/raw/retime_trainfit_pf"
LOG="${TFPF_LOG:-/e/refcv8_eval_kit/chain/trainfit_pf.log}"
JOB="refcv8-trainfit-pf-48852"
NEED="${TFPF_NEED_GB:-10}"; CPUNEED="${TFPF_CPU_NEED_GB:-6}"
LOCKWAIT="${TFPF_LOCK_WAIT_S:-43200}"
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TFL" "$OUTL" "$ROOTL/state" "$ROOTL/logs" "$(dirname "$LOG")"
mark() { echo "ZZTFPF$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
jq_ok() {   # $1 = file, $2 = python boolean expression over `j`
  _f=$(cygpath -m "$1" 2>/dev/null || echo "$1")
  "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$_f"
}
jget() { _f=$(cygpath -m "$1" 2>/dev/null || echo "$1"); "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); print($2)
except Exception: print('')" "$_f"; }
free_gb() {
  if [ -n "$TFPF_FREEGB_CMD" ]; then v=$(sh -c "$TFPF_FREEGB_CMD" 2>/dev/null)
  else v=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" 2>/dev/null | tr -d '\r ' | tr ',' '.'); fi
  case "$v" in ''|*[!0-9.]*) echo 0 ;; *) echo "$v" ;; esac
}
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit (a+0>=b+0)?0:1}'; }
mem_gate() {   # $1 GB, $2 name: two checks ${TFPF_GATE_GAP_S:-60} s apart; gives up after TFPF_MEM_MAXWAIT_S
  _t0=$(date +%s); _n=0
  while :; do
    _h1=$(free_gb)
    if ge "$_h1" "$1"; then sleep "${TFPF_GATE_GAP_S:-60}"; _h2=$(free_gb); ge "$_h2" "$1" && { mark MEMOK "$2: free commit ${_h1}/${_h2} GB"; return 0; }; fi
    _n=$((_n+1)); [ $((_n % 20)) -eq 1 ] && mark MEMWAIT "$2: free commit ${_h1} GB < $1"
    [ $(( $(date +%s) - _t0 )) -gt "${TFPF_MEM_MAXWAIT_S:-43200}" ] && { mark MEMTIMEOUT "$2"; return 1; }
    sleep "${TFPF_MEM_POLL_S:-30}"
  done
}
PIDF="$ROOTL/state/trainfit_pf.pid"
if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then mark DUP "another chain_trainfit_pf (pid $(cat "$PIDF")) is alive; exiting"; exit 5; fi
echo $$ > "$PIDF"
HAVE_LOCK=0
gpu_release() {
  [ "$HAVE_LOCK" = 1 ] || return 0
  "$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$ROOTL/logs/trainfit_pf_lock.log" 2>&1
  "$PY" "$BAT/chain_marker.py" remove --chain "$JOB" --pid $$ >> "$ROOTL/logs/trainfit_pf_lock.log" 2>&1
  HAVE_LOCK=0; mark RELEASED "$JOB"
}
finish() { gpu_release; [ "$(cat "$PIDF" 2>/dev/null)" = "$$" ] && rm -f "$PIDF"; }
trap finish EXIT
trap 'exit 143' TERM INT HUP
mark QSTART "pid $$ tree $TREE (perframe_full) -> $TFW ; frozen table -> $OUTW"
# ---- inputs: checkpoint + v9 release (an empty string is INCONCLUSIVE, never a match) ------------------------------------------------ #
CK_ACT=$(md5sum "$CKPT_L" 2>/dev/null | cut -d' ' -f1)
CK_REC=$(grep 'ckpt_ema_30000.pt' "$CKMD5F" 2>/dev/null | head -n 1 | cut -d' ' -f1)
if [ "${#CK_ACT}" -ne 32 ] || [ "${#CK_REC}" -ne 32 ] || [ "$CK_ACT" != "$CK_REC" ]; then mark CKPTBAD "ckpt_ema_30000.pt md5 '$CK_ACT' vs record '$CK_REC'"; exit 20; fi
mark CKPTOK "ckpt_ema_30000.pt md5 $CK_ACT (== $CKMD5F)"
V9_ACT=$(md5sum "$(cygpath -u "$V9" 2>/dev/null || echo "$V9")" 2>/dev/null | cut -d' ' -f1)
if [ "${#V9_ACT}" -ne 32 ] || [ "$V9_ACT" != "$V9MD5" ]; then mark V9BAD "train v9 md5 '$V9_ACT' != $V9MD5"; exit 20; fi
mark V9OK "train v9 release md5 $V9_ACT"
# ---- the rolls: ONE lock hold for both seeds ----------------------------------------------------------------------------------------- #
want_n() { jget "$CLIPREF_L/roll_s$1.json" "int(j['n_windows'])"; }
roll_ok() {
  _w=$(want_n "$1"); [ -n "$_w" ] || return 1
  jq_ok "$TFL/roll_s$1.json" "j.get('smoke') is False and int(j.get('n_windows',0))==$_w and j.get('n_episodes')==139 and j.get('device')=='cuda' and (j.get('pf') or {}).get('ok') is True" \
    && [ -s "$TFL/dump_s$1/manifest.json" ]
}
NEEDROLL=""
for s in 0 1; do if roll_ok "$s"; then mark ROLLHAVE "seed $s"; else NEEDROLL="$NEEDROLL $s"; fi; done
if [ -n "$NEEDROLL" ]; then
  mem_gate "$NEED" rolls || exit 11
  rm -f "$ROOTL/state/trainfit_pf_lock.json"
  mark LOCKWAIT "job $JOB (max ${LOCKWAIT}s; a foreign holder is never broken)"
  "$PY" "$BAT/gpu_lock.py" acquire --job "$JOB" --pid $$ --max-wait-s "$LOCKWAIT" --out "$ROOTW/state/trainfit_pf_lock.json" >> "$ROOTL/logs/trainfit_pf_lock.log" 2>&1
  jq_ok "$ROOTL/state/trainfit_pf_lock.json" "j.get('ok') is True" || { mark NOLOCK "not acquired within ${LOCKWAIT}s"; exit 11; }
  HAVE_LOCK=1
  "$PY" "$BAT/chain_marker.py" create --chain "$JOB" --pid $$ >> "$ROOTL/logs/trainfit_pf_lock.log" 2>&1
  mark LOCKED "seeds:$NEEDROLL"
  for s in $NEEDROLL; do
    att=0
    while :; do
      att=$((att+1))
      rm -rf "$TFL/dump_s$s"; rm -f "$TFL/roll_s$s.json"
      mark ROLLSTART "seed $s attempt $att"
      "$PY" -X utf8 "$ROLLPY" --ckpt "$CKPT" --config "$CFG" --seed "$s" --dump-dir "$TFW/dump_s$s" --out-json "$TFW/roll_s$s.json" --device cuda \
          >> "$ROOTL/logs/trainfit_pf_roll_s$s.log" 2>&1
      if roll_ok "$s"; then mark ROLL_s${s}_OK "$(jget "$TFL/roll_s$s.json" "'wall_s '+str(j.get('wall_s'))+' windows '+str(j.get('n_windows'))+' args_known '+str(j['pf']['census']['n_args_known'])+' token_differs '+str(j['pf']['census']['n_token_differs_from_clip'])")"; break; fi
      if [ "$att" -ge 2 ]; then mark ROLLGIVEUP "seed $s failed twice (see $ROOTW/logs/trainfit_pf_roll_s$s.log): HALT, no freeze"; exit 20; fi
      mark ROLLRETRY "seed $s: artifact not ok after attempt $att; retrying once"
    done
  done
  gpu_release
fi
# ---- FEED PARITY on the roll artifacts (independent of the overlay) ----------------------------------------------------------------- #
for s in 0 1; do
  PJ="$OUTL/FEED_PARITY_train_s$s.json"
  if jq_ok "$PJ" "j.get('status')=='PASS'"; then mark PARITYHAVE "seed $s"; continue; fi
  "$PY" -I -X utf8 "$PARPY" --pf-dump "$TFW/dump_s$s" --clip-dump "$CLIPREF_W/dump_s$s" --v9 "$V9" --v9-md5 "$V9MD5" --out "$OUTW/FEED_PARITY_train_s$s.json" \
      --label "train139 EMA-30k seed $s" >> "$ROOTL/logs/trainfit_pf_parity.log" 2>&1
  if jq_ok "$PJ" "j.get('status')=='PASS'"; then mark PARITY_s${s}_PASS "$(jget "$PJ" "j['counts']")"
  else mark PARITYFAIL "seed $s: $(jget "$PJ" "str(j.get('status'))+' '+str([k for k,v in j.get('checks',{}).items() if not v])+' '+str(j.get('error',''))"): HALT, no freeze"; exit 20; fi
done
# ---- CPU: panels -> freeze -> report ------------------------------------------------------------------------------------------------- #
cpu() { mem_gate "$CPUNEED" "$1" || return 9; shift; ( cd "$R8CODE" && env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$R8CODE" "$PY" -X utf8 "$@" ) >> "$ROOTL/logs/trainfit_pf_cpu.log" 2>&1; }
panels_ok() { for s in 0 1; do [ -s "$TFL/panel_s$s/manifest.json" ] && [ -s "$TFL/s6_s$s/manifest.json" ] || return 1; done; return 0; }
if panels_ok; then mark PANELSHAVE ""; else cpu panels "$FITPY" panels; if panels_ok; then mark PANELSOK ""; else mark PANELSFAIL "see $ROOTW/logs/trainfit_pf_cpu.log: HALT"; exit 20; fi; fi
FZ="$OUTL/FROZEN_GAINS_RETIME_train_pf.json"; FZS="$OUTL/FROZEN_GAINS_RETIME_train_pf.sha256"; FZP="$OUTL/FROZEN_GAINS_RETIME_train_pf.provenance.json"
frozen_ok() {
  [ -s "$FZ" ] && [ -s "$FZS" ] && [ -s "$FZP" ] || return 1
  _a=$(sha256sum "$FZ" 2>/dev/null | cut -d' ' -f1); _b=$(cut -d' ' -f1 < "$FZS" | head -n 1 | tr -d '\r ')
  [ "${#_a}" -eq 64 ] && [ "${#_b}" -eq 64 ] && [ "$_a" = "$_b" ]
}
if frozen_ok; then mark FROZENHAVE "$(cut -d' ' -f1 < "$FZS")"
else
  cpu freeze "$FITPY" freeze --parity-json "$OUTW/FEED_PARITY_train_s0.json" "$OUTW/FEED_PARITY_train_s1.json"
  if frozen_ok; then mark FROZEN "sha256 $(cut -d' ' -f1 < "$FZS") -> $OUTW/FROZEN_GAINS_RETIME_train_pf.json"
  else mark FREEZEFAIL "no self-consistent frozen pf table (see $ROOTW/logs/trainfit_pf_cpu.log and $OUTW/FREEZE_PF_REFUSED.json): HALT"; exit 20; fi
fi
"$PY" -I -X utf8 "$HOLDW/code/trainfit_pf_report.py" --tf-dir "$OUTW" --out "$HOLDW/TRAINFIT_PF_FROZEN.md" >> "$ROOTL/logs/trainfit_pf_cpu.log" 2>&1
if [ -s "$HOLDL/TRAINFIT_PF_FROZEN.md" ] && grep -q "$(cut -d' ' -f1 < "$FZS")" "$HOLDL/TRAINFIT_PF_FROZEN.md"; then mark REPORTOK "$HOLDW/TRAINFIT_PF_FROZEN.md"
else mark REPORTFAIL "TRAINFIT_PF_FROZEN.md missing or does not carry the sha256"; fi
mark END "frozen pf table sha256 $(cut -d' ' -f1 < "$FZS"); the 48,852 per-frame eval rolls stay gated on C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256"

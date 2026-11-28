#!/bin/sh
# ship_after_r8.sh -- the SHIP-R8 waiter.  usage: sh ship_after_r8.sh <tree>      (e.g. step40000ema_unc | step48852ema_unc)
#   NEW file (2026-10-19; follows tf40k_after_r8.sh).  Waits until raw/retime_trainfit/FORWARD_TRAINFIT_<tree>.json exists with controls.ALL_PASS true (tf40k_after_r8.sh writes the
#   40k one, chain_final_r8.sh the 48,852 one), then runs, CPU only and memory-gated (free commit >= 5 GB):
#       2a deploy8_forward_r8.py --tree <tree>  -> raw/retime_deploy/FORWARD_DEPLOY8_<tree>.json + DEPLOY8_POS_<tree>_s{0,1}.npz   (SCORED arm RETIME_DEPLOY8, AMENDMENT 2; a SIBLING's build:
#              checked for at RUN time, if the script is not there the stage is skipped with ZZSHIPDEP8MISSINGZZ and the verdict reads INCOMPLETE)
#       2  deploy_forward_r8.py --tree <tree>   -> raw/retime_deploy/FORWARD_DEPLOY_<tree>.json     (per-surface RETIME_DEPLOY, BESIDE; controls ALL_PASS or WITHHELD)
#       3  tac_retime_r8.py     --tree <tree>   -> raw/retime_deploy/TAC_<tree>.json + .md          (SHIP-R8 clause 5, scored on RETIME_DEPLOY8 read from the npz)
#       4  ship_r8_verdict.py   --tree <tree>   -> raw/ship_r8/SHIP_R8_<tree>.json + .md            (clauses 1-7, PASS / FAIL / INCOMPLETE; clause 7 reads raw/retime_deploy/PRIOR_PARITY.json)
#   Every stage is judged by its ARTIFACT (the json exists, parses, carries the expected field), never by an exit code; a stage's artifact is deleted BEFORE the stage runs, so a
#   stale file cannot pass for this run's output.  A failed stage 2 or 3 does not stop stage 4: the verdict then reads INCOMPLETE (a terminal artifact is always produced).
#   Opaque markers ZZSHIP<NAME>ZZ -> E:/refcv8_eval_kit/chain/ship_r8.log (one line each: marker, local time, tree + detail).  Max wait 60 h (SHIP_MAXWAIT_S), then ZZSHIPTIMEOUTZZ.
#   Terminal markers: ZZSHIPDONEZZ (stage 4's artifact exists; the line carries overall=PASS|FAIL|INCOMPLETE), ZZSHIPVERDICTFAILZZ (no verdict artifact), ZZSHIPTIMEOUTZZ; ZZSHIPENDZZ always last.
#   deploy8_forward_r8.py (stage 2a) loads `tanitad.retime_stage` from $RETIME_STAGE_FILE, else the landed stack, else its hold copy: set RETIME_STAGE_FILE in the environment the waiter is LAUNCHED from
#   (it passes through), or SHIP_PYPATH_PREFIX=<stack overlay dir> (prepended to PYTHONPATH), if the lgs snapshot at fb47a58 does not carry the module.
#   Test stubs (env): R8_PKGW / R8_PKGL (package, Windows / msys spelling), R8_KIT, R8_PY, SHIP_LOG, SHIP_OUT, SHIP_POLL_S, SHIP_MAXWAIT_S, SHIP_NEED_GB, SHIP_FREE_GB_STUB (pretend free commit).
PKGW="${R8_PKGW:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
PKGL="${R8_PKGL:-/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
KIT="${R8_KIT:-/e/refcv8_eval_kit}"
PY="${R8_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
T="${1:?usage: sh ship_after_r8.sh <tree>}"
LOG="${SHIP_LOG:-$KIT/chain/ship_r8.log}"
LG="${SHIP_OUT:-$KIT/chain/ship_$T.out}"
POLL="${SHIP_POLL_S:-120}"; MAXWAIT="${SHIP_MAXWAIT_S:-216000}"; NEED="${SHIP_NEED_GB:-5}"
TFJ="$PKGL/raw/retime_trainfit/FORWARD_TRAINFIT_$T.json"
TFW="$PKGL/raw/retime_trainfit/WITHHELD_$T.json"
DEPD="$PKGL/raw/retime_deploy"
DEPJ="$DEPD/FORWARD_DEPLOY_$T.json"
D8J="$DEPD/FORWARD_DEPLOY8_$T.json"
TACJ="$DEPD/TAC_$T.json"
SHIPJ="$PKGL/raw/ship_r8/SHIP_R8_$T.json"
mark() { echo "ZZSHIP$1ZZ $(date +%FT%T%z) $T $2" >> "$LOG"; }
free_gb() {
  if [ -n "$SHIP_FREE_GB_STUB" ]; then echo "$SHIP_FREE_GB_STUB"; return; fi
  v=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" 2>/dev/null | tr -d '\r ' | tr ',' '.')
  case "$v" in ''|*[!0-9.]*) echo 0 ;; *) echo "$v" ;; esac
}
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 >= b+0)}'; }
jq_ok() { ( cd "$PKGL" && "$PY" -X utf8 -c "import json,sys; j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)" "${1#$PKGL/}" ) 2>/dev/null; }
jq_get() { ( cd "$PKGL" && "$PY" -X utf8 -c "import json,sys; j=json.load(open(sys.argv[1],encoding='utf-8')); print($2)" "${1#$PKGL/}" ) 2>/dev/null | tr -d '\r'; }
gate_memory() {   # waits for free commit >= NEED twice, 20 s apart; returns 1 on timeout
  _n=0
  while :; do
    h1=$(free_gb); if ge "$h1" "$NEED"; then sleep "${SHIP_CONFIRM_S:-20}"; h2=$(free_gb); ge "$h2" "$NEED" && return 0; fi
    _n=$((_n+1)); [ $((_n % 20)) -eq 1 ] && mark MEMWAIT "free commit ${h1} GB < $NEED"
    [ $(( $(date +%s) - _t0 )) -gt "$MAXWAIT" ] && return 1
    sleep "${SHIP_MEMPOLL_S:-30}"
  done
}
run_py() {   # $1.. = script + args, run in the package's code dir, CPU only; output to $LG; the exit status is informational only (artifacts decide)
  ( cd "$PKGW/code" && env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 \
      PYTHONPATH="${SHIP_PYPATH_PREFIX:+$SHIP_PYPATH_PREFIX;}C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKGW/code" "$PY" -X utf8 "$@" ) >> "$LG" 2>&1
}

mark START "pid=$$ waiting for FORWARD_TRAINFIT_$T.json (controls ALL_PASS); max ${MAXWAIT}s"
_t0=$(date +%s); _n=0; withheld=0; _w0=""
while :; do
  if jq_ok "$TFJ" "j.get('controls',{}).get('ALL_PASS') is True"; then break; fi
  if [ -s "$TFW" ] && [ ! -s "$TFJ" ]; then   # a WITHHELD file may be debris of an attempt that the chain is re-running: give it a grace period before reading it as final
    [ -z "$_w0" ] && { _w0=$(date +%s); mark TFWITHHELDSEEN "WITHHELD_$T.json present and no FORWARD_TRAINFIT file: grace ${SHIP_WITHHELD_GRACE_S:-1800}s before stage 4 reads INCOMPLETE"; }
    [ $(( $(date +%s) - _w0 )) -ge "${SHIP_WITHHELD_GRACE_S:-1800}" ] && { withheld=1; break; }
  else _w0=""; fi
  _n=$((_n+1)); [ $((_n % 30)) -eq 1 ] && mark WAIT "no FORWARD_TRAINFIT_$T.json with controls ALL_PASS yet"
  [ $(( $(date +%s) - _t0 )) -gt "$MAXWAIT" ] && { mark TIMEOUT "after ${MAXWAIT}s: FORWARD_TRAINFIT_$T.json never appeared with controls ALL_PASS"; mark END "timeout"; exit 4; }
  sleep "$POLL"
done
if [ "$withheld" = 1 ]; then
  mark TFWITHHELD "the train-fit forward was WITHHELD (a known-value control failed): stages 2 and 3 are skipped, stage 4 will read INCOMPLETE"
else
  mark TFSEEN "FORWARD_TRAINFIT_$T.json has controls ALL_PASS"
  # ---- stage 2a: the SCORED arm (AMENDMENT 2): a sibling's build, looked for at RUN time
  if [ ! -f "$PKGL/code/deploy8_forward_r8.py" ]; then
    mark DEP8MISSING "code/deploy8_forward_r8.py does not exist at run time: RETIME_DEPLOY8 cannot be computed; clauses (1)-(5) will read INCOMPLETE"
  elif gate_memory; then
    rm -f "$D8J" "$DEPD/WITHHELD_DEPLOY8_$T.json" "$DEPD/DEPLOY8_POS_${T}_s0.npz" "$DEPD/DEPLOY8_POS_${T}_s1.npz"
    mark RUNDEP8 "deploy8_forward_r8.py"
    run_py deploy8_forward_r8.py --tree "$T"
    if jq_ok "$D8J" "j.get('controls',{}).get('ALL_PASS') is True" && [ -s "$DEPD/DEPLOY8_POS_${T}_s0.npz" ] && [ -s "$DEPD/DEPLOY8_POS_${T}_s1.npz" ]; then
      mark DEP8OK "raw/retime_deploy/FORWARD_DEPLOY8_$T.json (controls ALL_PASS) + DEPLOY8_POS_${T}_s0/s1.npz"
    else mark DEP8FAIL "no FORWARD_DEPLOY8 file with controls ALL_PASS, or a DEPLOY8_POS npz is missing (see $LG)"; fi
  else mark TIMEOUT "memory gate before stage 2a"; mark END "timeout"; exit 4; fi
  # ---- stage 2: the per-surface form (beside)
  if gate_memory; then
    rm -f "$DEPJ" "$DEPD/WITHHELD_DEPLOY_$T.json"
    mark RUNDEP "deploy_forward_r8.py"
    run_py deploy_forward_r8.py --tree "$T"
    if jq_ok "$DEPJ" "j.get('controls',{}).get('ALL_PASS') is True"; then mark DEPOK "raw/retime_deploy/FORWARD_DEPLOY_$T.json (controls ALL_PASS)"
    elif [ -s "$DEPD/WITHHELD_DEPLOY_$T.json" ]; then mark DEPWITHHELD "a deploy control failed (see WITHHELD_DEPLOY_$T.json); stage 4 will read INCOMPLETE"
    else mark DEPFAIL "no FORWARD_DEPLOY file (see $LG)"; fi
  else mark TIMEOUT "memory gate before stage 2"; mark END "timeout"; exit 4; fi
  # ---- stage 3: the tactical read
  if gate_memory; then
    rm -f "$TACJ" "$DEPD/TAC_$T.md"
    mark RUNTAC "tac_retime_r8.py"
    run_py tac_retime_r8.py --tree "$T"
    if jq_ok "$TACJ" "j.get('clause5',{}).get('status') in ('PASS','FAIL','INCOMPLETE')"; then
      mark TACOK "raw/retime_deploy/TAC_$T.json clause5=$(jq_get "$TACJ" "j['clause5']['status']") controls_ALL_PASS=$(jq_get "$TACJ" "j['controls']['ALL_PASS']")"
    else mark TACFAIL "no TAC json (see $LG)"; fi
  else mark TIMEOUT "memory gate before stage 3"; mark END "timeout"; exit 4; fi
fi
# ---- stage 4: the verdict (always attempted: its artifact is the terminal product)
rm -f "$SHIPJ" "$PKGL/raw/ship_r8/SHIP_R8_$T.md"
mark RUNVERDICT "ship_r8_verdict.py"
run_py ship_r8_verdict.py --tree "$T"
if jq_ok "$SHIPJ" "j.get('overall') in ('PASS','FAIL','INCOMPLETE')"; then
  mark VERDICT "$(jq_get "$SHIPJ" "j['label']")"
  mark DONE "overall=$(jq_get "$SHIPJ" "j['overall']") per_clause=$(jq_get "$SHIPJ" "j['per_clause']") -> raw/ship_r8/SHIP_R8_$T.json"
else
  mark VERDICTFAIL "no SHIP_R8_$T.json (see $LG)"
fi
mark END "$T"

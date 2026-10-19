#!/bin/sh
# tf40k_after_r8.sh -- the TRAIN-FIT retime forward on the step-40,000 EMA tree (frozen train gains, UNCHANGED).  usage: sh tf40k_after_r8.sh [tree]
#   NEW file (2026-10-19).  chain_40k_r8.sh (live, not editable) runs the three EVAL-fit frozen forwards but not the train-fit one; this waiter adds it
#   AFTER the 40k chain's CPU lane has finished the EMA tree (ZZC40LANETREEDONEZZ <tree>), so the compliant tree json it needs exists and it never
#   competes with lane d/e/g for CPU.  CPU only, memory-gated (dev box COMMIT limit).  Markers (opaque ZZT40...ZZ) -> E:/refcv8_eval_kit/chain/tf40k.log.
#   Terminal markers: ZZT40DONEZZ (controls ALL_PASS), ZZT40CTRLFAILZZ, ZZT40WITHHELDZZ, ZZT40FAILZZ, ZZT40NOLANEZZ (the 40k chain ended without the lane), ZZT40TIMEOUTZZ.
PKGW="${R8_PKGW:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
PKGL="${R8_PKGL:-/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
KIT="${R8_KIT:-/e/refcv8_eval_kit}"
PY="${R8_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
T="${1:-step40000ema_unc}"
C40LOG="${T40_C40_LOG:-$KIT/chain/chain_40k.log}"
LOG="${T40_LOG:-$KIT/chain/tf40k.log}"
LG="$KIT/chain/tf40k_$T.out"
POLL="${T40_POLL_S:-120}"; MAXWAIT="${T40_MAXWAIT_S:-129600}"; NEED="${T40_NEED_GB:-5}"
OUT="$PKGL/raw/retime_trainfit/FORWARD_TRAINFIT_$T.json"
mark() { echo "ZZT40$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
c40_alive() { ps -ef | awk '{print $NF}' | grep -q 'chain_40k_r8[.]sh$'; }
lane_done() { [ -s "$C40LOG" ] && grep -q "^ZZC40LANETREEDONEZZ .* $T\$" "$C40LOG"; }
c40_over() { [ -s "$C40LOG" ] && grep -Eq '^ZZC40(END|ABORT)ZZ' "$C40LOG"; }
free_gb() {
  v=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" 2>/dev/null | tr -d '\r ' | tr ',' '.')
  case "$v" in ''|*[!0-9.]*) echo 0 ;; *) echo "$v" ;; esac
}
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 >= b+0)}'; }
jq_ok() { ( cd "$PKGL" && "$PY" -X utf8 -c "import json,sys; j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)" "${1#$PKGL/}" ) 2>/dev/null; }

mark START "tree=$T pid=$$"
if jq_ok "$OUT" "j.get('controls',{}).get('ALL_PASS') is True"; then mark DONE "$T already has $OUT (controls ALL_PASS)"; exit 0; fi
_t0=$(date +%s); _n=0
while ! lane_done; do
  if c40_over || ! c40_alive; then
    sleep 5; lane_done && break
    mark NOLANE "$T: chain_40k ended or is not running and never logged LANETREEDONE for $T -- run by hand once the compliant tree exists: python retime_trainfit_r8.py forward --tree $T"; exit 4
  fi
  _n=$((_n+1)); [ $((_n % 30)) -eq 1 ] && mark WAIT "$T: waiting for the 40k chain's EMA lane"
  [ $(( $(date +%s) - _t0 )) -gt "$MAXWAIT" ] && { mark TIMEOUT "$T after ${MAXWAIT}s"; exit 4; }
  sleep "$POLL"
done
mark LANESEEN "$T"
_n=0
while :; do
  h1=$(free_gb); if ge "$h1" "$NEED"; then sleep 20; h2=$(free_gb); ge "$h2" "$NEED" && break; fi
  _n=$((_n+1)); [ $((_n % 20)) -eq 1 ] && mark MEMWAIT "free commit ${h1} GB < $NEED"
  [ $(( $(date +%s) - _t0 )) -gt "$MAXWAIT" ] && { mark TIMEOUT "$T memory gate"; exit 4; }
  sleep 30
done
mark RUN "$T"
( cd "$PKGW/code" && env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 \
    PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKGW/code" "$PY" -X utf8 retime_trainfit_r8.py forward --tree "$T" ) >> "$LG" 2>&1
if jq_ok "$OUT" "j.get('controls',{}).get('ALL_PASS') is True"; then mark DONE "$T -> raw/retime_trainfit/FORWARD_TRAINFIT_$T.json"
elif [ -s "$PKGL/raw/retime_trainfit/WITHHELD_$T.json" ]; then mark WITHHELD "$T: a known-value control failed (see WITHHELD_$T.json)"
elif [ -s "$OUT" ]; then mark CTRLFAIL "$T wrote $OUT but controls are not ALL_PASS: read before quoting"
else mark FAIL "$T wrote no json (see $LG)"; fi
mark END "$T"

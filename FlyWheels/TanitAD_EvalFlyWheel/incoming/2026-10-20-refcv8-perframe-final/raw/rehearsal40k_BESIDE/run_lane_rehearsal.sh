#!/bin/sh
# 40k REHEARSAL, BESIDE, not binding.  Mirrors chain_final_pf_r8.sh lane() (memory gate, skip-if-artifact, one retry, assert on the ARTIFACT).
REH=/c/Users/Admin/refcv8_pf/rehearsal40k; REHW=C:/Users/Admin/refcv8_pf/rehearsal40k
PKGW=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
HOLDW=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe; REGF=C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256
TAG=step40000ema_pf; PFPKG=$REHW/pfpkg; R=$REH/pfpkg/raw; LOG=$REH/rehearsal.log; TT=$REH/timings.tsv
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PF_LANE_DIR=$HOLDW/code
mark() { echo "ZZREH$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
freegb() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" | tr -d '\r' | tr ',' '.'; }
lane() {
  if [ -s "$2" ]; then mark "HAVE_$1" ""; return 0; fi
  fg=$(freegb); ok=$(awk -v f="$fg" 'BEGIN{print (f>=5)?1:0}')
  [ "$ok" = 1 ] || { mark "NOMEM_$1" "free commit $fg GB < 5"; return 25; }
  _try=0
  while [ "$_try" -lt 2 ]; do
    t0=$(date +%s%N)
    env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PF_R8_CODE="$PKGW/code" PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKGW/code" \
        "$PY" -X utf8 "$REHW/rehearsal_lane.py" "$1" --pkg "$PFPKG" --tree "$TAG" --registered "$REGF" >> "$REH/logs/lane_$1.log" 2>&1
    rc=$?; t1=$(date +%s%N); ms=$(( (t1 - t0) / 1000000 ))
    printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$_try" "$rc" "$ms" "$fg" >> "$TT"
    [ -s "$2" ] && { mark "OK_$1" "$2 rc=$rc wall_ms=$ms"; return 0; }
    _try=$((_try+1))
  done
  mark "FAIL_$1" "no $2 (see logs/lane_$1.log)"; return 4
}
mark START "pid $$"
lane clamp "$R/clamp_posthoc/trees/$TAG.json"
lane compliant "$R/compliant_reread/trees/$TAG.json"
lane tf_forward "$R/retime_trainfit/FORWARD_TRAINFIT_$TAG.json"
lane d8 "$R/retime_deploy/FORWARD_DEPLOY8_$TAG.json"
lane df "$R/retime_deploy/FORWARD_DEPLOY_$TAG.json"
lane tac "$R/retime_deploy/TAC_$TAG.json"
lane parity "$R/retime_deploy/PRIOR_PARITY.json"
mark END ""

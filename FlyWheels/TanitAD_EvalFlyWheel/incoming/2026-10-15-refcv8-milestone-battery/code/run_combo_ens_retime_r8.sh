#!/bin/sh
# run_combo_ens_retime_r8.sh [tree] [eval|train] -- the ONE command for the ENS-K -> RETIME stacking read (PREREG_COMBO.md).  CPU only.  NEW file; edits nothing that a chain uses.
#   sh <PKG>/code/run_combo_ens_retime_r8.sh                              # EMA-30k, eval-fitted gains (EXPLORATORY)  -> raw/combo_ens_retime/COMBO_step30000ema_unc_eval.{json,md}
#   sh <PKG>/code/run_combo_ens_retime_r8.sh step30000ema_unc train       # EMA-30k, train-fitted gains (clean form; needs FROZEN_GAINS_RETIME_train.json)
#   sh <PKG>/code/run_combo_ens_retime_r8.sh step48852ema_unc train       # THE REGISTERED READ (needs the K-draw rolls at 48,852 + the final battery trees + the train gains)
# Memory gate: Win32_OperatingSystem.FreeVirtualMemory >= 6 GB (polled 15 s, up to GATE_TRIES, default 160 = 40 min), as the sibling scripts do.  Assert on the ARTIFACT (the .json), not the exit code.
TREE="${1:-step30000ema_unc}"
GAINS="${2:-eval}"
PKGW="${R8_PKGW:-C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery}"
PY="${R8_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKGW/code"
OUTD="$PKGW/raw/combo_ens_retime"; [ "$TREE" = "step30000ema_unc" ] && [ "$GAINS" = "eval" ] || OUTD="$PKGW/raw/combo_ens_retime/${TREE}_${GAINS}"
i=0
while [ $i -lt "${GATE_TRIES:-160}" ]; do
  fv=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" 2>/dev/null | tr -d '\r ' | tr ',' '.')
  if [ -n "$fv" ] && awk "BEGIN{exit !($fv >= 6.0)}"; then break; fi
  i=$((i+1)); sleep 15
done
[ $i -lt "${GATE_TRIES:-160}" ] || { echo "ZZCOMBOGATETIMEOUTZZ"; exit 9; }
"$PY" -X utf8 "$PKGW/code/combo_ens_retime_r8.py" run --tree "$TREE" --gains "$GAINS" --out-dir "$OUTD"
echo "ZZCOMBODONEZZ rc=$? out=$OUTD"

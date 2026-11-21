#!/bin/sh
# KNOWN-VALUE CONTROL of pf_lane.py (EvalFlyWheel 2026-10-20): run the re-pointed CPU lane on the BANKED per-clip tree step40000ema_unc with the CLIP-FED table
# (b208c9c1...) installed as the "registered" one, into a scratch package; compare_lane_control.py then requires every numeric leaf to equal the banked per-clip
# artifacts (clamp tree, compliant tree, FORWARD_TRAINFIT, FORWARD_DEPLOY8, FORWARD_DEPLOY, TAC) -- the redirect must change WHERE files go, never WHAT is computed.
# CPU only.  Markers ZZLC...ZZ to $LCW/lane_control.log.
HOLD=/c/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
HOLDW=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
R8W=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
LCW=${LCW:-C:/Users/Admin/refcv8_pf/lanectl40k}
TREE=${LC_TREE:-step40000ema_unc}
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$LCW/lane_control.log
mark() { echo "ZZLC$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mkdir -p "$LCW/pkg"
echo b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee > "$LCW/REGISTERED_CONTROL.sha256"
mark START "pid $$ tree $TREE"
run() {
  env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PF_R8_CODE="$R8W/code" \
      PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$R8W/code" \
      "$PY" -X utf8 "$HOLDW/code/pf_lane.py" "$@" >> "$LCW/lane_control.out" 2>&1
}
run install --pkg "$LCW/pkg" --table "$R8W/raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json" --registered "$LCW/REGISTERED_CONTROL.sha256"
[ -s "$LCW/pkg/PF_LANE_INSTALL.json" ] && mark INSTALLOK "" || { mark INSTALLBAD ""; exit 2; }
for st in clamp compliant tf_forward d8 df tac parity; do
  run "$st" --pkg "$LCW/pkg" --tree "$TREE" --registered "$LCW/REGISTERED_CONTROL.sha256"
  echo "$st rc=$?" >> "$LCW/lane_control.out"
  mark "STAGE_$st" "done"
done
"$PY" -X utf8 "$HOLDW/code/compare_lane_control.py" --pkg "$LCW/pkg" --banked "$R8W/raw" --tree "$TREE" --out "$HOLDW/raw/lane_control/LANE_CONTROL_$TREE.json" >> "$LCW/lane_control.out" 2>&1
[ -s "$HOLDW/raw/lane_control/LANE_CONTROL_$TREE.json" ] && mark COMPARED "$(grep -o '"status": "[A-Z]*"' "$HOLDW/raw/lane_control/LANE_CONTROL_$TREE.json" | head -1)" || mark COMPAREMISSING ""
mark END ""

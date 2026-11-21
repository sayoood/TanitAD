#!/bin/sh
# wise_ship_lane.sh <TREE> [<SRC_DIR>] -- the SHIP-R8 AMENDMENT 3 verdict for a per-frame battery tree, run in a PRIVATE pf package so the banked
# 48,852 package is never written (its PRIOR_PARITY.json is not tree-keyed).  R8-WISEFT-PAI (44eec8f): the clauses are applied UNCHANGED to the
# blend's tree with the REGISTERED per-frame RETIME table.  Control mode: TREE=step48852ema_pf (already in the archive) must reproduce the banked
# SHIP_R8_step48852ema_pf.json clause by clause before any blend verdict is read.  CPU only.  Markers ZZWSL<NAME>ZZ in $W/lane.log.
set -u
TREE=${1:?tree}; SRC=${2:-}
HOLDW=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
PKGW=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
REGF=C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256
FROZEN=$HOLDW/raw/retime_trainfit_pf/FROZEN_GAINS_RETIME_train_pf.json
ARCH=C:/Users/Admin/refcv8_eval_archive/battery
REF_CLIP=$ARCH/step40000ema_unc
EV9=E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz; EV9MD5=6b5c7f207cffc3b7eb3cd527fd433599
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=C:/Users/Admin/refcv8_pf/wise_ship/$TREE; PK=$W/pfpkg
mkdir -p "$W"; LOG="$W/lane.log"
mark() { echo "ZZWSL$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mark START "tree $TREE src ${SRC:-archive}"
if [ ! -e "$ARCH/$TREE" ]; then
  [ -n "$SRC" ] && [ -d "$SRC/$TREE" ] || { mark NOTREE "$SRC/$TREE"; exit 2; }
  cp -r "$SRC/$TREE" "$ARCH/$TREE" || { mark COPYFAIL ""; exit 2; }
  mark TREECOPIED "$ARCH/$TREE"
fi
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PF_R8_CODE="$PKGW/code" PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKGW/code"
if [ ! -s "$PK/PF_LANE_INSTALL.json" ]; then
  mkdir -p "$PK"
  "$PY" -X utf8 "$HOLDW/code/pf_lane.py" install --pkg "$PK" --table "$FROZEN" --registered "$REGF" > "$W/lane_install.log" 2>&1
fi
"$PY" -I -c "import json,sys; j=json.load(open(sys.argv[1])); sys.exit(0 if j.get('rs_table_loads_and_matches') is True and j.get('installed_sha256')=='2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80' else 1)" "$PK/PF_LANE_INSTALL.json" \
  || { mark INSTALLBAD "see lane_install.log"; exit 3; }
mark INSTALLED "$PK"
R="$PK/raw"
for st in clamp:clamp_posthoc/trees/$TREE.json compliant:compliant_reread/trees/$TREE.json tf_forward:retime_trainfit/FORWARD_TRAINFIT_$TREE.json \
          d8:retime_deploy/FORWARD_DEPLOY8_$TREE.json df:retime_deploy/FORWARD_DEPLOY_$TREE.json tac:retime_deploy/TAC_$TREE.json parity:retime_deploy/PRIOR_PARITY.json; do
  s=${st%%:*}; a=$R/${st#*:}
  [ -s "$a" ] && { mark "HAVE_$s" ""; continue; }
  "$PY" -X utf8 "$HOLDW/code/pf_lane.py" "$s" --pkg "$PK" --tree "$TREE" --registered "$REGF" >> "$W/lane_$s.log" 2>&1
  [ -s "$a" ] && mark "OK_$s" "" || { mark "FAIL_$s" "see lane_$s.log"; exit 4; }
done
mkdir -p "$W/feed_parity"
for s in 0 1; do
  "$PY" -X utf8 "$HOLDW/code/feed_parity_pf.py" --pf-dump "$ARCH/$TREE/dump_s$s" --clip-dump "$REF_CLIP/dump_s$s" --v9 "$EV9" --v9-md5 "$EV9MD5" \
      --out "$W/feed_parity/FEED_PARITY_${TREE}_s$s.json" --label "$TREE s$s vs the clip-fed step40000ema_unc dump of the same windows" --known-values 3 > "$W/parity_s$s.log" 2>&1
  mark "PARITY_s$s" "$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$W/feed_parity/FEED_PARITY_${TREE}_s$s.json" 2>/dev/null)"
done
"$PY" -X utf8 "$HOLDW/code/ship_r8_verdict_pf.py" --tree "$TREE" --pf-pkg "$PK" --registered "$REGF" --frozen "$FROZEN" --parity-dir "$W/feed_parity" \
    --gate-json "$HOLDW/raw/REGISTRATION_GATE.json" --out-dir "$W/ship_r8" > "$W/verdict.log" 2>&1
OV=$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))['overall'])" "$W/ship_r8/SHIP_R8_$TREE.json" 2>/dev/null)
[ -n "$OV" ] && mark "VERDICT_$OV" "$W/ship_r8/SHIP_R8_$TREE.json" || mark VERDICTMISSING "see verdict.log"
mark END ""

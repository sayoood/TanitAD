#!/bin/sh
# pk_ship_lane.sh <TREE> -- PREREG R8-PICKER: the SHIP-R8 AMENDMENT 3 lane for a picker tree in the PICKER archive.
# = wise_ship_lane.sh's sequence (pf_lane install / clamp / compliant / tf_forward / d8 / df / tac / parity -> feed_parity_pf x2 -> ship_r8_verdict_pf),
# UNCHANGED code, a private pf package per tree, preceded by pair_read_r8.py (run_bcand.sh's invocation) to regenerate PAIRED_READ_<tree>.json.
# The ONLY difference: the archive root is C:/Users/Admin/refcv8_pf/picker/archive (pk_lane.py re-point; verdict --archive; parity --pf-dump).
# CPU only.  Markers ZZPKL<NAME>ZZ in lane/<TREE>/lane.log.
set -u
TREE=${1:?tree}
HOLDW=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
PKGW=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
REGF=C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256
FROZEN=$HOLDW/raw/retime_trainfit_pf/FROZEN_GAINS_RETIME_train_pf.json
ARCH=C:/Users/Admin/refcv8_pf/picker/archive
REF_CLIP=C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc
EV9=E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz; EV9MD5=6b5c7f207cffc3b7eb3cd527fd433599
CFG=E:/refcv8_eval_kit/ckpt/config.json
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
P=C:/Users/Admin/refcv8_pf/picker
W=$P/lane/$TREE; PK=$W/pfpkg
mkdir -p "$W"; LOG="$W/lane.log"
mark() { echo "ZZPKL$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mark START "tree $TREE archive $ARCH"
[ -d "$ARCH/$TREE/dump_s0" ] && [ -d "$ARCH/$TREE/dump_s1" ] || { mark NOTREE "$ARCH/$TREE"; exit 2; }
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PF_R8_CODE="$PKGW/code" PK_ARCHIVE="$ARCH" R8_ARCHIVE="$ARCH" PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKGW/code"
if [ ! -s "$ARCH/$TREE/PAIRED_READ_$TREE.json" ]; then
  "$PY" -X utf8 "$PKGW/code/pair_read_r8.py" --root "$ARCH/$TREE" --clock standalone --config "$CFG" --out-json "$ARCH/$TREE/PAIRED_READ_$TREE.json" \
      --out-md "$ARCH/$TREE/PAIRED_READ_$TREE.md" > "$W/pair_read.log" 2>&1
  [ -s "$ARCH/$TREE/PAIRED_READ_$TREE.json" ] && mark PAIRED "" || { mark PAIRFAIL "see pair_read.log"; exit 5; }
fi
if [ ! -s "$PK/PF_LANE_INSTALL.json" ]; then
  mkdir -p "$PK"
  "$PY" -X utf8 "$P/code/pk_lane.py" install --pkg "$PK" --table "$FROZEN" --registered "$REGF" > "$W/lane_install.log" 2>&1
fi
"$PY" -I -c "import json,sys; j=json.load(open(sys.argv[1])); sys.exit(0 if j.get('rs_table_loads_and_matches') is True and j.get('installed_sha256')=='2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80' else 1)" "$PK/PF_LANE_INSTALL.json" \
  || { mark INSTALLBAD "see lane_install.log"; exit 3; }
mark INSTALLED "$PK"
R="$PK/raw"
for st in clamp:clamp_posthoc/trees/$TREE.json compliant:compliant_reread/trees/$TREE.json tf_forward:retime_trainfit/FORWARD_TRAINFIT_$TREE.json \
          d8:retime_deploy/FORWARD_DEPLOY8_$TREE.json df:retime_deploy/FORWARD_DEPLOY_$TREE.json tac:retime_deploy/TAC_$TREE.json parity:retime_deploy/PRIOR_PARITY.json; do
  s=${st%%:*}; a=$R/${st#*:}
  [ -s "$a" ] && { mark "HAVE_$s" ""; continue; }
  "$PY" -X utf8 "$P/code/pk_lane.py" "$s" --pkg "$PK" --tree "$TREE" --registered "$REGF" >> "$W/lane_$s.log" 2>&1
  [ -s "$a" ] && mark "OK_$s" "" || { mark "FAIL_$s" "see lane_$s.log"; exit 4; }
done
mkdir -p "$W/feed_parity"
for s in 0 1; do
  "$PY" -X utf8 "$HOLDW/code/feed_parity_pf.py" --pf-dump "$ARCH/$TREE/dump_s$s" --clip-dump "$REF_CLIP/dump_s$s" --v9 "$EV9" --v9-md5 "$EV9MD5" \
      --out "$W/feed_parity/FEED_PARITY_${TREE}_s$s.json" --label "$TREE s$s vs the clip-fed step40000ema_unc dump of the same windows" --known-values 3 > "$W/parity_s$s.log" 2>&1
  mark "PARITY_s$s" "$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$W/feed_parity/FEED_PARITY_${TREE}_s$s.json" 2>/dev/null)"
done
"$PY" -X utf8 "$HOLDW/code/ship_r8_verdict_pf.py" --tree "$TREE" --pf-pkg "$PK" --registered "$REGF" --frozen "$FROZEN" --parity-dir "$W/feed_parity" \
    --gate-json "$HOLDW/raw/REGISTRATION_GATE.json" --out-dir "$W/ship_r8" --archive "$ARCH" > "$W/verdict.log" 2>&1
OV=$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))['overall'])" "$W/ship_r8/SHIP_R8_$TREE.json" 2>/dev/null)
[ -n "$OV" ] && mark "VERDICT_$OV" "$W/ship_r8/SHIP_R8_$TREE.json" || mark VERDICTMISSING "see verdict.log"
mark END ""

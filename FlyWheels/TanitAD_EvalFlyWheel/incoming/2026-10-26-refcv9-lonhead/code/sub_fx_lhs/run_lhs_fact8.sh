#!/bin/bash
# run_lhs_fact8.sh -- LONHEAD phase 2: SHIP-R8 Amendment-5 machinery (ship_r8_sub_fx, factored_scored mode) re-run with the LHS oracle-label traces in place
# of the SELFSHARP traces, so `FACTORED_DEPLOY8` in THIS output tree == LHS_FACT8.  Derived from ship_r8_sub_fx/run_ship_r8_sub.sh (factored_scored branch):
# differences = script dir (this copy; forward diff in ship_r8_sub_forward.DIFF.txt), outputs/logs under refcv9_lonhead/fwd_lhs (--out-root / --out-dir
# passed explicitly so NOTHING is written into ship_r8_sub_fx).  CPU only.  Markers ZZRUN<..>ZZ in fwd_lhs/logs/run.log.
set -u
S=C:/Users/Admin/refcv8_pf/refcv9_lonhead/code/sub_fx_lhs
W=C:/Users/Admin/refcv8_pf/refcv9_lonhead/fwd_lhs
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
TREE=step48852ema_pf
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
       TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp
mkdir -p "$W/logs" "$W/out/ship_r8"
LOG=$W/logs/run.log
mk() { echo "ZZRUN$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
has() { "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$1"; }
TAG=factored_scored; ARM=FACTORED_DEPLOY8; TACARMS="FACTORED_DEPLOY8 RETIME_DEPLOY8 OSONLY_DEPLOY8"
FWD=$W/out/forward/FORWARD_${TAG^^}_$TREE.json
mk START "mode factored_scored (LHS traces)"
t0=$(date +%s)
rm -f "$FWD"
"$PY" -X utf8 "$S/ship_r8_sub_forward.py" --mode factored --scored --tree $TREE --out-root "$W/out" > "$W/logs/fwd.log" 2>&1
has "$FWD" "j['controls']['ALL_PASS'] is True" || { mk FWDFAIL "no ALL_PASS forward (see logs/fwd.log)"; exit 4; }
mk FWD "$FWD wall $(( $(date +%s) - t0 )) s"
for A in $TACARMS; do
  t1=$(date +%s)
  AD=$W/out/arms/$TAG/$A
  rm -f "$AD/TAC_$TREE.json" "$AD/TAC_RECORD_$TREE.json"
  "$PY" -X utf8 "$S/ship_r8_sub_tac.py" --arm-dir "$AD" --arm $A --tree $TREE > "$W/logs/tac_$A.log" 2>&1
  has "$AD/TAC_RECORD_$TREE.json" "j['arm']=='$A' and j['controls_ALL_PASS'] is True" || { mk TACFAIL "$A (see logs/tac_$A.log)"; exit 4; }
  mk TAC "$A wall $(( $(date +%s) - t1 )) s"
done
"$PY" -X utf8 "$S/ship_r8_sub_verdict.py" --arm $ARM --forward "$FWD" --arms-dir "$W/out/arms/$TAG" --out-dir "$W/out/ship_r8" > "$W/logs/verdict.log" 2>&1
has "$W/out/ship_r8/SHIP_R8_$ARM.json" "j['overall'] in ('PASS','FAIL','INCOMPLETE')" || { mk VERDICTFAIL "no SHIP_R8_$ARM.json"; exit 4; }
mk END "$ARM(=LHS_FACT8) overall $("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['overall'])" "$W/out/ship_r8/SHIP_R8_$ARM.json") wall $(( $(date +%s) - t0 )) s"

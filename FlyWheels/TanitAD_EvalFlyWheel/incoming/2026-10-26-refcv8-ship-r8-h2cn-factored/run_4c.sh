#!/bin/bash
# run_4c.sh -- SHIP-R8 AMENDMENT 4c (H2CNL_DEPLOY8) one-command read: forward (--h2cnl) -> TAC rows -> verdicts (H2CNL_DEPLOY8 + the H2CN_DEPLOY8 re-read)
# -> reproduction vs the landed out/ -> bars L1/L2/L3 + mutation.  CPU only.  No git, nothing on G:.  Everything under out_4c/ (out/ and out_4b/ are refused
# by the tools); the transient RT.forward work dir lives on F: (C: is nearly full).  NEW file (EvalFlyWheel 2026-10-26); run_ship_r8_sub.sh is untouched.
# Every step is judged on its ARTIFACT (the json exists and carries the field), never on an exit code through a pipe.  Markers ZZRUN<..>ZZ in logs/run_4c.log.
set -u
W=C:/Users/Admin/refcv8_pf/ship_r8_sub
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
TREE=step48852ema_pf
OUTR=$W/out_4c
WORKR=F:/refcv9_4c_scratch
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$W/logs" "$WORKR"
LOG=$W/logs/run_4c.log
mk() { echo "ZZRUN$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
has() { "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$1"; }
TAG=h2cn_h2cnl
FWD=$OUTR/forward/FORWARD_${TAG^^}_$TREE.json
mk START "amendment 4c"
t0=$(date +%s)
rm -f "$FWD"
"$PY" -X utf8 "$W/ship_r8_4c_forward.py" --mode h2cn --h2cnl --tree $TREE --out-root "$OUTR" --work-root "$WORKR" > "$W/logs/fwd_4c.log" 2>&1
has "$FWD" "j['controls']['ALL_PASS'] is True and j.get('h2cnl') is True" || { mk FWDFAIL "no ALL_PASS forward (see logs/fwd_4c.log; a WITHHELD_*.json names the control)"; exit 4; }
mk FWD "$FWD wall $(( $(date +%s) - t0 )) s"
for A in H2CNL_DEPLOY8 H2CNL_MUT8 H2CN_DEPLOY8 H2CN_NEAR RETIME_DEPLOY8; do
  t1=$(date +%s)
  AD=$OUTR/arms/$TAG/$A
  rm -f "$AD/TAC_$TREE.json" "$AD/TAC_RECORD_$TREE.json"
  "$PY" -X utf8 "$W/ship_r8_sub_tac.py" --arm-dir "$AD" --arm $A --tree $TREE > "$W/logs/tac_4c_$A.log" 2>&1
  has "$AD/TAC_RECORD_$TREE.json" "j['arm']=='$A' and j['controls_ALL_PASS'] is True" || { mk TACFAIL "$A (see logs/tac_4c_$A.log)"; exit 4; }
  mk TAC "$A wall $(( $(date +%s) - t1 )) s"
done
for VA in H2CNL_DEPLOY8 H2CN_DEPLOY8; do
  "$PY" -X utf8 "$W/ship_r8_4c_verdict.py" --arm $VA --forward "$FWD" --arms-dir "$OUTR/arms/$TAG" --out-dir "$OUTR/ship_r8" > "$W/logs/verdict_4c_$VA.log" 2>&1
  has "$OUTR/ship_r8/SHIP_R8_$VA.json" "j['overall'] in ('PASS','FAIL','INCOMPLETE')" || { mk VERDICTFAIL "no SHIP_R8_$VA.json"; exit 4; }
  mk VERDICT "$VA overall $("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['overall'])" "$OUTR/ship_r8/SHIP_R8_$VA.json")"
done
"$PY" -X utf8 "$W/read_4c.py" > "$W/logs/read_4c.log" 2>&1
has "$OUTR/REPRO_H2CN_DEPLOY8_vs_landed.json" "'REPRODUCED_EXACTLY' in j" || { mk REPROFAIL "no REPRO record"; exit 4; }
mk REPRO "REPRODUCED_EXACTLY $("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['REPRODUCED_EXACTLY'])" "$OUTR/REPRO_H2CN_DEPLOY8_vs_landed.json")"
"$PY" -X utf8 "$W/bars_4c.py" > "$W/logs/bars_4c.log" 2>&1
has "$OUTR/BARS_4C.json" "'verdict' in j and 'L1' in j['bars']" || { mk BARSFAIL "no BARS_4C.json (see logs/bars_4c.log)"; exit 4; }
mk END "bars $("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['verdict'])" "$OUTR/BARS_4C.json") wall $(( $(date +%s) - t0 )) s"

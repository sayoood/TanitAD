#!/bin/bash
# run_hsgate_phase2.sh -- PREREG_HSGATE PHASE 2 one-command read: forward (--hsgate) -> TAC rows -> SHIP-R8 verdict on HSGATE8 -> bars G1-G4 -> manifest.
# CPU only.  No git, nothing on G:.  Outputs under C:/Users/Admin/refcv8_pf/refcv9_hsgate/out + result; the transient RT.forward work dir lives on
# F:/refcv9_hsgate_scratch (C: is nearly full).  REFUSES unless REGISTERED_HSGATE.sha256 (written by the Master Mind) matches the frozen gate.
# Every step is judged on its ARTIFACT (the json exists and carries the field), never on an exit code through a pipe.  Markers ZZHSG<..>ZZ in logs/run_phase2.log.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_hsgate
CODE=$W/code
SUB=C:/Users/Admin/refcv8_pf/ship_r8_sub
source "$CODE/env_hsgate.sh"
TREE=step48852ema_pf
OUTR=$W/out
WORKR=F:/refcv9_hsgate_scratch/fwd_work
mkdir -p "$W/logs" "$WORKR"
LOG=$W/logs/run_phase2.log
mk() { echo "ZZHSG$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
has() { "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$1"; }
reg=$(cut -d' ' -f1 "$W/REGISTERED_HSGATE.sha256" 2>/dev/null); got=$(sha256sum "$W/HSGATE_FROZEN.json" | cut -d' ' -f1)
if [ ${#reg} -ne 64 ] || [ ${#got} -ne 64 ] || [ "$reg" != "$got" ]; then mk NOREG "registered '$reg' file '$got' (INCONCLUSIVE or MISMATCH)"; exit 3; fi
TAG=h2cn_hsgate
FWD=$OUTR/forward/FORWARD_${TAG^^}_$TREE.json
mk START "gate $got"
t0=$(date +%s)
rm -f "$FWD"
"$PY" -X utf8 "$CODE/ship_r8_hsgate_forward.py" --mode h2cn --hsgate --tree $TREE --out-root "$OUTR" --work-root "$WORKR" > "$W/logs/fwd_hsgate.log" 2>&1
has "$FWD" "j['controls']['ALL_PASS'] is True and bool(j.get('hsgate'))" || { mk FWDFAIL "no ALL_PASS forward (see logs/fwd_hsgate.log; a WITHHELD_*.json names the control)"; exit 4; }
mk FWD "$FWD wall $(( $(date +%s) - t0 )) s"
for A in HSGATE8 H2CN_DEPLOY8 FACTORED_DEPLOY8 RETIME_DEPLOY8; do
  t1=$(date +%s)
  AD=$OUTR/arms/$TAG/$A
  rm -f "$AD/TAC_$TREE.json" "$AD/TAC_RECORD_$TREE.json"
  "$PY" -X utf8 "$SUB/ship_r8_sub_tac.py" --arm-dir "$AD" --arm $A --tree $TREE > "$W/logs/tac_hsgate_$A.log" 2>&1
  has "$AD/TAC_RECORD_$TREE.json" "j['arm']=='$A' and j['controls_ALL_PASS'] is True" || { mk TACFAIL "$A (see logs/tac_hsgate_$A.log)"; exit 4; }
  mk TAC "$A wall $(( $(date +%s) - t1 )) s"
done
"$PY" -X utf8 "$CODE/ship_r8_hsgate_verdict.py" --arm HSGATE8 --forward "$FWD" --arms-dir "$OUTR/arms/$TAG" --out-dir "$OUTR/ship_r8" > "$W/logs/verdict_hsgate.log" 2>&1
has "$OUTR/ship_r8/SHIP_R8_HSGATE8.json" "j['overall'] in ('PASS','FAIL','INCOMPLETE')" || { mk VERDICTFAIL "no SHIP_R8_HSGATE8.json"; exit 4; }
mk VERDICT "HSGATE8 overall $("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['overall'])" "$OUTR/ship_r8/SHIP_R8_HSGATE8.json")"
"$PY" -X utf8 "$CODE/hsgate_bars.py" > "$W/logs/bars_hsgate.log" 2>&1
has "$W/result/HSGATE_VERDICT.json" "'word' in j and 'G4' in j['bars']" || { mk BARSFAIL "no HSGATE_VERDICT.json (see logs/bars_hsgate.log)"; exit 4; }
mk END "word $("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['word'])" "$W/result/HSGATE_VERDICT.json") wall $(( $(date +%s) - t0 )) s"

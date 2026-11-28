#!/bin/bash
# run_mdfix_phaseB.sh -- PREREG_HSGATE_MDFIX PHASE B one-command read: forward (--hsgate on the MDFIX copy) -> TAC rows -> SHIP-R8 verdict on MDFIX8 -> bars.
# CPU only.  No git, nothing on G:.  Outputs under C:/Users/Admin/refcv8_pf/refcv9_hsgate/out_mdfix + result; the RT.forward work dir on F:.
# REFUSES unless every pin of REGISTERED_MDFIX.sha256 (exact path lines; written by the Master Mind) re-hashes now.  Steps judged on their ARTIFACT, never on an
# exit code through a pipe.  Markers ZZMDF<..>ZZ in logs/run_mdfixB.log.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_hsgate
CODE=$W/code
SUB=C:/Users/Admin/refcv8_pf/ship_r8_sub
source "$CODE/env_hsgate.sh"
TREE=step48852ema_pf
OUTR=$W/out_mdfix
WORKR=F:/refcv9_hsgate_scratch/fwd_work_mdfix
mkdir -p "$W/logs" "$WORKR"
LOG=$W/logs/run_mdfixB.log
mk() { echo "ZZMDF$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
has() { "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$1"; }
"$PY" -X utf8 -c "import sys; sys.path.insert(0, r'$CODE'); import mdfix_common as MC; MC.check_prereg_mdfix(); r = MC.check_registration(); print('REGOK' if r['ALL_MATCH'] else 'REGBAD')" > "$W/logs/mdfixB_regcheck.txt" 2>&1
grep -qx "REGOK" "$W/logs/mdfixB_regcheck.txt" || { mk NOREG "registration / prereg check failed (logs/mdfixB_regcheck.txt)"; exit 3; }
TAG=h2cn_mdfix
FWD=$OUTR/forward/FORWARD_${TAG^^}_$TREE.json
mk START "pins verified"
t0=$(date +%s)
"$PY" -X utf8 "$CODE/ship_r8_mdfix_forward.py" --mode h2cn --hsgate --tree $TREE --out-root "$OUTR" --work-root "$WORKR" > "$W/logs/fwd_mdfix.log" 2>&1
has "$FWD" "j['controls']['ALL_PASS'] is True and bool(j.get('hsgate'))" || { mk FWDFAIL "no ALL_PASS forward (see logs/fwd_mdfix.log; a WITHHELD_*.json names the control)"; exit 4; }
mk FWD "$FWD wall $(( $(date +%s) - t0 )) s"
for A in MDFIX8 H2CN_DEPLOY8 FACTORED_DEPLOY8 RETIME_DEPLOY8; do
  t1=$(date +%s)
  AD=$OUTR/arms/$TAG/$A
  "$PY" -X utf8 "$SUB/ship_r8_sub_tac.py" --arm-dir "$AD" --arm $A --tree $TREE > "$W/logs/tac_mdfix_$A.log" 2>&1
  has "$AD/TAC_RECORD_$TREE.json" "j['arm']=='$A' and j['controls_ALL_PASS'] is True" || { mk TACFAIL "$A (see logs/tac_mdfix_$A.log)"; exit 4; }
  mk TAC "$A wall $(( $(date +%s) - t1 )) s"
done
"$PY" -X utf8 "$CODE/ship_r8_mdfix_verdict.py" --arm MDFIX8 --forward "$FWD" --arms-dir "$OUTR/arms/$TAG" --out-dir "$OUTR/ship_r8" > "$W/logs/verdict_mdfix.log" 2>&1
has "$OUTR/ship_r8/SHIP_R8_MDFIX8.json" "j['overall'] in ('PASS','FAIL','INCOMPLETE')" || { mk VERDICTFAIL "no SHIP_R8_MDFIX8.json"; exit 4; }
mk VERDICT "MDFIX8 overall $("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['overall'])" "$OUTR/ship_r8/SHIP_R8_MDFIX8.json")"
"$PY" -X utf8 "$CODE/mdfix_bars.py" > "$W/logs/bars_mdfix.log" 2>&1
has "$W/result/MDFIX_VERDICT.json" "'word' in j and 'G4' in j['bars']" || { mk BARSFAIL "no MDFIX_VERDICT.json (see logs/bars_mdfix.log)"; exit 4; }
mk END "word $("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['word'])" "$W/result/MDFIX_VERDICT.json") wall $(( $(date +%s) - t0 )) s"

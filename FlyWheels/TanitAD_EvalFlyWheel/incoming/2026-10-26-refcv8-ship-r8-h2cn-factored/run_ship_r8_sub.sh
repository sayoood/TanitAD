#!/bin/bash
# run_ship_r8_sub.sh -- one read of SHIP-R8 AMENDMENT 4 or 5 on the registered battery tree: forward -> TAC rows -> verdict.  CPU only.  No git, nothing on G:.
#   bash run_ship_r8_sub.sh h2cn               # Amendment 4 (needs REGISTERED_H2CN_STAGE.sha256 == sha256(H2CN_STAGE_FROZEN.pt), written by the MM at registration)
#   bash run_ship_r8_sub.sh factored_scored    # Amendment 5 scored read (needs F:/refcv9_selfsharp/trace_selfsharp_s1.npz final + SSGUARD s1 REPLICATED)
#   bash run_ship_r8_sub.sh factored           # identity / control arms only (no FACTORED_DEPLOY8)
#   bash run_ship_r8_sub.sh h2cn_cap8          # AMENDMENT 4b (H2CN_CAP8): everything under out_4b/ (the landed out/ is never touched); also re-reads H2CN_DEPLOY8 for the reproduction
# Every step is judged on its ARTIFACT (the json exists and carries the field), never on an exit code through a pipe.  Markers ZZRUN<..>ZZ in logs/run_<mode>.log.
set -u
W=C:/Users/Admin/refcv8_pf/ship_r8_sub
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
MODE=${1:?usage: run_ship_r8_sub.sh h2cn|factored|factored_scored}
TREE=step48852ema_pf
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$W/logs"
LOG=$W/logs/run_$MODE.log
mk() { echo "ZZRUN$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
has() { "$PY" -I -c "import json,sys
try:
    j=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if ($2) else 1)
except Exception: sys.exit(1)" "$1"; }
OUTR=$W/out; ARM2=""
case "$MODE" in
  h2cn) FWD_ARGS="--mode h2cn"; TAG=h2cn; ARM=H2CN_DEPLOY8; TACARMS="H2CN_DEPLOY8 H2CN_NEAR RETIME_DEPLOY8" ;;
  h2cn_cap8) OUTR=$W/out_4b; FWD_ARGS="--mode h2cn --cap8 --out-root $OUTR"; TAG=h2cn_cap8; ARM=H2CN_CAP8; ARM2=H2CN_DEPLOY8; TACARMS="H2CN_CAP8 H2CN_DEPLOY8 H2CN_NEAR RETIME_DEPLOY8" ;;
  factored) FWD_ARGS="--mode factored"; TAG=factored; ARM=""; TACARMS="RETIME_DEPLOY8 OSONLY_DEPLOY8" ;;
  factored_scored) FWD_ARGS="--mode factored --scored"; TAG=factored_scored; ARM=FACTORED_DEPLOY8; TACARMS="FACTORED_DEPLOY8 RETIME_DEPLOY8 OSONLY_DEPLOY8" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
FWD=$OUTR/forward/FORWARD_${TAG^^}_$TREE.json
mk START "mode $MODE"
t0=$(date +%s)
rm -f "$FWD"
"$PY" -X utf8 "$W/ship_r8_sub_forward.py" $FWD_ARGS --tree $TREE > "$W/logs/fwd_$MODE.log" 2>&1
has "$FWD" "j['controls']['ALL_PASS'] is True" || { mk FWDFAIL "no ALL_PASS forward (see logs/fwd_$MODE.log; a WITHHELD_*.json names the control)"; exit 4; }
mk FWD "$FWD wall $(( $(date +%s) - t0 )) s"
for A in $TACARMS; do
  t1=$(date +%s)
  AD=$OUTR/arms/$TAG/$A
  rm -f "$AD/TAC_$TREE.json" "$AD/TAC_RECORD_$TREE.json"
  "$PY" -X utf8 "$W/ship_r8_sub_tac.py" --arm-dir "$AD" --arm $A --tree $TREE > "$W/logs/tac_${MODE}_$A.log" 2>&1
  has "$AD/TAC_RECORD_$TREE.json" "j['arm']=='$A' and j['controls_ALL_PASS'] is True" || { mk TACFAIL "$A (see logs/tac_${MODE}_$A.log)"; exit 4; }
  mk TAC "$A wall $(( $(date +%s) - t1 )) s"
done
if [ -z "$ARM" ]; then
  "$PY" -X utf8 "$W/ship_r8_sub_verdict.py" --identity-only --forward "$FWD" --arms-dir "$OUTR/arms/$TAG" > "$W/logs/verdict_$MODE.log" 2>&1
  has "$W/out/ship_r8/IDENTITY_factored_$TREE.json" "'ALL_IDENTITIES_PASS' in j" || { mk VERDICTFAIL "no identity record"; exit 4; }
  mk END "identity $(grep -o 'ZZSUBIDENT_[A-Z]*ZZ' "$W/logs/verdict_$MODE.log") wall $(( $(date +%s) - t0 )) s"
else
  for VA in $ARM $ARM2; do     # ARM2 (cap8 mode only) = the H2CN_DEPLOY8 re-read from the SAME forward, for the bit-for-bit reproduction against the landed verdict
    "$PY" -X utf8 "$W/ship_r8_sub_verdict.py" --arm $VA --forward "$FWD" --arms-dir "$OUTR/arms/$TAG" --out-dir "$OUTR/ship_r8" > "$W/logs/verdict_${MODE}_$VA.log" 2>&1
    has "$OUTR/ship_r8/SHIP_R8_$VA.json" "j['overall'] in ('PASS','FAIL','INCOMPLETE')" || { mk VERDICTFAIL "no SHIP_R8_$VA.json"; exit 4; }
    mk END "$VA overall $("$PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['overall'])" "$OUTR/ship_r8/SHIP_R8_$VA.json") wall $(( $(date +%s) - t0 )) s"
  done
fi

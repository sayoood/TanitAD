#!/bin/sh
# post30k_heading_r8.sh -- CPU-only follow-up of the step-30,000 milestone: the pre-registered turn-defect read (H-R8-TURN-UNDERCORRECT-1).
#   Waits (poll 5 min, <= 30 h) until the step-30,000 battery pass dumps are COMPLETE -- asserted on the ARTIFACT
#   (`battery_summary.json` with bars + finished + r8), never on a queue exit code -- then runs `heading_drift.py` exactly as at
#   steps 15,000 / 20,000: EMA-30k against EMA-20k/15k/10.5k/5k, and RAW-30k against RAW-20k/15k/10.5k/5k.  Each arm runs as
#   soon as ITS dump is complete (the EMA pass lands hours before the raw one).  No GPU; reads only.
#   Decision it feeds (registered 2026-10-17, GOALS_AND_CLAIMS d113bb3): if the EMA sharp-turn shortfall stays ~14 % vs
#   refcv7's ~9 % at step 30,000 -> recommend the near-horizon heading/curvature loss on refined candidates (PI decision for the
#   next arm); if it closes -> a training-phase effect.
# Markers ZZP30...ZZ in E:/refcv8_eval_kit/chain/post30k_heading_r8.log.
A="${R8_BATT_ROOT_W:-C:/Users/Admin/refcv8_eval_archive/battery}"
PKGL=/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=/e/refcv8_eval_kit/chain/post30k_heading_r8.log
mark() { echo "ZZP30$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
done_pass() { "$PY" -I -c "import json,sys; s=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if (s.get('bars') and s.get('finished') and 'r8' in s) else 1)" "$A/$1/battery_summary.json" 2>/dev/null; }
run_hd() {   # $1 pass dir  $2 label  $3 tag  $4 earlier prefix
  OUTF="$PKGL/raw/heading_drift/$3.json"
  if [ -s "$OUTF" ]; then mark HAVE "$3"; return 0; fi
  mark START "$3"
  ( cd "$PKGL" && OMP_NUM_THREADS=6 "$PY" raw/heading_drift/heading_drift.py --new-root "$A/$1" --label-new "$2" \
      --earlier "$4-20k=$A/step20000${5}_unc" "$4-15k=$A/step15000${5}_unc" "$4-10.5k=$A/step10500${5}_unc" "$4-5k=$A/step5000${5}_unc" \
      --out-dir raw/heading_drift --tag "$3" > "raw/heading_drift/run_$3.log" 2>&1 )
  if [ -s "$OUTF" ]; then mark DONE "$3 -> raw/heading_drift/$3.md"; else mark FAIL "$3 wrote no json (raw/heading_drift/run_$3.log)"; fi
}
mark BEGIN "pid $$; waiting for $A/step30000ema_unc and $A/step30000_unc"
E=0; R=0; n=0
while [ $n -lt 360 ]; do
  n=$((n+1))
  if [ $E = 0 ] && done_pass step30000ema_unc; then run_hd step30000ema_unc EMA-30k HEADING_DRIFT_step30000ema EMA ema; E=1; fi
  if [ $R = 0 ] && done_pass step30000_unc; then run_hd step30000_unc RAW-30k HEADING_DRIFT_step30000raw RAW ""; R=1; fi
  [ $E = 1 ] && [ $R = 1 ] && break
  [ $((n % 12)) -eq 1 ] && mark POLL "ema=$E raw=$R"
  sleep 300
done
mark END "ema=$E raw=$R"

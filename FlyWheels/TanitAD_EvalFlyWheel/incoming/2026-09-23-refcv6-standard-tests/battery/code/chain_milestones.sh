#!/bin/sh
# Unattended chain (dev box only; Thor read-only): step 5000 (G0/A1/A2 reused) -> step 30000 (full
# G0 + A1 + A2) -> wait for the run's summary.json -> FINAL (full G0 + A1 + A2). Each tag is rendered,
# sanitized into the package and APPENDED to LANDING_READY.txt for the Master Mind to land.
# The orchestrator never holds CUDA; every GPU stage is a child behind the dev-box gate.
B=/c/Users/Admin/ev6_battery
BW='C:/Users/Admin/ev6_battery'
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
REL=FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval"
export OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $B/code
LOG=$B/raw/chain.log
bank() {   # $1 = tag
  [ -s $B/raw/$1/battery_summary.json ] || return 0
  $PY summarize_battery.py "$BW/raw/$1" 0 > $B/raw/$1/TABLES_s0.md 2>&1
  [ -s $B/raw/$1/analysis_s1.json ] && $PY summarize_battery.py "$BW/raw/$1" 1 > $B/raw/$1/TABLES_s1.md 2>&1
  $PY bank_tag.py "$BW/raw/$1" "$PKG/raw/$1" --with-dumps > $B/raw/$1/bank.log 2>&1 || { echo "ZZBANKFAIL_$1ZZ" >> $LOG; return 0; }
  { echo ""; echo "## $(date +%F) battery tag $1 (auto-banked by chain_milestones.sh; sanitized, sha12 ids)";
    for f in $(ls "$PKG/raw/$1"); do echo "$REL/raw/$1/$f"; done; } >> "$PKG/LANDING_READY.txt"
  echo "ZZBANKED_$1ZZ $(date +%FT%T)" >> $LOG
}
echo "ZZCHAINMSTARTZZ $(date +%FT%T)" >> $LOG
# ---- stage 0: the A2 wrapper probe on the kit checkpoint (reported retroactively; SPEC A2) ----
if [ "${SKIP1000A2:-0}" != "1" ] && [ ! -s $B/raw/g0_step1000_A2_wrapper_probe.json ]; then
  $PY wrapper_probe.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step1000.pt --config D:/refcv6_eval_kit/ckpt/config.json       --out "$BW/raw/g0_step1000_A2_wrapper_probe.json" --g0-a1-json "$BW/raw/g0_step1000_A1.json"       > $B/raw/g0_step1000_A2_wrapper_probe.log 2>&1
  if [ -s $B/raw/g0_step1000_A2_wrapper_probe.json ]; then
    mkdir -p "$PKG/raw/g0_step1000"
    $PY a2_table.py "$BW/raw/g0_step1000_A2_wrapper_probe.json" > $B/raw/g0_step1000_A2_table.md 2>&1
    for f in g0_step1000_A2_wrapper_probe.json g0_step1000_A2_wrapper_probe.log g0_step1000_A2_table.md; do
      $PY sanitize_for_bank.py "$BW/raw/$f" "$PKG/raw/g0_step1000" > /dev/null 2>&1
    done
    { echo ""; echo "## $(date +%F) step-1000 G0-A2 wrapper probe (auto-banked by chain_milestones.sh)";
      for f in g0_step1000_A2_wrapper_probe.json g0_step1000_A2_wrapper_probe.log g0_step1000_A2_table.md; do
        echo "$REL/raw/g0_step1000/$f"; done; } >> "$PKG/LANDING_READY.txt"
    echo "ZZA2_1000DONEZZ $(date +%FT%T)" >> $LOG
  else
    echo "ZZA2_1000NOJSONZZ $(date +%FT%T)" >> $LOG
  fi
fi
if [ "${SKIP5000:-0}" != "1" ]; then
  [ -f $B/raw/step5000/battery_summary_G0A1_stop_20260924.json ] || cp $B/raw/step5000/battery_summary.json $B/raw/step5000/battery_summary_G0A1_stop_20260924.json
  $PY run_battery.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_5000.pt --tag step5000 \
      --g0-json "$BW/raw/step5000/g0.json" --g0-a1-json "$BW/raw/step5000/g0_A1.json" \
      --g0-a2-json "$BW/raw/step5000/g0_A2_wrapper_probe.json" > $B/raw/battery_step5000.log 2>&1
  echo "ZZB5000DONEZZ $(date +%FT%T)" >> $LOG; bank step5000
fi
if [ "${SKIP30000:-0}" != "1" ]; then
  $PY run_battery.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_30000.pt --tag step30000 \
      --metrics "$BW/raw/thor_reads/metrics_20260926T0951.jsonl" > $B/raw/battery_step30000.log 2>&1
  echo "ZZB30000DONEZZ $(date +%FT%T)" >> $LOG; bank step30000
fi
out=$(sh pull_final.sh 2>&1 | tail -1)
echo "$(date +%FT%T) $out" >> $LOG
case "$out" in
  *ZZFINALOKZZ*)
    M=$(echo "$out" | sed -n 's/.*metrics \(metrics_[0-9T]*\.jsonl\).*/\1/p')
    $PY run_battery.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_final.pt --tag final \
        --metrics "$BW/raw/thor_reads/$M" > $B/raw/battery_final.log 2>&1
    echo "ZZBFINALDONEZZ $(date +%FT%T)" >> $LOG; bank final ;;
  *) echo "ZZNOFINALZZ $(date +%FT%T) $out" >> $LOG ;;
esac
echo "ZZCHAINMENDZZ $(date +%FT%T)" >> $LOG

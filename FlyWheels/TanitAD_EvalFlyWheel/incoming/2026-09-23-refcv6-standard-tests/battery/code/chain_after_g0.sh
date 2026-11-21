#!/bin/sh
# Unattended chain (dev box only; nothing runs on Thor except read-only ls/stat/md5sum/scp):
#   1. the battery on the KIT checkpoint (step 1000) = PIPELINE VALIDATION, reusing its G0 artifact;
#   2. wait for ckpt_5000 on Thor (poll every 5 min), pull it read-only (md5-verified);
#   3. the battery on step 5000 WITH its own G0 = the first real (T1, early-checkpoint) reading.
# Every GPU step goes through the dev-box gate inside run_battery.py. Markers are opaque (ZZ...ZZ).
B=/c/Users/Admin/ev6_battery
BW='C:/Users/Admin/ev6_battery'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval"
export OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $B/code
LOG=$B/raw/chain.log
echo "ZZCHAINSTARTZZ $(date +%FT%T)" >> $LOG
if [ "${SKIP_STEP1000:-0}" != "1" ]; then
  $PY run_battery.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step1000.pt --tag step1000 \
      --g0-json "$BW/raw/g0_step1000.json" --g0-a1-json "$BW/raw/g0_step1000_A1.json"       > $B/raw/battery_step1000.log 2>&1
  [ -s $B/raw/step1000/battery_summary.json ] && echo "ZZB1000SUMMARYZZ $(date +%FT%T)" >> $LOG
fi
# ---- the milestone ----
i=0
while [ $i -lt 144 ]; do
  out=$(sh pull_milestone.sh 5000 2>&1 | tail -1)
  echo "$(date +%FT%T) $out" >> $LOG
  case "$out" in *ZZPULLOK5000ZZ*) break;; *ZZPULLFAIL5000ZZ*) break;; esac
  sleep 300; i=$((i+1))
done
case "$out" in
  *ZZPULLOK5000ZZ*)
    M=$(echo "$out" | sed -n 's/.*metrics \(metrics_[0-9T]*\.jsonl\).*/\1/p')
    $PY run_battery.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_5000.pt --tag step5000 \
        --metrics "$BW/raw/thor_reads/$M" > $B/raw/battery_step5000.log 2>&1
    [ -s $B/raw/step5000/battery_summary.json ] && echo "ZZB5000SUMMARYZZ $(date +%FT%T)" >> $LOG ;;
  *) echo "ZZNOMILESTONEZZ $(date +%FT%T) $out" >> $LOG ;;
esac
echo "ZZCHAINENDZZ $(date +%FT%T)" >> $LOG

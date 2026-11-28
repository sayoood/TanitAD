#!/bin/sh
# Unattended (dev box only; Thor: read-only ls/stat/md5sum/scp):
#   wait for ckpt_5000 (poll every 5 min), pull it md5-verified, then the battery WITH its own G0 + G0-A1.
# The orchestrator waits on the GPU gate WITHOUT holding a CUDA context; every GPU stage is a child.
B=/c/Users/Admin/ev6_battery
BW='C:/Users/Admin/ev6_battery'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval"
export OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $B/code
LOG=$B/raw/chain.log
echo "ZZCHAIN5000STARTZZ $(date +%FT%T)" >> $LOG
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
echo "ZZCHAIN5000ENDZZ $(date +%FT%T)" >> $LOG

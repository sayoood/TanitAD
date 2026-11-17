#!/bin/bash
# bank_n1s10.sh -- run ON the pod by after_s10.sh: bank the two STRAT1 N1 seeds arms PRIVATE on HF; prints rc=<exit code> and the uploader's last line.
cd /root/strat1/tmp || { echo rc=9; exit 9; }
/root/venv/bin/python strat1_bank_pod_n1s10.py > bank_n1s10.log 2>&1
rc=$?
echo "rc=$rc"
tail -n 1 bank_n1s10.log

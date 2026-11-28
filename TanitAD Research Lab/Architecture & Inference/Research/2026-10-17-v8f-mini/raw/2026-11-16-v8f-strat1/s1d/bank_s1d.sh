#!/bin/bash
# bank_s1d.sh -- run ON the pod by after_s1d.sh: bank the four STRAT1 S1d arms PRIVATE on HF; prints rc=<exit code> and the uploader's last line.
cd /root/strat1/tmp || { echo rc=9; exit 9; }
/root/venv/bin/python strat1_bank_pod_s1d.py > bank_s1d.log 2>&1
rc=$?
echo "rc=$rc"
tail -n 1 bank_s1d.log

#!/bin/bash
# bank_ckpt120.sh -- run ON the pod by after_ckpt120.sh: bank the seven CKPT120 arms PRIVATE on HF; prints rc=<exit code> and the uploader's last line.
cd /root/strat1/tmp || { echo rc=9; exit 9; }
/root/venv/bin/python strat1_bank_pod_ckpt120.py > bank_ckpt120.log 2>&1
rc=$?
echo "rc=$rc"
tail -n 1 bank_ckpt120.log

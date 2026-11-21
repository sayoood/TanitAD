#!/bin/bash
# bank_c0rep.sh -- run ON the pod by after_c0rep.sh: bank the two STRAT1 C0 replicate arms PRIVATE on HF; prints rc=<exit code> and the uploader's last line.
cd /root/strat1/tmp || { echo rc=9; exit 9; }
/root/venv/bin/python strat1_bank_pod_c0rep.py > bank_c0rep.log 2>&1
rc=$?
echo "rc=$rc"
tail -n 1 bank_c0rep.log

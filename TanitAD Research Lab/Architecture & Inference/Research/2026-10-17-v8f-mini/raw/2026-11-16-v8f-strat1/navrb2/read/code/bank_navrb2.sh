#!/bin/bash
# bank_navrb2.sh -- run ON the pod: bank the 4 NAVRB2 base arms PRIVATE on HF (light: nice/ionice, no GPU, no model load); prints rc=<code> and the uploader's last line.
cd /root/strat1/tmp || { echo rc=9; exit 9; }
nice -n 15 /root/venv/bin/python strat1_bank_pod_navrb2.py > bank_navrb2.log 2>&1
rc=$?
echo "rc=$rc" > bank_navrb2.rc
echo "rc=$rc"
tail -n 1 bank_navrb2.log

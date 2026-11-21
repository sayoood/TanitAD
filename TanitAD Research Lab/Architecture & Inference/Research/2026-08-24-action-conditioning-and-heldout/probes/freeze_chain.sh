#!/usr/bin/env bash
# THE FREEZE HYPOTHESIS — is low drift a property of FROZEN encoders?
# splitp30k (frozen) reads 0.199; six trained-encoder arms read 0.359-0.679.
# n=1 on the frozen side is the exact confound shape just retracted as C164, so
# measure the SECOND frozen arm (splitfrz10k) and two more trained arms to fill
# the table. No training: every checkpoint is already local.
set -u
SPD="$(cd "$(dirname "$0")" && pwd)"; cd "$SPD" || exit 1
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
SPD_ARMS="splitfrz10k,ro128p30k,o11p30k" SPD_OUT="$SPD/freezetest.json" \
  PYTHONIOENCODING=utf-8 "$PY" latentmotion.py > freezetest.log 2>&1
echo "ZZFREEZE rc=$? ZZ"

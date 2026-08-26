#!/usr/bin/env bash
# Latent-target panel + its matched null.
#
# ⛔ THE WAIT LOOP THAT USED TO BE HERE IS GONE, AND WHY MATTERS. It polled
#     Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'null_geom|...' }
# which enumerates EVERY process INCLUDING the powershell.exe running that very
# query — whose own command line contains the searched string. **The count could
# never reach zero**, so the chain waited forever while the box sat idle and the
# sentinel reported ZZIDLE-DEV. That is the `pgrep -f` self-match trap from
# CLAUDE.md wearing a PowerShell costume: THE EMITTED TOKEN MUST BE DISJOINT FROM
# THE SEARCHED TOKEN.
#
# ⇒ Two fixes, and the second is the one that generalises:
#   1. there is nothing to wait for — launch directly when the box is free;
#   2. any future poll here uses `-Filter "Name='python.exe'"`, which WMI applies
#      BEFORE Where-Object, so the querying shell can never be in the set.
set -u
SPD="$(cd "$(dirname "$0")" && pwd)"
cd "$SPD" || exit 1
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"

PYTHONIOENCODING=utf-8 "$PY" latentmotion.py > latentmotion.log 2>&1
echo "ZZREAL-DONE rc=$? ZZ"

# the matched null, through the IDENTICAL code path — a null measured by a
# different program drifts from the panel it is meant to calibrate.
for s in 0 1 2 3; do
  SPD_NULL=1 SPD_NULL_SEED="$s" SPD_OUT="$SPD/latentnull_$s.json" \
    PYTHONIOENCODING=utf-8 "$PY" latentmotion.py > "$SPD/latentnull_$s.log" 2>&1
  echo "ZZNULL $s rc=$? ZZ"
done
echo "ZZCHAIN-COMPLETE ZZ"

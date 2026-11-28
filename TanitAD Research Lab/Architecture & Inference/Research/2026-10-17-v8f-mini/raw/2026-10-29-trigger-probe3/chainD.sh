#!/bin/bash
# chainD.sh -- the conditional C2 band, after the linear stage (chain A) has banked all linear arms for both populations.
export PATH=/usr/bin:/bin:$PATH
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
cd /e/Projects/TanitAD-artifacts/dagland/trigger_probe3 || exit 1
n=0; until [ -f raw/CHAIN_linear.done ]; do sleep 20; n=$((n+1)); [ $n -gt 600 ] && exit 1; done
echo "$(date -Iseconds) START c2band" >> raw/chain.log; "$PY" -I c2band.py >> raw/c2band.log 2>&1; echo "$(date -Iseconds) END c2band rc=$?" >> raw/chain.log
"$PY" -I tables_g.py >> raw/tables.log 2>&1; touch raw/CHAIN_c2band.done

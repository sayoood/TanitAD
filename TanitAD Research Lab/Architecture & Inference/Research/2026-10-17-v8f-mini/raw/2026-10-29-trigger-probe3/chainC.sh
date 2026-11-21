#!/bin/bash
# chainC.sh -- POST-FREEZE sensitivity arms (postfreeze_g.py); cannot move the verdict.
export PATH=/usr/bin:/bin:$PATH
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
cd /e/Projects/TanitAD-artifacts/dagland/trigger_probe3 || exit 1
echo "$(date -Iseconds) START pf_pstop" >> raw/chain.log; "$PY" -I postfreeze_g.py --pop pstop >> raw/pf_pstop.log 2>&1; echo "$(date -Iseconds) END pf_pstop rc=$?" >> raw/chain.log
n=0; until [ -f raw/pcaG_pmove.json ]; do sleep 20; n=$((n+1)); [ $n -gt 300 ] && break; done
echo "$(date -Iseconds) START pf_pmove" >> raw/chain.log; "$PY" -I postfreeze_g.py --pop pmove >> raw/pf_pmove.log 2>&1; echo "$(date -Iseconds) END pf_pmove rc=$?" >> raw/chain.log
touch raw/CHAIN_pf.done

#!/bin/bash
# post_chain.sh -- waits for chain_l1c.sh to finish (ZZL1CENDZZ) and then writes RESULT.md mechanically (l1c_report.py).  Idle-waits with a 60 s poll; gives up on an ABORT/STOP/NOLOCK marker.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_l1c; CH=$W/chain.log
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
while true; do
  if grep -q "ZZL1CENDZZ" "$CH" 2>/dev/null; then break; fi
  if grep -qE "ZZL1C(ABORT|STOP|NOLOCK)ZZ" "$CH" 2>/dev/null; then echo "ZZL1CPOSTSTOPZZ $(date +%FT%T%z) chain stopped before END; no report" >> "$CH"; exit 0; fi
  sleep 60
done
if [ -s "$W/result/VERDICT.json" ] && [ -s "$W/result/L1C_RESULT.json" ]; then
  "$PY" -X utf8 "$W/code/l1c_report.py" "$W/result" "$W/RESULT.md" > "$W/logs/report.log" 2>&1
  if [ -s "$W/RESULT.md" ]; then echo "ZZL1CREPORTZZ $(date +%FT%T%z) RESULT.md" >> "$CH"; else echo "ZZL1CREPORTFAILZZ $(date +%FT%T%z) see logs/report.log" >> "$CH"; fi
else
  echo "ZZL1CREPORTFAILZZ $(date +%FT%T%z) no VERDICT.json / L1C_RESULT.json" >> "$CH"
fi

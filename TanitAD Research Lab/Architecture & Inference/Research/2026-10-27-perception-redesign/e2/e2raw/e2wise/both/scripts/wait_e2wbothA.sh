#!/bin/bash
# DERIVED by k2dc/make_wboth_kit.py from the WISE-ENC script of the same role: WISE-BOTH = S_plan (20 prefixes) + core.encoder, alpha 0.5,
# pre-registered in e2/e2raw/PREREG_E2WISE_2026-11-17_ADDENDUM2.md (landed 8153ee7d, before any WISE-ENC result). Runs only if WISE-ENC fails (d).
# wait_e2wbothA.sh -- exits when E2-WISE-BOTH stage A ends (" DONE stage A" or " ABORT-"), or after 4 h (exit 8). Matches the sentinels exactly.
export PATH="/usr/bin:/bin:$PATH"
ST=/home/nvidia/r8ns/r9e2/runs/E2WISE_BOTH/stageA/STATUS.txt
end=$(( $(date +%s) + 14400 ))
while [ "$(date +%s)" -lt "$end" ]; do
  S=$(ssh -o ConnectTimeout=20 -o BatchMode=yes tanitad-thor "tail -n 40 $ST" < /dev/null 2>/dev/null | tr -d '\r')
  if printf '%s\n' "$S" | grep -q -e ' DONE stage A' -e ' ABORT-'; then
    echo "WBOTH-A-ENDED $(date -u +%H:%M:%SZ)"; printf '%s\n' "$S" | grep -e 'EVAL-DONE' -e 'ABORT-' -e ' DONE ' -e 'FIT-DONE' -e 'DEVDUMP-DONE' -e 'FITDUMP-DONE'
    exit 0
  fi
  sleep 300
done
echo "WBOTH-A-WAIT-TIMEOUT $(date -u +%H:%M:%SZ)"; printf '%s\n' "$S" | tail -n 4; exit 8

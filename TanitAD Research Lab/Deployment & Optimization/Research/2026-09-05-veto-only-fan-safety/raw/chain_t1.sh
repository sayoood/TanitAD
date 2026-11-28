#!/bin/sh
# P4 -- the SECONDARY endpoint (four families at T1) for the veto-only arms, queued so the
# 4060 never idles and never runs two torch processes at once.
# Waits for the T0 chain to finish, then evaluates BOTH seeds of the shipping dose: the ADE
# guard must be read against the SEED-REPLICATE floor, and one seed cannot supply one.
set -u
while ! grep -q "ZZCHAIN-COMPLETE" /c/Users/Admin/veto_run/chain.log 2>/dev/null; do
  if grep -q "ZZCHAIN-ABORT" /c/Users/Admin/veto_run/chain.log 2>/dev/null; then
    echo "ZZT1-ABORT-chain-failed-ZZ"; exit 1
  fi
  sleep 30
done
echo "ZZT1-CHAIN-DONE-$(date -u +%H:%M:%S)Z-ZZ"
cd /c/Users/Admin/veto_run
ARMS="s0/veto200 s1/veto200" sh run_eval_veto.sh

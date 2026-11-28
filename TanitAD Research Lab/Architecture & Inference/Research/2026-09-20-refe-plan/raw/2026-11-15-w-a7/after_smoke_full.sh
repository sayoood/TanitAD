#!/usr/bin/bash
# Waits for the running smoke chain to end; starts the full run only if the smoke reached its DONE marker (no Traceback/STOP).
W=/e/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_launchkit/w_a7
SL="$W/run_smoke/w_a7.log"; FL="$W/run/w_a7.log"
mkdir -p "$W/run"
echo "$(date +%T) after_smoke_full heartbeat: waiting for smoke" >> "$FL"
until grep -q "chain end" "$SL" 2>/dev/null; do sleep 30; done
CT=$(grep "controls:" "$SL" | tail -1 | sed 's/"ii_red_arm": false//')
if grep -q "ZZW_A7 DONE SMOKE" "$SL" && ! grep -qE "Traceback|ZZW_A7 STOP" "$SL" && ! echo "$CT" | grep -q false; then
  echo "$(date +%T) smoke OK -> full run" >> "$FL"
  bash "$W/run_w_a7.sh" "$W/run"
else
  echo "$(date +%T) ZZW_A7 STOP smoke did not reach DONE SMOKE; full run NOT started (see run_smoke/w_a7.log)" >> "$FL"
fi

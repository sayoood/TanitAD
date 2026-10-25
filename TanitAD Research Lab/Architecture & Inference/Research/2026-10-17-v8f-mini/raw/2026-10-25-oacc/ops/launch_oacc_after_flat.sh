#!/usr/bin/env bash
# launch_oacc_after_flat.sh <tip export dir> <FLAT driver pid>: PREREG_V8F_OACC queued behind FLAT on box 2 (box 2 is allocated: PI-V8F-2026-10-25b).
# Waits (script polling, no model turns) for FLAT's chain to write ZZV8FMINI-DONE / -STOP; then starts OACC's guard and OACC's chain.
# It never decides whether OACC may follow FLAT: the chain's own predecessor predicate (after_decision, PROCEED / HOLD) reads FLAT's log.
# FLAT's driver gone with no marker => ABORT (its box-2 trainer may still run; FLAT's busy marker would stop OACC anyway).
L=/c/Users/Admin/box2_ship/logs; X=${1:?export dir}; FP=${2:?flat driver pid}
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
echo "$(date '+%m-%d %H:%M') waiting for FLAT (driver $FP); export $X"
for i in $(seq 1 1440); do
  grep -aq '^ZZV8FMINI-STOP\|^ZZV8FMINI-DONE' $L/chain_v8f_flat.out && break
  if ! tasklist //FI "PID eq $FP" //NH 2>/dev/null | grep -qi bash; then
    sleep 30; grep -aq '^ZZV8FMINI-STOP\|^ZZV8FMINI-DONE' $L/chain_v8f_flat.out && break
    echo "ZZOACC-LAUNCH-ABORT FLAT driver $FP gone with no marker"; exit 3
  fi
  sleep 60
done
grep -aq '^ZZV8FMINI-STOP\|^ZZV8FMINI-DONE' $L/chain_v8f_flat.out || { echo "ZZOACC-LAUNCH-ABORT waited 24 h"; exit 4; }
echo "$(date '+%m-%d %H:%M') FLAT ended: $(grep -a '^ZZV8FMINI' $L/chain_v8f_flat.out | tail -1 | tr -d '\r')"
cd "$X" || { echo "ZZOACC-LAUNCH-ABORT no export $X"; exit 2; }
echo "predicate: $(bash "./$PK/code/box2_ops/chain_v8f_oacc.sh" --classify-predecessor $L/chain_v8f_flat.out $L/v8fflat_guard.out)"
V8F_CHAIN_OUT=$L/chain_v8f_oacc.out V8F_CURRENT=$L/v8foacc_current.txt bash /c/Users/Admin/box2_ship/v8f_guard.sh < /dev/null >> $L/v8foacc_guard.out 2>&1 &
bash "./$PK/code/box2_ops/chain_v8f_oacc.sh" < /dev/null > $L/chain_v8f_oacc.out 2>&1
echo "$(date '+%m-%d %H:%M') OACC chain ended: $(grep -a '^ZZV8FMINI' $L/chain_v8f_oacc.out | tail -1 | tr -d '\r')"
wait

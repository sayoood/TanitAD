#!/usr/bin/env bash
# dry_gate.sh <arms file> <ARM> [card] : the selfcheck of ONE arm with the argv the chain would use (chain_arms.sh --dry-run), then gate_extra.py with the
# expectation strings of the arms file. Trains nothing; one CPU-heavy phase at a time (heavy.lock). Run from /workspace/v8proof/chain2.
# Reason: the gate of PR_s1 refused an expectation STRING the first time (2026-11-14); every arm line is checked here before its turn in the queue.
set -u
W=/workspace/v8proof; C2=$W/chain2; PY=/workspace/venv/bin/python
ARMS=${1:?arms file}; ARM=${2:?arm}; CARD=${3:-PREREG_V8F_PROOF_ADD18}
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "$C2" || exit 1
OUT=$W/runs/motion_dry/$ARM; mkdir -p "$OUT"
ln=$(grep -v -E '^[[:space:]]*(#|$)' "$ARMS" | tr -d '\r' | grep "^ARM|$ARM|")
[ -n "$ln" ] || { echo "ZZ DRY $ARM REFUSED: no such arm in $ARMS"; exit 1; }
IFS='|' read -r typ name tree kind seed ex cache extra expect <<< "$ln"
read -r -a EXP <<< "$expect"
T=$(C2_CARD=$CARD bash chain_arms.sh dryrun_$ARM "$ARMS" --dry-run | grep "^TRAIN $ARM: " | sed "s|^TRAIN $ARM: ||")
[ -n "$T" ] || { echo "ZZ DRY $ARM REFUSED: the chain printed no TRAIN line"; exit 1; }
read -r -a TA <<< "$T"
GA=(); skip=0
for w in "${TA[@]}"; do
  if [ $skip = 1 ]; then GA+=("$OUT/out"); skip=0; continue; fi
  GA+=("$w"); [ "$w" = "--out" ] && skip=1
done
export PYTHONPATH=$tree/stack:$tree/taniteval TREE=$tree PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
echo "[$(ts)] dry gate $ARM start"
flock -w 5400 "$W/chains/heavy.lock" timeout 900 "${GA[@]}" --selfcheck-only > "$OUT/gate.log" 2>&1; rc=$?
[ $rc = 0 ] || { echo "ZZ DRY $ARM FAIL: selfcheck rc=$rc: $(grep -a -m1 -E 'ZZDVB|REFUSED|Error' "$OUT/gate.log" | head -c 300)"; exit 1; }
grep -aq "SELFCHECK PASS" "$OUT/gate.log" || { echo "ZZ DRY $ARM FAIL: rc 0 but no SELFCHECK PASS line"; exit 1; }
if "$PY" "$C2/gate_extra.py" check --selfcheck "$OUT/out/selfcheck.json" --card "$CARD" --out "$OUT/extra_token.json" --expect "${EXP[@]}" >> "$OUT/gate.log" 2>&1; then
  echo "ZZ DRY $ARM PASS ($(ts)): selfcheck and every expectation string of the arms file"
else
  echo "ZZ DRY $ARM REFUSED: $(grep -a REFUSED "$OUT/gate.log" | tail -n 1 | head -c 400)"; exit 1
fi
exit 0

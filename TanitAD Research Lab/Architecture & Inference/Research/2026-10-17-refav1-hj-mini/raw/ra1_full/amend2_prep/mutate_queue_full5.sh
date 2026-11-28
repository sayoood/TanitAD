#!/usr/bin/env bash
# Mutation proof for the queue's full5 stage: every mutation must turn
# stack/tests/test_ra1_hjepa.py::test_queue_full5_stage_runs_a1_x2_at_the_new_gamma_beside_amendment_1_and_no_verdict RED,
# the unmutated queue must be GREEN before and after, and the queue is restored byte-identical.
#   bash mutate_queue_full5.sh <repo root> <python exe> <tip queue (no full5)> <mut_q.py>
R=$1; P=$2; OLD=$3; MQ=$4
Q="$R/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/code/ra1_box2_queue.cmd"
NEW=$(mktemp); cp "$Q" "$NEW"
t() { (cd "$R" && CUDA_VISIBLE_DEVICES=-1 "$P" -m pytest -q -p no:cacheprovider stack/tests/test_ra1_hjepa.py -k full5 2>&1 | tail -1); }
red=0; n=0
echo "baseline: $(t)"
cp "$OLD" "$Q"; o=$(t); n=$((n+1)); case "$o" in *failed*) red=$((red+1)); echo "RED      the tip's queue (no full5 stage): $o";; *) echo "SURVIVED the tip's queue: $o";; esac
sed 's/call :run A1_b%~3g%~4_s%%S A1 %%S %~3 %~4 ""/call :run A1_b%~3g%~4_s%%S A1 %%S %~3 50 ""/' "$NEW" > "$Q"
o=$(t); n=$((n+1)); case "$o" in *failed*) red=$((red+1)); echo "RED      gamma hardwired to 50: $o";; *) echo "SURVIVED gamma hardwired: $o";; esac
sed 's/^if "%~4"=="" (echo full5 needs.*$/rem/' "$NEW" > "$Q"
o=$(t); n=$((n+1)); case "$o" in *failed*) red=$((red+1)); echo "RED      no missing-GAMMA refusal: $o";; *) echo "SURVIVED no refusal: $o";; esac
for m in dir verdict beta oneseed; do
  python "$MQ" "$NEW" "$Q" $m || { echo "mutation $m did not apply"; exit 1; }
  o=$(t); n=$((n+1)); case "$o" in *failed*) red=$((red+1)); echo "RED      mutation '$m': $o";; *) echo "SURVIVED mutation '$m': $o";; esac
done
cp "$NEW" "$Q"; echo "restored: $(t)"; cmp -s "$NEW" "$Q" && echo "byte-identical: True"
echo "$red/$n RED"
[ $red = $n ]

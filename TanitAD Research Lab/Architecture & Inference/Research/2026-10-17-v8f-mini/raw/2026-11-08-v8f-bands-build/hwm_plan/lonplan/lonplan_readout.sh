#!/bin/bash
# lonplan_readout.sh -- AMEND_HWMPLAN_3 section 5 contrasts once the SCORE chain has ended DONE (zero training; dev box CPU, 2 threads).  Per chassis k (s0, s1):
#   S_k (lonplan_shrink.py, post-hoc from A0_k and the frozen alpha), then contrast_l1full_strat.py for A1 - A0 (+ A1 - HOLD, A0 - HOLD, and the inference-seed replicate rows),
#   A1 - S, R - A0, and lonplan_diracc.py for A0 / A1 / S / R.  The MM reads the JSONs against section 6 (the label is not computed here).
# Tier: T0 (oracle goal), a world-model diagnostic.  Estimator: paired episode-cluster bootstrap B 2,000 seed 0 (inside hwm_eval.contrast), 994 windows / 71 episodes per chassis.
# Last line of readout/readout.log: "ZZ LONREAD DONE ZZ" or "ZZ LONREAD FAIL ZZ <why>".
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
# the code of record the SCORE chain ran on (score_chain.py env): the ladder tree, Windows ;-joined paths
export WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree
export PYTHONPATH="F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
L=F:/TanitAD-scratch/actrsp/lonplan
B=$L/build
S=$L/score
O=$L/readout
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
LOG=$O/readout.log
mkdir -p "$O"
fail() { echo "ZZ LONREAD FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "[lonplan readout] start $(date -u +%FT%TZ)" >> "$LOG"
for i in $(seq 1 72); do
  t=$(tail -n 2 "$L/score_chain.log")
  printf '%s\n' "$t" | grep -q "ZZ LONSCORE CHAIN FAIL" && fail "the SCORE chain failed: $(printf '%s' "$t" | tail -n 1)"
  printf '%s\n' "$t" | grep -q "ZZ LONSCORE CHAIN DONE" && break
  [ "$i" -eq 72 ] && fail "no SCORE chain sentinel after 6 h"
  sleep 300
done
echo "$(date -u +%T) chain DONE seen" >> "$LOG"
cd "$B" || fail "no build folder"
for k in s0 s1; do
  case $k in s0) want=371c753bcc3f28f1052c0e5e7e4898c0bbbed6140477b3338c8133a3632bdaee;; s1) want=8833e2923c1781d7678f8483775487db476117e3ddd563a444d2d122334061f0;; esac
  W=$S/A0_$k
  got=$(sha256sum "$W/windows.npz" | cut -c1-64)
  [ "$got" = "$want" ] || fail "A0_$k windows sha256 $got is not the SCORE windows of $k"
  for arm in A1 R; do
    [ -f "$S/${arm}_$k/windows.npz" ] && { cmp -s "$W/windows.npz" "$S/${arm}_$k/windows.npz" || fail "${arm}_$k windows differ from A0_$k"; }
  done
  $PY -u "$WC" 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
  mkdir -p "$S/S_$k"
  $PY -u lonplan_shrink.py --plans "$S/A0_$k/plans_hwm_oracle_s0.npz" --windows "$W/windows.npz" --selection "$L/sel_$k/LAMBDA_SELECTION_R35_$k.json" \
      --out "$S/S_$k/plans_hwm_oracle_s0.npz" >> "$LOG" 2>&1 || fail "shrink $k"
  $PY -u contrast_l1full_strat.py --windows "$W" --a "$S/A1_$k/plans_hwm_oracle_s0.npz" --b "$S/A0_$k/plans_hwm_oracle_s0.npz" \
      --rep-a "$S/A1_$k/plans_hwm_oracle_s1.npz" --rep-b "$S/A0_$k/plans_hwm_oracle_s1.npz" --out "$O/c_A1_A0_$k.json" >> "$LOG" 2>&1 || fail "contrast A1-A0 $k"
  $PY -u contrast_l1full_strat.py --windows "$W" --a "$S/A1_$k/plans_hwm_oracle_s0.npz" --b "$S/S_$k/plans_hwm_oracle_s0.npz" --out "$O/c_A1_S_$k.json" >> "$LOG" 2>&1 || fail "contrast A1-S $k"
  $PY -u contrast_l1full_strat.py --windows "$W" --a "$S/R_$k/plans_hwm_oracle_s0.npz" --b "$S/A0_$k/plans_hwm_oracle_s0.npz" --out "$O/c_R_A0_$k.json" >> "$LOG" 2>&1 || fail "contrast R-A0 $k"
  $PY -u lonplan_diracc.py --windows "$W/windows.npz" --arm "A0=$S/A0_$k/plans_hwm_oracle_s0.npz" --arm "A1=$S/A1_$k/plans_hwm_oracle_s0.npz" \
      --arm "S=$S/S_$k/plans_hwm_oracle_s0.npz" --arm "R=$S/R_$k/plans_hwm_oracle_s0.npz" --out "$O/diracc_$k.json" >> "$LOG" 2>&1 || fail "diracc $k"
  echo "$(date -u +%T) chassis $k done" >> "$LOG"
done
echo "ZZ LONREAD DONE ZZ" | tee -a "$LOG"

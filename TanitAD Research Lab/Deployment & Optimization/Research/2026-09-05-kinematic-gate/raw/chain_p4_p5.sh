#!/bin/sh
# Waits for draw A's banked npz, then runs P4 (share of the 8.56x) and P5 (the displacement
# frontier that scopes the successor). Both are ZERO GPU and ZERO model calls -- they read
# the banked tensors -- so they run beside the main panel's draw B without contending.
set -u
R=/c/Users/Admin/kingate/raw
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export TANITAD_REPO=C:/Users/Admin/refcv4b_repo
export PYTHONPATH="C:/Users/Admin/refcv4b_repo/stack;C:/Users/Admin/refcv4b_repo"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=3
i=0
while [ ! -s "$R/kingate_bank_drawA.npz" ] && [ $i -lt 240 ]; do i=$((i+1)); sleep 10; done
[ -s "$R/kingate_bank_drawA.npz" ] || { echo "ZZP4P5-NONPZ-ZZ"; exit 1; }
sleep 20
echo "ZZP4-START-$(date -u +%H:%M:%S)Z-ZZ"
"$PY" -u "$R/gate_share_of_gap.py" --npz "$R/kingate_bank_drawA.npz" \
  --out "$R/gate_share_of_gap.json" --n-boot 4000 --seed 11 > "$R/gate_share_of_gap.log" 2>&1
echo "ZZP4-DONE-rc$?-ZZ"
echo "ZZP5-START-$(date -u +%H:%M:%S)Z-ZZ"
"$PY" -u "$R/displacement_frontier.py" --npz "$R/kingate_bank_drawA.npz" \
  --ckpt /c/Users/Admin/rl_refcv3_min/base/ckpt_step40284_frozen.pt \
  --out "$R/displacement_frontier.json" > "$R/displacement_frontier.log" 2>&1
echo "ZZP5-DONE-rc$?-ZZ"

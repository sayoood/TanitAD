#!/bin/sh
# The T-B queue, STRICTLY SEQUENTIAL (one GPU job at a time, the dev-box rule).
#   1. zero-area mask on 2 episodes  -- the BIT-IDENTITY control
#   2. lead mask, infer-seed 0
#   3. area-matched random off-agent mask, infer-seed 0
set -e
# ⛔ `cmd | tee log` returns TEE's status, so a dead roll reads as SUCCESS and the next
# stage runs on a partial dump. MEASURED here 2026-09-16: a killed roll printed the
# queue's DONE marker. (`never-chain-the-lander-behind-a-pipe`.)
set -o pipefail
H=$(dirname "$0")
O=/c/Users/Admin/d3_out
echo "=== [1/3] zero-area mask, 2 episodes (bit-identity control) ==="
rm -rf "$O/zeromask_dump"
bash "$H/maskroll.sh" zeromask zero 0 --episodes-n 2
C:/Users/Admin/venvs/tanitad/Scripts/python.exe "$H/provgate.py" \
  /c/Users/Admin/refcv5cmp/out/refcv5-v2_dump "$O/zeromask_dump" 0,1 \
  | tee "$O/zeromask_identity.txt"
echo "=== [2/3] LEAD mask, infer-seed 0 ==="
rm -rf "$O/leadmask_dump"
bash "$H/maskroll.sh" leadmask lead 0
echo "=== [3/3] RANDOM area-matched mask, infer-seed 0 ==="
rm -rf "$O/randmask_dump"
bash "$H/maskroll.sh" randmask rand 0
echo "=== QUEUE_TB_DONE ==="

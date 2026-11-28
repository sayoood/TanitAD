#!/bin/sh
# Queue 2, STRICTLY SEQUENTIAL and strictly AFTER queue_tb.sh (one GPU job at a time).
#   4. T-A ego_zero, full roll
#   5. T-G attention attribution on the lead windows
set -e
# ⛔ `cmd | tee log` returns TEE's status, so a dead roll reads as SUCCESS and the next
# stage runs on a partial dump. MEASURED here 2026-09-16: a killed roll printed the
# queue's DONE marker. (`never-chain-the-lander-behind-a-pipe`.)
set -o pipefail
H=$(dirname "$0")
O=/c/Users/Admin/d3_out
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8
until grep -q "QUEUE_TB_DONE" "$O/queue_tb.log" 2>/dev/null; do sleep 20; done
echo "=== [4] T-A ego_zero ==="
rm -rf "$O/egozero_dump"
bash "$H/roll.sh" egozero 0 --ablate ego_zero
echo "=== [5] T-G attention ==="
export PYTHONPATH="C:\\Users\\Admin\\refcv5cmp\\repo\\stack;C:\\Users\\Admin\\refcv5cmp\\repo\\taniteval;C:\\Users\\Admin\\refcv5cmp\\repo"
cd /c/Users/Admin/refcv5cmp/repo
"$PY" "$H/t_g_attention.py" --max-windows 600 --out "$O/t_g.json" 2>&1 | tee "$O/t_g.log"
echo "=== QUEUE2_DONE ==="

#!/usr/bin/env bash
# S0 (PREREG_V7_SEED_POS §6): E-SEED-2 at the v7F launch geometry, ViT-B/16. Stages are verified by
# their ARTIFACTS, never by an exit code. Opaque ZZ tokens so a log watcher cannot match itself.
set -u
W=/c/Users/Admin/s0_b16/code
B=$W/eseed2
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="C:/Users/Admin/tipsnap/b3f7ea6f/stack"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
cd "$W" || exit 2
echo "ZZS0-START $(date '+%F %T')ZZ"

# 1. bank (CPU) -- skip if the artifact already exists
if [ ! -s "$B/frames_u8.npy" ]; then
  CUDA_VISIBLE_DEVICES="" "$PY" e_seed2_bank.py > "$B/bank.log" 2>&1
fi
[ -s "$B/frames_u8.npy" ] && [ -s "$B/labels.npy" ] && [ -s "$B/bank_meta.json" ] \
  || { echo "ZZS0-FAIL-BANK (see bank.log)ZZ"; exit 3; }
echo "ZZS0-BANK-OK $(date '+%T')ZZ"

# 2. GPU gate: the panel needs well under 1 GB, the card is shared with other streams' evals
waited=0
while :; do
  g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')
  case "$g" in ''|*[!0-9]*) g=99999 ;; esac
  if [ "$g" -le 6000 ]; then echo "ZZS0-GATE-CLEAR gpu=${g}MiB waited=${waited}sZZ"; break; fi
  if [ "$waited" -ge 14400 ]; then echo "ZZS0-GATE-TIMEOUT gpu=${g}MiB after ${waited}sZZ"; exit 4; fi
  sleep 60; waited=$((waited + 60))
done

# 3. panel (GPU, bf16, as E-SEED-2)
"$PY" e_seed2_panel.py > "$B/panel.log" 2>&1
[ -s "$B/eseed2_panel.json" ] && [ -s "$B/eseed2_preds.npz" ] \
  || { echo "ZZS0-FAIL-PANEL (see panel.log)ZZ"; exit 5; }
echo "ZZS0-PANEL-OK $(date '+%T')ZZ"

# 4. relabel -> rff (the register's binding instrument) -> table (CPU)
CUDA_VISIBLE_DEVICES="" "$PY" e_seed2_relabel.py > "$B/relabel.log" 2>&1
[ -s "$B/eseed2b_panel.json" ] || { echo "ZZS0-FAIL-RELABEL (see relabel.log)ZZ"; exit 6; }
CUDA_VISIBLE_DEVICES="" "$PY" e_seed2_rff.py > "$B/rff.log" 2>&1
[ -s "$B/eseed2c_rff.json" ] || { echo "ZZS0-FAIL-RFF (see rff.log)ZZ"; exit 7; }
CUDA_VISIBLE_DEVICES="" "$PY" e_seed2_table.py > "$B/table.log" 2>&1
echo "ZZS0-DONE $(date '+%F %T') table_rc=$?ZZ"

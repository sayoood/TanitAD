#!/usr/bin/env bash
# WP-D step 10 -- CLOSE A3: probe + exact bootstrap on the REPLICATE panel.
# ⛔ --az-sign prog EXPLICITLY (PANEL_RESULT section 5b): the convention is settled, and
# re-selecting it per run would make the replicate differ from the panel in the
# INSTRUMENT as well as in the seed.
set -u
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8
CODE=/c/Users/Admin/wpd-probe/code
BANK='C:\Users\Admin\wpd-probe\bank_rep'
RAW=/c/Users/Admin/wpd-probe/raw
ARMS=tok_D0,tok_D0b,tok_D0c,tok_D1,tok_D2
mkdir -p "$RAW"
cd "$CODE" || exit 9

for G in cart pol; do
  echo "=== PROBE $G ==="
  "$PY" b2_probe_wpd.py --bank "$BANK" --geom "$G" --az-sign prog \
        --arms "$ARMS" --out "C:\Users\Admin\wpd-probe\raw\panel_rep_${G}.json" \
        > "$RAW/probe_rep_${G}.log" 2>&1
  echo "PROBE_${G}_EXIT=$?"
  tail -20 "$RAW/probe_rep_${G}.log"

  echo "=== BOOT $G ==="
  "$PY" b5_fast_boot.py --bank "$BANK" --geom "$G" --n-boot 2000 --arms "$ARMS" \
        --out "C:\Users\Admin\wpd-probe\raw\boot_rep_${G}.json" \
        > "$RAW/boot_rep_${G}.log" 2>&1
  echo "BOOT_${G}_EXIT=$?"
  tail -25 "$RAW/boot_rep_${G}.log"
done
echo "B10_ALL_DONE"
